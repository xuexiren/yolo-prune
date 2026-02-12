import argparse
import os
import types
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch_pruning as tp
import yaml
from ultralytics import YOLO
from ultralytics.models import yolo
from ultralytics.nn.modules import Detect, v10Detect
from ultralytics.nn.modules.block import PSA, AAttn, C3k2, PSABlock
from ultralytics.utils import LOGGER, colorstr

from transform_weight import transform_model, transform_yolo11_yolo26, transform_yolov8


class DetectionFinetune(yolo.detect.DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = torch.load(self.model, map_location=self.device, weights_only=False)
        model = model["ema" if model.get("ema") else "model"].float()
        for p in model.parameters():
            p.requires_grad_(True)
        LOGGER.info(colorstr("prune_model info:"))
        model.info()
        return model

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return
        model, weights = self.model, None
        self.model = self.get_model(cfg=model, weights=weights)  # calls Model(cfg, weights)


def get_ignored_layers(model):
    """Get layers to be ignored for pruning based on the model architecture."""

    def forward_for_pruning(self, x: list[torch.Tensor]) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            # x_detach = [xi.detach() for xi in x]
            x_detach = x
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    ignored_layers = []

    # 1. Define detection head attribute names to traverse (covering v8, v10, v11, v26)
    head_attrs = ["cv2", "cv3", "one2one_cv2", "one2one_cv3"]

    for m in model.modules():
        # --- Handle Detection Head (Detect) ---
        if isinstance(m, (Detect)):
            # 1.1 Monkey Patch: Replace forward method to support NMS-free detection head pruning tracing
            m.forward = types.MethodType(forward_for_pruning, m)

            # 1.2 Dynamically add detection head output layers
            for attr_name in head_attrs:
                if hasattr(m, attr_name):
                    layer_list = getattr(m, attr_name)
                    # Iterate through ModuleList (corresponding to P3, P4, P5, etc. scales)
                    if isinstance(layer_list, (list, nn.ModuleList)):
                        for head_block in layer_list:
                            # Get the last layer in Sequential (usually Conv2d), equivalent to [2] in source code
                            if isinstance(head_block, nn.Sequential):
                                ignored_layers.append(head_block[-1])
                            else:
                                ignored_layers.append(head_block)

            # 1.3 Handle DFL (Distribution Focal Loss) layer
            if hasattr(m, "dfl"):
                ignored_layers.append(m.dfl)

        elif isinstance(m, PSABlock):
            ignored_layers.append(m.attn)

        elif isinstance(m, (PSA, AAttn)):
            ignored_layers.append(m)

    return ignored_layers


def get_pruner(model, example_inputs, ignored_layers, args):
    """Initialize the pruner."""
    if args.prune_method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.prune_method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.prune_method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    else:
        raise NotImplementedError

    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=args.iterative_steps,
        pruning_ratio=1.0,
        max_pruning_ratio=1.0,
        ignored_layers=ignored_layers,
        round_to=args.round_to,
        root_module_types=[nn.Conv2d, nn.Linear],
    )
    return pruner


def run(args):

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Model Operator Translation
    # Automatically determine translation strategy (check for C3k2 module to distinguish v11/v26 from v8)
    LOGGER.info("Step 1: Model Operator Translation")
    base_yolo = YOLO(args.model_path).model
    has_c3k2 = any(isinstance(m, C3k2) for m in base_yolo.model.modules())
    trans_func = transform_yolo11_yolo26 if has_c3k2 else transform_yolov8
    transformed_path = transform_model(args.model_path, trans_func)
    LOGGER.info("-" * 50)

    LOGGER.info("Step 2: Start Pruning")
    # Load transformed model
    ckpt = torch.load(transformed_path, map_location=device, weights_only=False)
    model = ckpt["model"].float().to(device)
    # Get ignored layers and set Monkey Patch
    ignored_layers = get_ignored_layers(model)
    imgsz = args.imgsz if isinstance(args.imgsz, int) else args.imgsz[0]
    example_inputs = torch.randn(1, 3, imgsz, imgsz).to(device)
    # Get pruner instance for the current model
    pruner = get_pruner(model, example_inputs, ignored_layers, args)

    # Calculate baseline FLOPs and Params
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    target_macs = base_macs / args.speed_up
    LOGGER.info(f"Base MACs: {base_macs / 1e9:.3f} G, Params: {base_nparams / 1e6:.3f} M")
    LOGGER.info(f"Target MACs: {target_macs / 1e9:.3f} G (Speedup: {args.speed_up}x)")

    # Iterative pruning loop
    for i in range(args.iterative_steps):
        # Execute pruning step
        pruner.step()

        # Check current status
        curr_macs, curr_nparams = tp.utils.count_ops_and_params(model, example_inputs)

        # Simple logging
        if i % 10 == 0 or curr_macs <= target_macs:
            LOGGER.info(f"Iter {i}: {curr_macs / 1e9:.3f} G (Current Speedup: {base_macs / curr_macs:.2f}x)")

        # Termination conditions
        if curr_macs <= target_macs:
            LOGGER.info(f"Target speed-up reached at iter {i}.")
            break
        if pruner.current_step == pruner.iterative_steps:
            LOGGER.info("Max iterative steps reached.")
            break

    # Clean up Monkey Patch (remove dynamically bound forward method to avoid affecting subsequent fine-tuning)
    for m in model.modules():
        if isinstance(m, (Detect, v10Detect)) and "forward" in m.__dict__:
            del m.forward

    # Construct save path
    dir_name = os.path.dirname(args.model_path)
    base_name = f"pruned_model_{args.prune_method}"
    save_path = os.path.join(dir_name, f"{base_name}.pt")
    yaml_path = os.path.join(dir_name, f"{base_name}_args.yaml")  # YAML save path
    LOGGER.info("-" * 50)

    LOGGER.info("Step 3: Save Pruned Model and Configuration")

    torch.save({"model": deepcopy(model).half()}, save_path)
    args_dict = vars(args)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(args_dict, f, sort_keys=False, default_flow_style=False)
    LOGGER.info("Pruning Complete.")
    LOGGER.info(f"Final MACs: {curr_macs / 1e9:.3f} G, Params: {curr_nparams / 1e6:.3f} M")
    LOGGER.info(f"Saved to: {save_path}")
    LOGGER.info("-" * 50)
    if args.data:
        LOGGER.info("Step 4: Start Fine-tuning")
        with open(args.cfg_path, "r", encoding="utf-8") as file:
            # Use safe_load to safely convert YAML to Python dictionary
            config = yaml.safe_load(file)
        config["model"] = save_path
        config["data"] = args.data
        config["project"] = dir_name
        config["name"] = "fine-tuning"
        ft_model = DetectionFinetune(overrides=config)
        ft_model.train()
    else:
        LOGGER.warning("Skipping fine-tuning, no dataset provided.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--model_path", type=str, default="yolo26n.pt", help="Input model path (e.g., yolo11n.pt)")
    parser.add_argument("--device", type=str, default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--imgsz", type=int, default=640)

    # Pruning control parameters
    parser.add_argument("--speed_up", type=float, default=2.0, help="Target speed up ratio (e.g., 2.0 = 50% FLOPs)")
    parser.add_argument("--global_pruning", action="store_true", default=True, help="Use global pruning")
    parser.add_argument("--iterative_steps", type=int, default=200, help="Pruning steps")
    parser.add_argument("--prune_method", type=str, default="lamp", choices=["random", "l1", "lamp"])
    parser.add_argument("--round_to", type=int, default=32, help="Channel alignment (hardware friendly)")

    # Fine-tuning parameters
    parser.add_argument("--data", type=str, default="", help="Path to dataset.yaml (Required for fine-tuning)")
    parser.add_argument("--cfg_path", type=str, default="default.yaml", help="Fine-tuning cfg")
    args = parser.parse_args()

    run(args)