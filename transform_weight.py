import os

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2PSA, C2f, C3k, C3k2

from prune_module import C2f_v2, C2PSA_v2, C3k2_v2


def infer_shortcut(block):
    """Check for residual connection."""
    # If Sequential (e.g., with Attention), take the first submodule
    block = block[0] if isinstance(block, nn.Sequential) else block

    # Directly access 'add' attribute; default to False if missing
    return getattr(block, "add", False)


def transfer_weights(old, new):
    """Transfer weights."""
    new.cv2 = old.cv2
    new.m = old.m

    state_dict = old.state_dict()
    state_dict_v2 = new.state_dict()

    old_weight = state_dict["cv1.conv.weight"]
    half_channels = old_weight.shape[0] // 2
    state_dict_v2["cv0.conv.weight"] = old_weight[:half_channels]
    state_dict_v2["cv1.conv.weight"] = old_weight[half_channels:]

    for bn_key in ["weight", "bias", "running_mean", "running_var"]:
        old_bn = state_dict[f"cv1.bn.{bn_key}"]
        state_dict_v2[f"cv0.bn.{bn_key}"] = old_bn[:half_channels]
        state_dict_v2[f"cv1.bn.{bn_key}"] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith("cv1."):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(old):
        attr_value = getattr(old, attr_name)
        if not callable(attr_value) and "_" not in attr_name:
            setattr(new, attr_name, attr_value)

    new.load_state_dict(state_dict_v2)


def replace_C3k2_with_C3k2_v2(child_module):
    """Convert C3k2 operator."""
    shortcut = infer_shortcut(child_module.m[0])
    first_block = child_module.m[0]

    is_sequential = isinstance(first_block, nn.Sequential)
    if is_sequential:
        # If Sequential, attention is enabled
        actual_block = first_block[0]  # Extract internal Bottleneck to determine c3k/groups
        attn = True
    else:
        actual_block = first_block
        attn = False

    c3k = isinstance(actual_block, C3k)

    if c3k:
        g = actual_block.m[0].cv2.conv.groups
    else:
        g = actual_block.cv2.conv.groups

    c3k2_v2 = C3k2_v2(
        child_module.cv1.conv.in_channels,
        child_module.cv2.conv.out_channels,
        n=len(child_module.m),
        shortcut=shortcut,
        g=g,
        c3k=c3k,
        attn=attn,
        e=child_module.c / child_module.cv2.conv.out_channels,
    )
    return c3k2_v2


def replace_C2PSA_with_C2PSA_v2(child_module):
    """Convert C2PSA operator."""
    c2PSA_v2 = C2PSA_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels, n=len(child_module.m), e=child_module.c / child_module.cv2.conv.out_channels)
    return c2PSA_v2


def replace_C2f_with_C2f_v2(child_module):
    """Convert C2f operator."""
    shortcut = infer_shortcut(child_module.m[0])
    c2f_v2 = C2f_v2(
        child_module.cv1.conv.in_channels,
        child_module.cv2.conv.out_channels,
        n=len(child_module.m),
        shortcut=shortcut,
        g=child_module.m[0].cv2.conv.groups,
        e=child_module.c / child_module.cv2.conv.out_channels,
    )
    return c2f_v2


def transform_yolo11_yolo26(module):
    """Transform YOLO11/YOLO26 operators."""
    for name, child_module in module.named_children():
        if isinstance(child_module, C3k2):
            c3k2_v2 = replace_C3k2_with_C3k2_v2(child_module)
            transfer_weights(child_module, c3k2_v2)
            setattr(module, name, c3k2_v2)
        elif isinstance(child_module, C2PSA):
            c2PSA_v2 = replace_C2PSA_with_C2PSA_v2(child_module)
            transfer_weights(child_module, c2PSA_v2)
            setattr(module, name, c2PSA_v2)
        else:
            transform_yolo11_yolo26(child_module)


def transform_yolov8(module):
    """Transform YOLOv8 operators."""
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(
                child_module.cv1.conv.in_channels,
                child_module.cv2.conv.out_channels,
                n=len(child_module.m),
                shortcut=shortcut,
                g=child_module.m[0].cv2.conv.groups,
                e=child_module.c / child_module.cv2.conv.out_channels,
            )
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            transform_yolov8(child_module)


def transform_model(model_path, transform_method):
    """
    Model Transformation

    :param model_path: Path to the model
    :param transform_method: Method used for model transformation
    :return Path to the transformed model
    """
    model = YOLO(model_path)
    print("Starting transformation...")
    transform_method(model.model)
    print("Transformation complete.")

    # Save
    dir_name = os.path.dirname(model_path)
    save_path = os.path.join(dir_name, "transform_model.pt")
    torch.save({"model": model.model.half()}, save_path)
    print(f"Saved modified model to {save_path}")
    return save_path


if __name__ == "__main__":
    model_path = "yolo26n.pt"
    transform_model(model_path, transform_yolo11_yolo26)