import warnings

warnings.filterwarnings("ignore")
import argparse
import os
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import select_device


def get_weight_size(path):
    stats = os.stat(path)
    return f"{stats.st_size / 1024 / 1024:.1f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="/home/bei-top/yolo_prune/ultralytics/runs/detect/train14/weights/best.pt", help="trained weights path")
    parser.add_argument("--batch", type=int, default=32, help="total batch size for all GPUs")
    parser.add_argument("--imgs", nargs="+", type=int, default=[640, 640], help="[height, width] image sizes")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--warmup", default=200, type=int, help="warmup time")
    parser.add_argument("--testtime", default=1000, type=int, help="test time")
    parser.add_argument("--half", action="store_true", default=False, help="fp16 mode.")
    opt = parser.parse_args()

    device = select_device(opt.device)

    # Model
    weights = opt.weights
    if weights.endswith(".pt"):
        model = load_checkpoint(weights, device=device, fuse=True)[0]
        print(f"Loaded {weights}")  # report
    else:
        model = YOLO(weights).model
        model.fuse()

    example_inputs = torch.randn((opt.batch, 3, *opt.imgs)).to(device)

    if opt.half:
        model = model.half()
        example_inputs = example_inputs.half()

    print("begin warmup...")
    for i in tqdm(range(opt.warmup), desc="warmup...."):
        model(example_inputs)

    print("begin test latency...")
    time_arr = []

    for i in tqdm(range(opt.testtime), desc="test latency...."):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        model(example_inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        time_arr.append(end_time - start_time)

    std_time = np.std(time_arr)
    infer_time_per_image = np.sum(time_arr) / (opt.testtime * opt.batch)

    if weights.endswith(".pt"):
        print(f"model weights:{opt.weights} size:{get_weight_size(opt.weights)}M (bs:{opt.batch})Latency:{infer_time_per_image:.5f}s +- {std_time:.5f}s fps:{1 / infer_time_per_image:.1f}")
    else:
        print(f"model yaml:{opt.weights} (bs:{opt.batch})Latency:{infer_time_per_image:.5f}s +- {std_time:.5f}s fps:{1 / infer_time_per_image:.1f}")
