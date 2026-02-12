# YOLO-Pruning: Pruning for YOLOv8 / YOLO11 / YOLO26 🚀

A structured pruning tool for YOLO models based on [Ultralytics](https://github.com/ultralytics/ultralytics) and [Torch-Pruning](https://github.com/VainF/Torch-Pruning).

This project specifically addresses the common **Channel Dependency Issues** found in structured pruning of YOLO series models. It supports fully automatic pruning and fine-tuning for **YOLOv8**, **YOLO11**, and **YOLO26**.

## ✨ Features

- **Auto Operator Translation**:
  - Automatically identifies and replaces pruning-unfriendly modules (e.g., converts `C2f` to `C2f_v2`, `C3k2` to `C3k2_v2`).
  - Solves broken dependencies caused by `Split` and `Concat` operations, ensuring the structural integrity of the pruned model.
- **Multi-Generation Support**:
  - ✅ **YOLOv8** (uses `C2f` structure)
  - ✅ **YOLO11 / YOLO26** (uses `C3k2`, `C2PSA` structures)
- **Multiple Pruning Strategies**: Integrates mainstream importance estimation algorithms including `Lamp` (Layer-Adaptive Magnitude Pruning), `L1-Norm`, and `Random`.
- **Auto Fine-tuning**: Automatically attaches the Ultralytics trainer after pruning to rapidly recover model accuracy.
- **Performance Evaluation**: Built-in FPS benchmarking scripts to intuitively compare inference latency and parameter reduction before and after pruning.

## 🛠️ Installation

Please ensure your environment has PyTorch (>= 1.13 recommended) and CUDA installed.

```Bash
# 1. Clone this repository
git clone https://github.com/xuexiren/yolo-prune.git
cd yolo-prune

# 2. Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Pruning

Use the `prune.py` script to complete the "Transform - Prune - Fine-tune" pipeline.

**Basic Usage:**

```Bash
python prune.py --model_path yolov8n.pt --data coco8.yaml --speed_up 2.0 --prune_method lamp
```

**Parameter Details:**

| **Argument**        | **Type** | **Default**  | **Description**                                              |
| ------------------- | -------- | ------------ | ------------------------------------------------------------ |
| `--model_path`      | `str`    | `yolo26n.pt` | Path to the original model (`.pt`).                          |
| `--data`            | `str`    | `""`         | Dataset config (e.g., `coco128.yaml`). **If set, fine-tuning starts automatically after pruning.** |
| `--speed_up`        | `str`    | `2.0`        | Target speed-up ratio (FLOPs reduction). E.g., `2.0` implies ~50% reduction in computation. |
| `--prune_method`    | `str`    | `lamp`       | Pruning strategy: `lamp` (recommended), `l1`, `random`.      |
| `--global_pruning`  | `bool`   | `True`       | Whether to enable global pruning (compare importance across layers). |
| `--iterative_steps` | `int`    | `200`        | Steps for iterative pruning. More steps result in a smoother process and potentially less accuracy loss. |
| `--imgsz`           | `int`    | `640`        | Input image size for calibration and fine-tuning.            |

### 2. Benchmark (FPS)

Use `get_fps.py` to test the actual inference speed of the model before and after pruning.

```Bash
# Test original model
python get_fps.py --weights yolov8n.pt --batch 32 --imgs 640 640

# Test pruned model (e.g., saved in runs/detect/train/weights/best.pt)
python get_fps.py --weights pruned_model_lamp.pt --batch 32 --half
```

**Parameter Details:**

- `--weights`: Path to the model weights.
- `--batch`: Inference Batch Size (default: 32).
- `--imgs`: Input size for testing.
- `--device`: Device to run the test on.
- `--testtime`: Number of iterations to calculate average FPS.
- `--warmup`: Number of warmup iterations for the GPU.
- `--half`: Whether to enable FP16 half-precision inference.

## 🧩 How it works

1. **Operator Transformation**:
   - Loads original weights.
   - Checks the model structure (e.g., for `C3k2` modules).
   - Replaces `C2f`, `C2PSA`, etc., with `_v2` versions defined in `prune_module.py`. These versions are mathematically equivalent but topologically friendly for pruning.
2. **Tracing & Pruning**:
   - Builds a dependency graph using `torch-pruning`.
   - Automatically identifies and ignores the terminal layers of the Detect Head to prevent output shape mismatches that would break decoding.
   - Calculates importance based on the specified algorithm (e.g., LAMP) and iteratively prunes channels.
3. **Fine-tuning**:
   - Utilizes the Ultralytics training engine to retrain the sparse model and recover accuracy.

## ⚠️ Notes

1. **Custom Module Dependency**:

   The pruned model (`.pt` file) contains custom module classes (e.g., `C2f_v2`).

   - If you load the model within **this project directory**, the code will recognize it automatically.
   - If you load the weights in **another project**, you must ensure that the project contains the class definitions from `prune_module.py`, or export the pruned model to ONNX.


   ```Python
   # Example of loading the pruned model in another script
   from prune_module import C2f_v2, C3k2_v2, C2PSA_v2 # Must import custom classes first
   from ultralytics import YOLO
   
   model = YOLO("pruned_model.pt")
   ```
   
2. **Export to ONNX**:

   Pruned models support ONNX export. Once exported, the model no longer depends on the custom Python code and is suitable for deployment.

   ```Bash
   yolo export model=pruned_model.pt format=onnx opset=13
   ```

## 🤝 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

- [Torch-Pruning](https://github.com/VainF/Torch-Pruning)
