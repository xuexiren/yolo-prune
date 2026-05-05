# YOLO-Pruning：适用于 YOLOv8 / YOLO11 / YOLO26 的剪枝工具 🚀

一个基于 [Ultralytics](https://github.com/ultralytics/ultralytics) 和 [Torch-Pruning](https://github.com/VainF/Torch-Pruning) 的 YOLO 模型结构化剪枝工具。

本项目专门解决 YOLO 系列模型在结构化剪枝中常见的 **通道依赖问题**。它支持对 **YOLOv8**、**YOLO11** 和 **YOLO26** 进行全自动剪枝与微调。

## ✨ 功能特性

- **自动算子转换**：
  - 自动识别并替换不利于剪枝的模块（例如将 `C2f` 转换为 `C2f_v2`，将 `C3k2` 转换为 `C3k2_v2`）。
  - 解决由 `Split` 和 `Concat` 操作导致的依赖断裂问题，确保剪枝后模型结构完整。
- **多代 YOLO 支持**：
  - ✅ **YOLOv8**（使用 `C2f` 结构）
  - ✅ **YOLO11 / YOLO26**（使用 `C3k2`、`C2PSA` 结构）
- **多种剪枝策略**：集成主流重要性评估算法，包括 `Lamp`（Layer-Adaptive Magnitude Pruning）、`L1-Norm` 和 `Random`。
- **自动微调**：剪枝后自动接入 Ultralytics 训练器，快速恢复模型精度。
- **性能评估**：内置 FPS 基准测试脚本，可直观比较剪枝前后的推理延迟和参数量缩减效果。

## 🛠️ 安装

请确保你的环境已安装 PyTorch（推荐 >= 1.13）和 CUDA。

```Bash
# 1. 克隆本仓库
git clone https://github.com/xuexiren/yolo-prune.git
cd yolo-prune

# 2. 安装依赖
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 剪枝

使用 `prune.py` 脚本完成“转换 - 剪枝 - 微调”流程。

**基本用法：**

```Bash
python prune.py --model_path yolov8n.pt --data coco8.yaml --speed_up 2.0 --prune_method lamp
```

**参数说明：**

| **参数**            | **类型** | **默认值**   | **说明**                                                     |
| ------------------- | -------- | ------------ | ------------------------------------------------------------ |
| `--model_path`      | `str`    | `yolo26n.pt` | 原始模型路径（`.pt`）。                                     |
| `--data`            | `str`    | `""`         | 数据集配置文件（例如 `coco128.yaml`）。**如果设置该参数，剪枝后会自动开始微调。** |
| `--speed_up`        | `str`    | `2.0`        | 目标加速比例（FLOPs 缩减）。例如 `2.0` 表示计算量约减少 50%。 |
| `--prune_method`    | `str`    | `lamp`       | 剪枝策略：`lamp`（推荐）、`l1`、`random`。                   |
| `--global_pruning`  | `bool`   | `True`       | 是否启用全局剪枝（跨层比较重要性）。                         |
| `--iterative_steps` | `int`    | `200`        | 迭代剪枝步数。步数越多，剪枝过程越平滑，可能带来更小的精度损失。 |
| `--imgsz`           | `int`    | `640`        | 用于校准和微调的输入图像尺寸。                               |

### 2. 基准测试（FPS）

使用 `get_fps.py` 测试模型在剪枝前后的实际推理速度。

```Bash
# 测试原始模型
python get_fps.py --weights yolov8n.pt --batch 32 --imgs 640 640

# 测试剪枝后的模型（例如保存在 runs/detect/train/weights/best.pt）
python get_fps.py --weights pruned_model_lamp.pt --batch 32 --half
```

**参数说明：**

- `--weights`：模型权重路径。
- `--batch`：推理 Batch Size（默认：32）。
- `--imgs`：测试输入尺寸。
- `--device`：运行测试的设备。
- `--testtime`：用于计算平均 FPS 的迭代次数。
- `--warmup`：GPU 预热迭代次数。
- `--half`：是否启用 FP16 半精度推理。

## 🧩 工作原理

1. **算子转换**：
   - 加载原始权重。
   - 检查模型结构（例如是否包含 `C3k2` 模块）。
   - 将 `C2f`、`C2PSA` 等模块替换为 `prune_module.py` 中定义的 `_v2` 版本。这些版本在数学上等价，但拓扑结构更适合剪枝。
2. **追踪与剪枝**：
   - 使用 `torch-pruning` 构建依赖图。
   - 自动识别并忽略 Detect Head 的末端层，避免输出形状不匹配导致解码失败。
   - 根据指定算法（例如 LAMP）计算重要性，并迭代剪除通道。
3. **微调**：
   - 使用 Ultralytics 训练引擎重新训练稀疏模型，以恢复精度。

## ⚠️ 注意事项

1. **自定义模块依赖**

   剪枝后的模型（`.pt` 文件）包含自定义模块类（例如 `C2f_v2`）。

   - 如果在**本项目目录**中加载模型，代码会自动识别这些模块。
   - 如果在**其他项目**中加载权重，必须确保该项目包含来自 `prune_module.py` 的类定义，或者将剪枝后的模型导出为 ONNX。


   ```Python
   # 在其他脚本中加载剪枝模型的示例
   from prune_module import C2f_v2, C3k2_v2, C2PSA_v2 # 必须先导入自定义类
   from ultralytics import YOLO
   
   model = YOLO("pruned_model.pt")
   ```
   
2. **导出为 ONNX**

   剪枝后的模型支持导出为 ONNX。导出后，模型不再依赖自定义 Python 代码，适合用于部署。

   ```Bash
   yolo export model=pruned_model.pt format=onnx opset=13
   ```

## 🤝 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

- [Torch-Pruning](https://github.com/VainF/Torch-Pruning)
