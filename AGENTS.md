# tapnet — AGENTS.md

Google DeepMind 的 Tracking Any Point (TAP) 代码库。主框架为 JAX，PyTorch 为辅助实现。

## 安装

```bash
pip install .                    # 仅推理
pip install .[torch]             # 包含 PyTorch 支持
pip install -r requirements.txt  # 训练（JAX + TF + Kubric）
```

训练还需要 `ffmpeg` 和 `libopenexr-dev`（Linux），以及 `kubric` 子目录。

需要将以下路径加入 `PYTHONPATH`：
```bash
export PYTHONPATH="$(cd ../ && pwd):$(pwd):$PYTHONPATH"
```

JAX 的 CUDA 支持需按照 [jax 手册](https://github.com/jax-ml/jax#installation) 手动安装。

## 项目结构

| 路径 | 内容 |
|---|---|
| `tapnet/models/` | JAX 模型实现（TAP-Net, TAPIR, SSM-ViT, ResNet） |
| `tapnet/torch/` | TAPIR 的 PyTorch 重实现 |
| `tapnet/tapnext/` | TAPNext 模型（PyTorch，基于下一 token 预测的跟踪器） |
| `tapnet/tapnextpp/` | TAPNext++ 微调检查点、AJ_RD 指标、数据增强 |
| `tapnet/training/` | 基于 jaxline 的训练/评估框架 |
| `tapnet/tapvid/` | TAP-Vid 数据集读取器和指标（`evaluation_datasets.py`） |
| `tapnet/tapvid3d/` | TAPVid-3D 数据集生成、评估、指标 |
| `tapnet/robotap/` | RoboTAP 聚类代码 |
| `tapnet/trajan/` | TRAJAN 轨迹自编码器 |
| `configs/` | 模型配置：`tapnet_config`、`tapir_config`、`causal_tapir_config`、`tapir_bootstrap_config` |
| `colabs/` | 11 个 Colab 笔记本，一键运行 |

## 关键命令

**实时演示**（JAX causal TAPIR）：
```bash
mkdir -p checkpoints && wget -P checkpoints https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy
python3 ./tapnet/live_demo.py
```

**训练**（jaxline experiment）：
```bash
python3 -m tapnet.training.experiment --config ./tapnet/configs/tapir_config.py
```

**评估** TAP-Vid 数据集：
```bash
python3 -m tapnet.training.experiment \
  --config=./tapnet/configs/tapir_config.py \
  --jaxline_mode=eval_davis_points \
  --config.checkpoint_dir=./tapnet/checkpoint/ \
  --config.experiment_kwargs.config.davis_points_path=/path/to/tapvid_davis.pkl
```

**推理**单段视频：
```bash
python3 -m tapnet.training.experiment \
  --config=./tapnet/configs/tapnet_config.py \
  --jaxline_mode=eval_inference \
  --config.checkpoint_dir=./tapnet/checkpoint/ \
  --config.experiment_kwargs.config.inference.input_video_path=horsejump-high.mp4 \
  --config.experiment_kwargs.config.inference.output_video_path=result.mp4 \
  --config.experiment_kwargs.config.inference.resize_height=256 \
  --config.experiment_kwargs.config.inference.resize_width=256 \
  --config.experiment_kwargs.config.inference.num_points=20
```

## 检查点

可在 HuggingFace [google/tapnet](https://huggingface.co/google/tapnet) 或 GCS 获取。通过 `NumpyFileCheckpointer` 加载（JAX 用 `.npy`/`.npz`，PyTorch 用 `.pt`）。

关键检查点下载命令：
```bash
# TAPIR
wget https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy
# TAPNext
wget https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz
# TAPNext++
wget https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt
```

## 坐标约定

- **存储格式**：`(x, y)` 归一化到 `[0, 1]`；`(0,0)` 为左上像素的左上角
- **代码（光栅）**：`(y, x)` 范围 `[0, h) x [0, w)`；`(h, w)` 为右下角
- **3D 内部**：`(t, y, x)`，其中 `t` 为帧索引（浮点数，0 表示第一帧）

## 无测试和 CI

不包含 pytest 测试套件，也没有 `.github/workflows/`。验证通过 Colab 笔记本或手动推理/评估运行。

## 遗留导入

`tapnet/__init__` 为了向后兼容暴露了 `tapir_model`、`tapnet_model`、`tapir_clustering`、`evaluation_datasets`。建议直接从子包导入。

## TAPVid-3D 额外说明

- 独立许可证（见 `tapvid3d/LICENSE`）
- 生成脚本：`python3 -m tapnet.tapvid3d.annotation_generation.generate_{adt,pstudio,drivetrack} --help`
- 需要接受 Aria Digital Twin、Waymo Open Dataset 和 Panoptic Studio 的许可协议
- Pip 安装额外依赖：`pip install .[tapvid3d_eval,tapvid3d_generation]`

## TAPNext++ 额外说明

- 重新检测 AJ 指标：`tapnet.tapnextpp.metrics.aj_rd.compute_redetection_metrics`
- 自定义数据增强：`tapnet/tapnextpp/augmentations/{roll,homography}.py`