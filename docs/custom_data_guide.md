# Robo-Dopamine 自定义数据使用指南

> 本文档基于「抓取胡萝卜放在黄色盘子上」任务的实际数据编写，涵盖从原始数据到推理、微调的完整流程。

---

## 1. 原始数据预处理

### 1.1 数据来源

```
成功案例: /home/dais/workspace/bag2video/pick_3_suc@MASTER_SLAVE_MODE@2026_05_12_23_05_32
失败案例: /home/dais/workspace/bag2video/pick_3_fail@MASTER_SLAVE_MODE@2026_05_12_23_11_53
```

每个目录包含三个相机视角的视频：

| 文件 | 说明 | 示例帧数 |
|------|------|---------|
| `cam_high.mp4` | 正上方俯视相机 | 606 / 666 |
| `cam_left_wrist.mp4` | 左腕相机 | 611 / 673 |
| `cam_right_wrist.mp4` | 右腕相机 | 612 / 673 |

视频参数：640x480, 30fps, 约 20-22 秒。

### 1.2 帧数对齐（必须）

由于三个相机录制的帧数可能存在微小差异（如 606/611/612），需要先对齐到最短的帧数，否则推理时会报错：

```
ValueError: Frame count mismatch among cameras: [606, 611, 612]
```

**对齐脚本**：

```python
import cv2
import os

SRC = "/home/dais/workspace/bag2video/pick_3_suc@MASTER_SLAVE_MODE@2026_05_12_23_05_32"
DST = "/home/dais/workspace/Robo-Dopamine/aligned_data_suc"
os.makedirs(DST, exist_ok=True)

TARGET_FRAMES = 666  # 以最短的 cam_high 为准

for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
    src = os.path.join(SRC, f"{cam}.mp4")
    dst = os.path.join(DST, f"{cam}.mp4")

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst, fourcc, fps, (w, h))

    count = 0
    while count < TARGET_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        count += 1

    cap.release()
    writer.release()
    print(f"{cam}: wrote {count} frames")
```

对齐后的数据保存在：

```
aligned_data/          # 失败案例（606帧）
aligned_data_suc/      # 成功案例（666帧）
```

---

## 2. 推理使用方式

### 2.1 模型选择

本地可用的模型路径：

```
./pretrained_models/GRM-2.0-4B-Preview    # 4B 轻量版（~9GB 显存）
# 如有网络，也可使用：
# tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview   # 8B 完整版（~18GB 显存）
# tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview   # 4B 轻量版
```

### 2.2 推理脚本

创建推理脚本 `test_my_data.py`：

```python
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from examples.inference import GRMInference

# ==================== 配置区 ====================
MODEL_PATH = "./pretrained_models/GRM-2.0-4B-Preview"   # 模型路径
DATA_DIR = "./aligned_data_suc"                          # 帧对齐后的视频目录
OUTPUT_ROOT = "./results/my_test"                        # 输出目录
TASK_INSTRUCTION = "pick the carrot and place it on the yellow plate"  # 任务描述
GOAL_IMAGE = "./examples/blank_goal.png"                 # 目标图（无则用空白图）

# ==================== 加载模型 ====================
model = GRMInference(MODEL_PATH)

# ==================== 运行推理 ====================
for mode in ["forward", "incremental", "backward"]:
    output_dir = model.run_pipeline(
        cam_high_path  = os.path.join(DATA_DIR, "cam_high.mp4"),
        cam_left_path  = os.path.join(DATA_DIR, "cam_left_wrist.mp4"),
        cam_right_path = os.path.join(DATA_DIR, "cam_right_wrist.mp4"),
        out_root       = OUTPUT_ROOT,
        task           = TASK_INSTRUCTION,
        frame_interval = 10,
        batch_size     = 1,
        goal_image     = GOAL_IMAGE,
        eval_mode      = mode,
        visualize      = True,
    )
    print(f"[{mode}] Output: {output_dir}")
```

运行：

```bash
cd /home/dais/workspace/Robo-Dopamine
conda activate robo-dopamine
python test_my_data.py
```

### 2.3 参数说明与选择建议

#### 核心参数

| 参数 | 默认值 | 说明 | 选择建议 |
|------|--------|------|---------|
| `frame_interval` | 10 | 帧采样间隔 | 10 适用于 30fps 视频（每 0.33s 采样一次）；incremental 模式下不宜太小 |
| `batch_size` | 1 | 推理批大小 | 显存充足时增大到 4-8 可加速；4B 模型单 A100 可设 4 |
| `eval_mode` | "forward" | 评估模式 | 建议三种都跑，取平均作为最终奖励 |
| `goal_image` | blank_goal.png | 目标参考图 | 有完成状态照片时提供真实图片；无则用空白图 |
| `visualize` | True | 是否生成可视化视频 | 建议开启，方便直观检查 |

#### 三种评估模式

| 模式 | 原理 | 特点 |
|------|------|------|
| **forward** | 以起始帧为基准，衡量当前帧的绝对进度 | 进度单调递增，最稳定，推荐 RL 使用 |
| **incremental** | 相邻帧之间的增量变化 | 能捕捉细微倒退，适合细粒度奖励 |
| **backward** | 从目标反推距离完成还有多远 | 与 forward 互补，提供另一个视角 |

**RL 使用建议**：三种模式的 progress 取平均作为最终 reward。

#### goal_image 参数

| 场景 | 设置 | 影响 |
|------|------|------|
| 有目标图 | `goal_image = "path/to/goal.png"` | 模型能精确锚定完成状态，分数校准更准 |
| 无目标图 | `goal_image = "./examples/blank_goal.png"` | 仍能工作，模型依赖 BEFORE/AFTER 对比判断进度 |
| 不传参数 | `goal_image = None` | 使用视频最后一帧作为 ref_end（成功案例效果好，失败案例可能误导） |

### 2.4 推理输出

输出目录结构：

```
results/my_test/26-XX-XX-XX-XX-XX_forward_mode_.../
├── pred_vllm.json       # 核心结果：每个采样点的 pred, hop, progress
├── sample.json           # 输入样本
├── reward_vis.mp4        # 可视化视频（三视角 + 奖励曲线）
└── .cache/               # 抽帧缓存
    ├── ref_end.png       # 目标参考图（= blank_goal.png）
    ├── cam_high/         # cam_high 抽帧
    ├── cam_left_wrist/   # 左腕抽帧
    └── cam_right_wrist/  # 右腕抽帧
```

`pred_vllm.json` 数据格式：

```json
[
  {
    "id": "step-...-0000-start_000000-af_000010",
    "task": "pick the carrot and place it on the yellow plate",
    "pred": "<score>+4.3%</score>",
    "hop": 0.043,
    "progress": 0.043
  },
  ...
]
```

- `pred`: 模型原始输出（带 `<score>` 标签）
- `hop`: 当前步的增量变化（forward 模式下 = progress 差值）
- `progress`: 累积进度（0-1 范围）

### 2.5 实测结果参考

使用 GRM-2.0-4B-Preview 模型，blank_goal（无目标图），frame_interval=10：

| 案例 | Forward | Incremental | Backward | 融合 |
|------|---------|-------------|----------|------|
| **成功 (pick_3_suc)** | 100.0% | 76.7% | 90.0% | **88.9%** |
| **失败 (pick_3_fail)** | 4.3% | 1.3% | 4.5% | **3.4%** |

模型能清晰区分成功与失败（融合进度相差约 26 倍）。

---

## 3. 微调数据预处理

微调数据需要经过三步处理管线，将视频转化为训练样本（JSON + 图片）。

### 3.1 Step 0: 组织原始数据目录

将视频数据按以下结构放置：

```
dataset/my_raw_data/
├── task_instruction.json          # 必需：同义任务描述列表
├── episode_001/
│   ├── annotated_keyframes.json   # 必需：子任务关键帧标注
│   ├── cam_high.mp4               # 必需：正上方相机
│   ├── cam_left_wrist.mp4         # 必需：左腕相机
│   └── cam_right_wrist.mp4        # 必需：右腕相机
├── episode_002/
│   └── ...
└── episode_XXX/
    └── ...
```

#### task_instruction.json 格式

提供多条同义描述，训练时随机选取，增强泛化：

```json
[
    "pick the carrot and place it on the yellow plate",
    "grasp the carrot and put it onto the yellow plate",
    "pick up the carrot and place it on the yellow plate",
    "grab the carrot and put it on the yellow plate",
    "move the carrot to the yellow plate"
]
```

#### annotated_keyframes.json 格式

标注子任务的起止帧。如果不需要分子任务，可以将整个视频作为单个任务：

```json
[
    {
        "anotation": "pick_carrot_place_plate",
        "start_frame_id": 0,
        "end_frame_id": 605
    }
]
```

如需细粒度子任务划分：

```json
[
    {"anotation": "approach_carrot",   "start_frame_id": 0,   "end_frame_id": 150},
    {"anotation": "grasp_carrot",      "start_frame_id": 150, "end_frame_id": 300},
    {"anotation": "lift_carrot",       "start_frame_id": 300, "end_frame_id": 400},
    {"anotation": "place_on_plate",    "start_frame_id": 400, "end_frame_id": 605}
]
```

> 注意：帧 ID 从 0 开始，`end_frame_id` 应 ≤ 视频总帧数 - 1。三个相机帧数需一致（提前对齐）。

### 3.2 Step 1: 预处理（视频抽帧）

从视频中提取所有帧，并基于关键帧标注均匀采样。

```bash
cd /home/dais/workspace/Robo-Dopamine/dataset

python -m utils.0_preprocess_data \
  --raw_dir ./my_raw_data \
  --cvt_dir ./my_train_data \
  --sample_factor 20
```

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--raw_dir` | 原始数据目录 | `./my_raw_data` |
| `--cvt_dir` | 输出目录 | `./my_train_data` |
| `--sample_factor` | 采样因子，值越小采样越密 | 20（约每 20 帧采样 1 帧） |

输出内容：
- 三个相机目录下的所有帧图片 (`frame_XXXXXX.jpg`)
- `sample_frames.json`（采样的关键帧 ID 列表）
- 复制 `annotated_keyframes.json` 和 `task_instruction.json`

### 3.3 Step 2: 生成训练样本（Bin-Sampling）

使用 Bin-Sampling 策略，确保各进度区间和帧间距的均衡覆盖。

```bash
python -m utils.1_generate_data \
  --base-dir ./my_train_data \
  --score-bins 25 \
  --gap-bins 4 \
  --oversample-factor 100 \
  --zero-ratio 0.05 \
  --max_sample_num 500 \
  --workers 4
```

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--base-dir` | Step 1 的输出目录 | `./my_train_data` |
| `--max_sample_num` | 每个 episode 生成的样本数 | 500-1000（数据量少时 500 即可） |
| `--score-bins` | 分数桶数量 | 25 |
| `--gap-bins` | 帧间距桶数量 | 4 |
| `--oversample-factor` | 候选池过采样因子 | 100 |
| `--zero-ratio` | 零分样本比例 | 0.05（5%） |
| `--workers` | 并行进程数 | 根据数据量调整，4-64 |

每个训练样本包含：
- 8 张图片（Ref Start + Ref End + BEFORE 三视角 + AFTER 三视角）
- Prompt（包含任务描述）
- Score 标签（`<score>+NN%</score>` 格式）

### 3.4 Step 3: 后处理（合并 + 图片替换增强）

合并所有 episode 的训练数据，并进行图片替换数据增强。

```bash
python -m utils.2_posprocess_data \
  --root-dir ./my_train_data \
  --merged-json ./my_train_data/train_jsons/finetune_data_wo_replace.json \
  --final-json ./my_train_data/train_jsons/finetune_data_final.json \
  --replace-prob 0.75
```

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--root-dir` | 训练数据目录 | `./my_train_data` |
| `--merged-json` | 合并后未增强的 JSON 路径 | 见上 |
| `--final-json` | 最终训练 JSON 路径 | 见上 |
| `--replace-prob` | 前两张图片替换概率 | 0.75（75%） |

图片替换增强的作用：以 75% 概率将每个样本的 Ref Start 和 Ref End 替换为其他样本的，让模型不依赖特定参考图，增强泛化能力。

输出两个文件：
- `finetune_data_wo_replace.json`：原始合并数据
- `finetune_data_final.json`：经过替换增强的数据（**微调时使用这个**）

### 3.5 验证数据

```bash
# 快速验证
python3 -c "
import json
with open('dataset/my_train_data/train_jsons/finetune_data_final.json') as f:
    data = json.load(f)
print(f'Total samples: {len(data)}')
print(f'Sample 0 keys: {list(data[0].keys())}')
print(f'Sample 0 images: {len(data[0][\"image\"])}')
print(f'Sample 0 task: {data[0][\"task\"]}')
print(f'Sample 0 score: {data[0][\"conversations\"][1][\"value\"]}')
"
```

---

## 4. 微调运行

### 4.1 注册数据集

编辑 `train/qwenvl/data/__init__.py`，添加你的数据集：

```python
MY_CARROT_DATASET = {
    "annotation_path": "./dataset/my_train_data/train_jsons/finetune_data_final.json",
    "data_path": "./dataset",
}

data_dict = {
    ...
    "my_carrot_dataset": MY_CARROT_DATASET,
}
```

> `annotation_path` 指向 Step 3 输出的最终 JSON 文件；`data_path` 指向 dataset 根目录。

### 4.2 配置训练脚本

创建或修改训练脚本 `train/scripts/finetune_my_carrot.sh`：

```bash
#!/bin/bash
export PYTHONPATH=$(pwd)
export WANDB_MODE=disabled

# ==================== 路径配置 ====================
MODEL_PATH="./pretrained_models/GRM-2.0-4B-Preview"   # 基础模型路径
OUTPUT_DIR="./checkpoints/my_carrot_finetune"          # 输出目录
DATASETS=my_carrot_dataset                              # 数据集名（与 __init__.py 一致）

CACHE_DIR="./.cache"
mkdir -p $OUTPUT_DIR

# ==================== 训练参数（4x A100-80G） ====================
torchrun --nproc_per_node=4 --nnodes=1 --master_port=29515 \
    qwenvl/train/train_qwen.py \
    --model_name_or_path $MODEL_PATH \
    --tune_mm_llm True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --dataset_use $DATASETS \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --bf16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 5e-7 \
    --optim adamw_torch \
    --model_max_length 32768 \
    --data_flatten False \
    --data_packing False \
    --max_pixels 76800 \
    --min_pixels 12544 \
    --base_interval 2 \
    --video_max_frames 8 \
    --video_min_frames 4 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 3 \
    --eval_strategy "no" \
    --deepspeed ./scripts/zero3.json \
2>&1 | tee ${OUTPUT_DIR}/train.log
```

### 4.3 关键训练参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `tune_mm_llm` | True | 训练语言模型 |
| `tune_mm_vision` | True | 训练视觉编码器 |
| `tune_mm_mlp` | True | 训练投影层 |
| `learning_rate` | 1e-5 | LLM 学习率 |
| `mm_projector_lr` | 1e-5 | 投影层学习率 |
| `vision_tower_lr` | 5e-7 | 视觉塔学习率（比 LLM 小一个量级） |
| `per_device_train_batch_size` | 2 | 每卡 batch size |
| `gradient_accumulation_steps` | 4 | 梯度累积 |
| **有效 Batch Size** | **2 × 4 × 4 = 32** | per_device × grad_accum × GPU 数 |
| `num_train_epochs` | 3 | 训练轮数 |
| `deepspeed` | zero3.json | DeepSpeed ZeRO-3 |

### 4.4 不同硬件配置的调整

| 硬件配置 | `nproc_per_node` | `gradient_accumulation_steps` | `deepspeed` |
|----------|-------------------|-------------------------------|-------------|
| 4×A100-80G | 4 | 4 | zero3.json |
| 8×A100-80G | 8 | 2 | zero3.json |
| 2×A100-80G | 2 | 8 | zero3.json |
| 1×A100-80G | 1 | 16 | zero3_offload.json |
| LoRA 微调（任意 GPU） | 按实际 | 调整使有效 batch=32 | zero2.json |

如需使用 LoRA（显存更少）：

```bash
# 在训练参数中添加/修改：
--lora_enable True \
--tune_mm_llm False \
--tune_mm_vision False \
--tune_mm_mlp False \
```

### 4.5 启动训练

```bash
cd /home/dais/workspace/Robo-Dopamine/train
conda activate robo-dopamine
bash scripts/finetune_my_carrot.sh
```

训练日志会同时输出到终端和 `${OUTPUT_DIR}/train.log`。

### 4.6 使用微调后的模型推理

训练完成后，模型保存在 `OUTPUT_DIR` 目录下。推理时将模型路径指向该目录即可：

```python
model = GRMInference("./checkpoints/my_carrot_finetune")
```

---

## 附录：完整目录结构参考

```
Robo-Dopamine/
├── pretrained_models/
│   └── GRM-2.0-4B-Preview -> /mnt/public1/ljx_test/.../Robo-Dopamine-GRM-2.0-4B-Preview
│
├── aligned_data/                      # 失败案例帧对齐后
│   ├── cam_high.mp4
│   ├── cam_left_wrist.mp4
│   └── cam_right_wrist.mp4
├── aligned_data_suc/                  # 成功案例帧对齐后
│   ├── cam_high.mp4
│   ├── cam_left_wrist.mp4
│   └── cam_right_wrist.mp4
│
├── dataset/
│   ├── my_raw_data/                   # 组织好的原始数据
│   │   ├── task_instruction.json
│   │   └── episode_001/
│   │       ├── annotated_keyframes.json
│   │       ├── cam_high.mp4
│   │       ├── cam_left_wrist.mp4
│   │       └── cam_right_wrist.mp4
│   ├── my_train_data/                 # 处理后的训练数据
│   │   ├── task_instruction.json
│   │   ├── episode_001/
│   │   │   ├── annotated_keyframes.json
│   │   │   ├── sample_frames.json
│   │   │   ├── cam_high/ (606 帧 jpg)
│   │   │   ├── cam_left_wrist/ (606 帧 jpg)
│   │   │   ├── cam_right_wrist/ (606 帧 jpg)
│   │   │   ├── train.json
│   │   │   └── info.txt
│   │   └── train_jsons/
│   │       ├── finetune_data_wo_replace.json  # 合并未增强
│   │       └── finetune_data_final.json       # 最终训练数据
│   └── utils/ (0_preprocess, 1_generate, 2_postprocess)
│
├── train/
│   ├── qwenvl/data/__init__.py        # 已注册 my_carrot_dataset
│   └── scripts/finetune_my_carrot.sh  # 微调脚本
│
├── test_my_data.py                    # 失败案例推理脚本
├── test_my_data_suc.py                # 成功案例推理脚本
│
└── results/
    ├── carrot_pick/                   # 失败案例推理结果
    │   ├── ..._forward_mode_.../
    │   │   ├── pred_vllm.json
    │   │   └── reward_vis.mp4
    │   ├── ..._incremental_mode_.../
    │   └── ..._backward_mode_.../
    └── carrot_pick_suc/               # 成功案例推理结果
        ├── ..._forward_mode_.../
        ├── ..._incremental_mode_.../
        └── ..._backward_mode_.../
```
