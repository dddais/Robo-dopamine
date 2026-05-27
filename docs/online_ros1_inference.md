# Robo-Dopamine ROS1 在线推理说明

本文档说明 `dev_franka` 分支中新增的 ROS1 在线推理脚本：

```text
examples/online_inference_ros1.py
```

该脚本用于把原本“离线处理完整视频”的 Robo-Dopamine 推理流程，增量扩展为“在线读取机器人相机画面并持续输出当前任务进度”。实现上不修改原有 `examples/inference.py`，而是复用其中的 `GRMInference.inference_batch()`。

## 1. 方案结构

在线方案由四层组成：

1. ROS1 图像输入层
   - 订阅三路 `sensor_msgs/Image`：
     - front camera
     - left wrist camera
     - right wrist camera
   - 使用 `message_filters.ApproximateTimeSynchronizer` 做近似时间同步。
   - 使用 `cv_bridge` 将 ROS Image 转为 OpenCV 图像。

2. 在线帧缓存层
   - ROS callback 收到同步帧后写入 `LatestFrameBuffer`。
   - 推理主循环按 `--sample-period` 周期读取最新同步帧。
   - 每次推理前将三路当前帧保存为 PNG，便于复现和离线检查。

3. GRM 推理层
   - 在线脚本导入并复用：

```python
from examples.inference import GRMInference
```

   - 每个在线 step 构造三条 sample：
     - `forward`
     - `incremental`
     - `backward`
   - 三条 sample 一次性送入 `GRMInference.inference_batch()`。

4. 进度融合与输出层
   - 解析模型输出中的 `<score>...</score>`。
   - 按原离线脚本中的三种模式公式更新 progress。
   - 将三种模式的 progress 平均后得到 fused progress。
   - 持续写入：
     - `online_pred.jsonl`
     - `latest_progress.json`
     - 退出时生成 `pred_vllm_online.json`

整体数据流：

```text
ROS Image topics
    -> ApproximateTimeSynchronizer
    -> cv_bridge / OpenCV image
    -> save frame triplet
    -> build forward/incremental/backward samples
    -> GRMInference.inference_batch()
    -> parse score
    -> per-mode progress update
    -> fused progress
    -> JSONL / latest JSON / terminal log
```

## 2. 推理原理

### 2.1 输入图像顺序

Robo-Dopamine 的 GRM prompt 期望 8 张图像，顺序与离线脚本保持一致：

```text
1. ref_start front image
2. ref_end image
3. before front image
4. before left wrist image
5. before right wrist image
6. after front image
7. after left wrist image
8. after right wrist image
```

在线脚本启动后，会等待第一组三路同步帧，并将它们作为本次在线 episode 的 `ref_start` 和初始 `previous`。

真实目标图由 `--goal-image` 提供，并复制到输出目录的：

```text
.cache/ref_end.png
```

### 2.2 三种模式的构造方式

每个在线 step 都会构造三种 sample：

| 模式 | before 图像 | after 图像 | 含义 |
| --- | --- | --- | --- |
| `forward` | 启动时第一帧 `ref_start` | 当前帧 | 估计从起点到当前的绝对进度 |
| `incremental` | 上一次推理帧 `previous` | 当前帧 | 估计最近一步是否前进或后退 |
| `backward` | 目标图 `goal_image` | 当前帧 | 估计当前距离目标完成状态还有多远 |

注意：为了与现有离线逻辑保持一致，`backward` 模式下会把同一张 `goal_image` 同时放入 before front、before left wrist、before right wrist 三个位置。

### 2.3 分数解析

模型输出格式应为：

```text
<score>+NN%</score>
<score>-NN%</score>
<score>0%</score>
```

脚本将其解析为 `[-1, 1]` 的浮点数：

```text
+70% -> 0.70
-20% -> -0.20
0%   -> 0.00
```

如果输出格式异常，当前 step 的 score 会退化为 `0.0`。

### 2.4 进度计算

三种模式的 progress 更新方式与 `examples/inference.py` 保持一致。

`forward`：

```text
progress = score
hop = progress - previous_forward_progress
```

`incremental`：

```text
第一步:
progress = score

后续 step:
score >= 0: progress = prev + (1 - prev) * score
score <  0: progress = prev + prev * score

hop = score
```

`backward`：

```text
progress = clamp(1 + score, 0, 1)
hop = progress - previous_backward_progress
```

最终在线进度：

```text
fused_progress = clamp(mean(forward_progress, incremental_progress, backward_progress), 0, 1)
```

控制侧建议读取 `latest_progress.json` 中的 `progress` 字段作为当前任务进度。

## 3. 使用方法

### 3.1 环境准备

进入仓库并确认分支：

```bash
cd /home/dais/workspace/Robo-Dopamine
git switch dev_franka
```

加载 ROS1 和 Python 环境：

```bash
source /opt/ros/noetic/setup.bash
conda activate robo-dopamine
```

如果你的 ROS workspace 里包含相机驱动或 Franka 相关包，还需要 source 对应工作空间：

```bash
source /path/to/your/catkin_ws/devel/setup.bash
```

### 3.2 ROS topic 检查

运行前先确认三路图像 topic 存在且在发布：

```bash
rostopic list
rostopic hz /your/front/image_raw
rostopic hz /your/left_wrist/image_raw
rostopic hz /your/right_wrist/image_raw
```

确认 topic 类型是：

```bash
rostopic type /your/front/image_raw
```

期望输出：

```text
sensor_msgs/Image
```

当前脚本面向 raw `sensor_msgs/Image`。如果相机发布的是 `sensor_msgs/CompressedImage`，需要另写 compressed 输入适配或先在 ROS 中转成 raw Image。

### 3.3 准备目标图

`--goal-image` 应该是一张真实完成状态图片，例如：

```text
/home/dais/workspace/Robo-Dopamine/examples/exp_suc_1.png
```

对于 `backward` 模式，真实目标图会明显优于 `blank_goal.png`。如果没有真实目标图，也可以先使用 `examples/blank_goal.png` 做连通性测试，但 fused progress 的可信度会下降。

### 3.4 启动在线推理

推荐先用 `--max-steps 3` 做短测试：

```bash
python examples/online_inference_ros1.py \
  --model-path /home/dais/workspace/Robo-Dopamine/train/checkpoints/my_carrot_finetune_big \
  --front-topic /your/front/image_raw \
  --left-topic /your/left_wrist/image_raw \
  --right-topic /your/right_wrist/image_raw \
  --task "pick the carrot and put it on yellow plate" \
  --goal-image /path/to/real_goal.png \
  --out-root ./results/online_franka \
  --sample-period 1.0 \
  --sync-slop 0.05 \
  --max-steps 3
```

确认输出正常后，去掉 `--max-steps` 持续运行：

```bash
python examples/online_inference_ros1.py \
  --model-path /home/dais/workspace/Robo-Dopamine/train/checkpoints/my_carrot_finetune_big \
  --front-topic /your/front/image_raw \
  --left-topic /your/left_wrist/image_raw \
  --right-topic /your/right_wrist/image_raw \
  --task "pick the carrot and put it on yellow plate" \
  --goal-image /path/to/real_goal.png \
  --out-root ./results/online_franka \
  --sample-period 1.0 \
  --sync-slop 0.05
```

### 3.5 重要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-path` | `/home/dais/workspace/Robo-Dopamine/train/checkpoints/my_carrot_finetune_big` | GRM 模型路径 |
| `--front-topic` | 必填 | 正面相机 raw Image topic |
| `--left-topic` | 必填 | 左腕相机 raw Image topic |
| `--right-topic` | 必填 | 右腕相机 raw Image topic |
| `--task` | 必填 | 任务文本，会进入 GRM prompt |
| `--goal-image` | 必填 | 真实目标图，作为 ref_end 和 backward anchor |
| `--out-root` | `./results/online` | 在线输出根目录 |
| `--sample-period` | `1.0` | 两次推理之间的最小间隔，实际频率还受模型推理耗时限制 |
| `--sync-slop` | `0.05` | 三路相机近似同步允许的时间差，单位秒 |
| `--sync-queue-size` | `10` | message_filters 同步队列长度 |
| `--wait-timeout` | `30.0` | 等待同步帧的超时时间，单位秒 |
| `--max-steps` | `0` | 最大推理 step 数；0 表示一直运行 |
| `--image-encoding` | `bgr8` | cv_bridge 输出编码 |

## 4. 输出目录与文件

每次运行会在 `--out-root` 下新建一个目录：

```text
results/online_franka/
└── 26-XX-XX-XX-XX-XX_online_pick_the_carrot_and_put_it_on_yellow_plate/
    ├── metadata.json
    ├── online_pred.jsonl
    ├── latest_progress.json
    ├── pred_vllm_online.json
    └── .cache/
        ├── ref_end.png
        ├── cam_high/
        │   ├── frame_000000.png
        │   ├── frame_000001.png
        │   └── ...
        ├── cam_left_wrist/
        └── cam_right_wrist/
```

文件说明：

| 文件 | 用途 |
| --- | --- |
| `metadata.json` | 保存模型路径、任务文本、ROS topic、同步参数等运行元信息 |
| `online_pred.jsonl` | 每个在线 step 追加一行结果，适合实时读取和长期记录 |
| `latest_progress.json` | 始终覆盖为最新一步结果，适合控制侧轮询 |
| `pred_vllm_online.json` | Ctrl-C 或正常结束时由 JSONL 汇总生成，方便离线分析 |
| `.cache/` | 保存每个 step 的三路相机帧和目标图 |

## 5. 输出格式

`latest_progress.json` 与 `online_pred.jsonl` 单行记录格式一致：

```json
{
  "step": 12,
  "wall_time": 1770000000.0,
  "ros_stamp": 1770000000.0,
  "latency_s": 2.4,
  "task": "pick the carrot and put it on yellow plate",
  "progress": 0.73,
  "progress_percent": 73.0,
  "modes": {
    "forward": {
      "pred": "<score>+70%</score>",
      "score": 0.7,
      "hop": 0.04,
      "progress": 0.7
    },
    "incremental": {
      "pred": "<score>+8%</score>",
      "score": 0.08,
      "hop": 0.08,
      "progress": 0.68
    },
    "backward": {
      "pred": "<score>-20%</score>",
      "score": -0.2,
      "hop": 0.02,
      "progress": 0.8
    }
  },
  "frames": {
    "cam_high": ".../frame_000012.png",
    "cam_left_wrist": ".../frame_000012.png",
    "cam_right_wrist": ".../frame_000012.png"
  }
}
```

控制侧最常用字段：

```text
progress
progress_percent
modes.forward.progress
modes.incremental.progress
modes.backward.progress
```

## 6. 终端日志

每次推理完成后，脚本会打印类似：

```text
[ONLINE] Step 000012 fused= 73.00% forward= 70.00% incremental= 68.00% backward= 80.00% latency=2.40s
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `Step` | 在线推理 step 编号 |
| `fused` | 三模式平均后的最终进度 |
| `forward` | forward 模式进度 |
| `incremental` | incremental 模式进度 |
| `backward` | backward 模式进度 |
| `latency` | 当前 step 从取帧到输出结果的耗时 |

## 7. 验证与排错

### 7.1 静态检查

在非 ROS 环境也可以运行：

```bash
python examples/online_inference_ros1.py --help
python -m py_compile examples/online_inference_ros1.py
```

### 7.2 等不到同步帧

现象：

```text
Timed out waiting for synchronized ROS image frames.
```

检查项：

1. 三路 topic 是否都在发布。
2. topic 类型是否都是 `sensor_msgs/Image`。
3. 三路消息 header stamp 是否合理。
4. 适当增大 `--sync-slop`，例如从 `0.05` 调到 `0.1` 或 `0.2`。
5. 适当增大 `--sync-queue-size`。

### 7.3 图像颜色异常

默认 `--image-encoding bgr8`，保存 PNG 时直接使用 OpenCV 写入。如果你的相机 topic 编码不是 BGR，可以尝试：

```bash
--image-encoding rgb8
```

如果颜色只影响可视化但不影响模型判断，可以先保持默认；如果模型进度明显不稳定，应检查编码。

### 7.4 推理频率低

`--sample-period` 只是两次采样的最小间隔。实际在线频率还取决于：

1. 模型大小。
2. GPU 显存和算力。
3. vLLM 当前 batch 推理耗时。
4. 三模式同时推理带来的额外开销。

如果需要更低延迟，可以临时只保留 `forward + incremental`，但这需要修改脚本中的 `VALID_MODES` 和融合逻辑。

### 7.5 backward 结果不可靠

优先检查 `--goal-image` 是否为真实完成状态图片。`backward` 对目标图质量敏感，目标图偏差会直接影响 fused progress。

## 8. 与离线推理的关系

离线脚本 `examples/inference.py` 的职责是：

```text
完整视频或图片目录 -> 抽帧 -> 构造 sample.json -> 批量 GRM 推理 -> pred_vllm.json
```

在线脚本 `examples/online_inference_ros1.py` 的职责是：

```text
ROS 实时图像 -> 周期采样 -> 构造当前 step 的 sample -> GRM 推理 -> 实时 progress JSON
```

两者共享：

1. 同一套 prompt。
2. 同一个 `GRMInference.inference_batch()`。
3. 同样的 8 图输入顺序。
4. 同样的三模式 progress 后处理思想。

在线脚本是增量扩展，不替代离线脚本。建议先用离线视频验证任务和模型效果，再切换到 ROS 在线推理。
