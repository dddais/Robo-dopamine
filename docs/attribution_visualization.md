# Robo-Dopamine Attribution Visualization

本文档总结 `visualize_my_data_attribution.py` 的改动内容、实现原理和使用方法。该脚本用于回答：

> GRM 基于视频给出进度分数时，图像中哪些区域对分数起到了关键作用？

脚本会输出带热力图叠加的视频，显示 attention 和 gradient attribution 两类可视化结果。

## 1. 本次新增内容

新增文件：

```text
visualize_my_data_attribution.py
```

该脚本独立于现有 `examples.inference.GRMInference`，不会修改原有 vLLM 评分流程。它的主要能力是：

- 读取与 `test_my_data_suc.py` 类似的配置：
  - `MODEL_PATH`
  - `DATA_DIR`
  - `OUTPUT_ROOT`
  - `TASK_INSTRUCTION`
  - `GOAL_IMAGE`
  - `INTERVAL`
- 支持三种评估模式：
  - `forward`
  - `incremental`
  - `backward`
- 对每个 sample 构造与原 GRM 推理一致的 8 张图输入：
  - reference start
  - reference end / goal image
  - before high / left wrist / right wrist
  - after high / left wrist / right wrist
- 输出三视角并排视频：
  - `cam_high`
  - `cam_left_wrist`
  - `cam_right_wrist`
- 每帧视频上下两排：
  - 上排：attention heatmap
  - 下排：gradient attribution heatmap

默认输出路径形如：

```text
results/pick_3_fail/exp_suc1_inter20_ckpt_attribution/<timestamp>/<mode>_mode_<task>/attribution_vis.mp4
```

同时会保存：

```text
sample.json
pred_attribution.json
heatmaps/sample_XXXX.npz
```

## 2. 为什么单独用 HuggingFace Transformers

原项目的主推理路径使用 vLLM：

```python
from vllm import LLM
outputs = self.model.generate(...)
```

vLLM 适合快速批量生成分数，优点是速度快、显存调度好、已有流程稳定。但 attribution 可视化需要访问模型内部信息：

- 每层 text attention 权重
- image token 的 embedding
- gradient / backward
- image token 与原图 patch 的映射

这些能力在 vLLM 中不是稳定公开接口。尤其是 gradient attribution 需要 autograd 反传，而 vLLM 的设计目标是高效推理，不是保留中间激活并做梯度分析。

因此脚本使用：

```python
Qwen3VLForConditionalGeneration.from_pretrained(...)
```

也就是 HuggingFace Transformers 路径来做解释性分析。这样做的取舍是：

- 原有 vLLM 评分流程不受影响。
- 可解释性脚本可以拿 attention 和 gradient。
- 运行速度比 vLLM 慢。
- 显存占用更高，建议单独运行该脚本，不要同时保留 vLLM 进程。

## 3. 可视化原理

### 3.1 输入和目标

模型的输出是文本形式的分数，例如：

```text
<score>+35%</score>
```

脚本将这个 score 文本作为解释目标，分析：

> 哪些 image tokens 支持模型生成这个分数？

如果没有提供已有 vLLM 的 `pred_vllm.json`，脚本会先用 HuggingFace 模型自己生成 score。  
如果提供了 `pred_vllm.json`，脚本会读取其中的 `pred` 字段，然后对该分数做 teacher-forced attribution。

### 3.2 Attention heatmap

attention heatmap 表示：

> 模型生成 score token 时，最后若干层 text attention 看向哪些 image tokens。

实现方式：

1. 使用与 GRM 推理一致的 prompt 和 8 张图。
2. 将 score 文本拼接到 prompt 后面。
3. 注册 hook，抓取 Qwen3-VL text attention 模块的 attention weights。
4. 默认聚合最后 4 层、所有 heads。
5. 取 score answer token 作为 query，image tokens 作为 key。
6. 将 image token 的 attention 分数映射回每张图的 patch grid。
7. resize 到原图大小并叠加显示。

attention 更像是“模型生成分数时关注哪里”，不是严格因果解释。

### 3.3 Gradient attribution heatmap

gradient attribution 更接近 Grad-CAM / saliency 的思路，但不是传统 CNN Grad-CAM。

传统 Grad-CAM 一般对 CNN 的最后卷积特征图求梯度。但 Qwen3-VL 是 VLM 架构：

- 图像先被切成 patch。
- patch 被视觉编码器转换成 image tokens。
- image tokens 与文本 tokens 共同进入语言模型。
- 输出是 `<score>...</score>` 这样的文本，而不是分类 head 的单个 logit。

因此脚本对 image tokens 做归因：

1. 得到视觉编码器输出的 image features。
2. 将 image features 替换到文本输入 embedding 中。
3. 对 score tokens 的 log-prob 求和，作为 attribution 目标。
4. 对 image features 做反向传播。
5. 使用 `grad * activation` 得到每个 image token 的重要性。
6. 将 image token 重要性映射回每张图的 patch grid。
7. resize 到原图大小并叠加显示。

这个方案的好处：

- 适配 Qwen3-VL 的真实计算单位，即 image tokens。
- 支持多图输入，不限于单张图。
- 能分别显示 high / left wrist / right wrist 三视角。
- 解释目标直接对应模型生成的 score 文本。
- 不需要修改模型结构，也不需要重新训练。

## 4. 输出视频布局

每个输出视频的单帧尺寸为：

```text
1152 x 576
```

布局如下：

```text
+----------------------+----------------------+----------------------+
| attention cam_high   | attention left wrist | attention right wrist|
+----------------------+----------------------+----------------------+
| gradient cam_high    | gradient left wrist  | gradient right wrist |
+----------------------+----------------------+----------------------+
```

视频顶部会显示当前 sample 的信息：

```text
score=<score>...</score>  hop=...  progress=...
```

颜色含义：

- 红 / 黄：相对重要区域更强。
- 蓝 / 深色：相对重要性更弱。

注意：热力图是归一化后的相对强度，不同帧之间不应直接用颜色绝对值做严格比较。

## 5. 使用方法

### 5.1 激活环境

```bash
source /mnt/public1/dais/miniconda3/etc/profile.d/conda.sh
conda activate robo-dopamine
```

### 5.2 快速测试

建议先只跑每个模式 1 个 sample，确认显存和输出正常：

```bash
python visualize_my_data_attribution.py \
  --modes forward incremental backward \
  --max-samples 1
```

短测试输出示例：

```text
results/pick_3_fail/exp_suc1_inter20_ckpt_attribution/<timestamp>/forward_mode_<task>/attribution_vis.mp4
results/pick_3_fail/exp_suc1_inter20_ckpt_attribution/<timestamp>/incremental_mode_<task>/attribution_vis.mp4
results/pick_3_fail/exp_suc1_inter20_ckpt_attribution/<timestamp>/backward_mode_<task>/attribution_vis.mp4
```

### 5.3 跑完整默认配置

```bash
python visualize_my_data_attribution.py
```

默认会读取脚本顶部配置：

```python
MODEL_PATH = "/home/dais/workspace/Robo-Dopamine/train/checkpoints/my_carrot_finetune_big"
DATA_DIR = "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_1"
OUTPUT_ROOT = "./results/pick_3_fail/exp_suc1_inter20_ckpt_attribution"

TASK_INSTRUCTION = "pick the carrot and put it on yellow plate "
GOAL_IMAGE = "./examples/blank_goal.png"
INTERVAL = 20
EVAL_MODES = ["forward", "incremental", "backward"]
```

### 5.4 通过命令行覆盖配置

```bash
python visualize_my_data_attribution.py \
  --model-path /path/to/checkpoint \
  --data-dir /path/to/data_dir \
  --out-root ./results/my_attribution \
  --task "pick the carrot and put it on yellow plate" \
  --goal-image ./examples/blank_goal.png \
  --interval 20 \
  --modes forward incremental backward
```

其中 `data-dir` 下需要包含：

```text
cam_high.mp4
cam_left_wrist.mp4
cam_right_wrist.mp4
```

### 5.5 使用已有 vLLM 预测结果

如果已经用 `test_my_data_suc.py` 或 `GRMInference.run_pipeline()` 得到了 `pred_vllm.json`，可以让 attribution 脚本解释这些已有分数：

```bash
python visualize_my_data_attribution.py \
  --pred-json-forward /path/to/forward/pred_vllm.json \
  --pred-json-incremental /path/to/incremental/pred_vllm.json \
  --pred-json-backward /path/to/backward/pred_vllm.json
```

这样脚本不会重新自由生成 score，而是读取 `pred_vllm.json` 中的 `pred` 字段作为 attribution 目标。

## 6. 参数说明

常用参数：

```text
--model-path              模型 checkpoint 路径
--data-dir                三视角视频目录
--out-root                输出根目录
--task                    任务文本
--goal-image              goal image；没有目标图时可用 examples/blank_goal.png
--interval                抽帧间隔
--modes                   要运行的模式，可选 forward incremental backward
--max-samples             每个模式最多处理多少个 sample，用于调试
--min-pixels              图像最小像素限制
--max-pixels              图像最大像素限制
--pred-json-forward       forward 模式已有 pred_vllm.json
--pred-json-incremental   incremental 模式已有 pred_vllm.json
--pred-json-backward      backward 模式已有 pred_vllm.json
```

## 7. 性能和显存注意事项

该脚本比普通 vLLM 推理更慢，原因是：

- 使用 Transformers 加载模型。
- 需要保留 attention。
- 需要做 backward 计算 gradient attribution。
- 默认 batch size 等价于 1。

如果显存不足，可以按顺序降低成本：

1. 先用 `--max-samples 1` 验证。
2. 只跑一个模式，例如：

   ```bash
   python visualize_my_data_attribution.py --modes forward --max-samples 1
   ```

3. 降低图像分辨率：

   ```bash
   python visualize_my_data_attribution.py --max-pixels 40401
   ```

4. 在脚本中降低：

   ```python
   ATTENTION_LAST_N_LAYERS = 1
   ```

## 8. 已验证结果

已完成以下验证：

```bash
python -m py_compile visualize_my_data_attribution.py
```

通过。

三模式短验证通过：

```bash
python visualize_my_data_attribution.py \
  --modes forward incremental backward \
  --max-samples 1 \
  --out-root ./results/attribution_smoke_3modes
```

验证输出：

```text
results/attribution_smoke_3modes/26-05-26-22-39-53/forward_mode_pick_the_carrot_and_put_it_on_yellow_plate_/attribution_vis.mp4
results/attribution_smoke_3modes/26-05-26-22-39-53/incremental_mode_pick_the_carrot_and_put_it_on_yellow_plate_/attribution_vis.mp4
results/attribution_smoke_3modes/26-05-26-22-39-53/backward_mode_pick_the_carrot_and_put_it_on_yellow_plate_/attribution_vis.mp4
```

三个视频均可被 OpenCV 读取，帧尺寸为：

```text
576 x 1152 x 3
```

## 9. 解释时的注意边界

- attention heatmap 表示模型生成分数时的注意力分布，不等价于严格因果贡献。
- gradient attribution 更接近“哪些 image tokens 会影响 score token 的概率”，但它仍是局部梯度解释。
- 热力图颜色是单帧内归一化结果，主要用于观察一帧内哪些区域相对重要。
- 多视角任务中，腕部视角可能对抓取、接触、滑移等细节更关键；主视角通常对全局位置和目标关系更关键。
- 如果使用 blank goal image，`backward` 模式的解释意义需要谨慎理解，因为 goal anchor 本身不包含真实目标视觉信息。

