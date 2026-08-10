# v2 cross-model image-sequence 实验使用说明

本轮实验对应 `exp_plan_crossmodel.md`。所有命令从仓库根目录运行，模型实验使用
`robo-dopamine` 环境。配置位于 `mydata_bench/configs/v2_crossmodel/`，结果严格写入
规划中指定的 `results/mydata_bench/experiments_v2_corssmodel/`（保留原规划里的
`corssmodel` 拼写）。

## 冻结的输入与干预口径

- 从单视角 rollout 均匀采样 8 帧，始终包含首帧和真实终帧。
- 8 帧以 8 个独立 `image` item 输入，不再使用 `video` item，也不形成二帧
  temporal tubelet。
- 仍使用 RoboReward 的离散 rubric 和严格的 `ANSWER: <1-5>` 输出，保证输出指标
  口径不变；该输入消融不是 checkpoint-native video 协议。
- 为单独检验输入表示变化，实验 3、4、7、8 延续原 last-frame 实验：target
  bbox 只映射到最后一张独立图像，`negative_scope=target_span`，其它 ranking、top-k、
  bias 和 all-query 设置与 v2 保持一致。
- 实验 9--12 是 all-frame 干预：每张独立图像都按该采样时刻的 tracking bbox 映射
  target token，`temporal_intervention_scope=all_frames`；全部 target token 接受正 bias，
  `negative_scope=all_visual` 使 8 帧内其余所有视觉 token 接受负 bias。
- 实验 13--16 使用共享的 GRM 式交错协议：任务文本之后依次放置
  `OBSERVATION n` 文本和对应 image item，最后一张图之后再放置原离散 rubric 与
  `ANSWER: <1-5>` 输出约束。RoboReward 与 Qwen 使用完全相同的 prompt 文本和
  media order；13/15 为 all-frame，14/16 为 last-frame 干预。
- 每个 content order 单独计算 head ranking，禁止跨顺序或跨模型复用 ranking。

## Baseline

```bash
conda run -n robo-dopamine python mydata_bench/run_roboreward_eval.py run --config mydata_bench/configs/v2_crossmodel/baseline_01_roboreward_text_images.yaml
conda run -n robo-dopamine python mydata_bench/run_roboreward_eval.py score --run-dir results/mydata_bench/experiments_v2_corssmodel/baseline_01_roboreward_text_images

conda run -n robo-dopamine python mydata_bench/run_roboreward_eval.py run --config mydata_bench/configs/v2_crossmodel/baseline_02_roboreward_images_text.yaml
conda run -n robo-dopamine python mydata_bench/run_roboreward_eval.py score --run-dir results/mydata_bench/experiments_v2_corssmodel/baseline_02_roboreward_images_text

conda run -n robo-dopamine python mydata_bench/run_qwen_eval.py run --config mydata_bench/configs/v2_crossmodel/baseline_05_qwen_text_images.yaml
conda run -n robo-dopamine python mydata_bench/run_qwen_eval.py score --run-dir results/mydata_bench/experiments_v2_corssmodel/baseline_05_qwen_text_images

conda run -n robo-dopamine python mydata_bench/run_qwen_eval.py run --config mydata_bench/configs/v2_crossmodel/baseline_06_qwen_images_text.yaml
conda run -n robo-dopamine python mydata_bench/run_qwen_eval.py score --run-dir results/mydata_bench/experiments_v2_corssmodel/baseline_06_qwen_images_text
```

`run` 支持已有的 `--shard-id/--num-shards` 覆盖，可按 GPU 0--3 分片。不同分片共享
同一个 output directory，全部完成后再执行 `score`。

## Attention ranking 与 steering

RoboReward 使用 `attention_03_roboreward_text_images.yaml` 和
`attention_04_roboreward_images_text.yaml`；Qwen 使用
`attention_07_qwen_text_images.yaml` 和 `attention_08_qwen_images_text.yaml`。
每份配置依次执行下面六步，不能跳过 processor alignment 验证：

```bash
CROSS_CFG=mydata_bench/configs/v2_crossmodel/attention_03_roboreward_text_images.yaml
conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py prepare-ranking --config "$CROSS_CFG"
conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py validate-ranking --config "$CROSS_CFG"
conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py rank --config "$CROSS_CFG"
conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py prepare-cohort --config "$CROSS_CFG"
conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --retry-failed --config "$CROSS_CFG"
conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py score --config "$CROSS_CFG"
```

对 attention 04 重复上述 RoboReward 命令。attention 07、08 使用相同子命令顺序，
把入口替换成 `mydata_bench/run_qwen_attention.py`。`steer` 可增加
`--shard-id N --num-shards 4`；四个分片齐全后代码会确定性合并 `steering.jsonl`。

实验 9--12 使用完全相同的六步流水线。RoboReward 配置为
`attention_09_roboreward_text_images_all_frames.yaml`、
`attention_10_roboreward_images_text_all_frames.yaml`；Qwen 配置为
`attention_11_qwen_text_images_all_frames.yaml`、
`attention_12_qwen_images_text_all_frames.yaml`。

实验 13--16 也使用该六步流水线。RoboReward 配置为
`attention_13_roboreward_interleaved_all_frames.yaml`、
`attention_14_roboreward_interleaved_last_frame.yaml`；Qwen 配置为
`attention_15_qwen_interleaved_all_frames.yaml`、
`attention_16_qwen_interleaved_last_frame.yaml`。四项都固定
`protocol=roborewardbench_interleaved_image_sequence` 和
`content_order=interleaved`，不得使用模型专属 prompt。

验收实验 3、4、7、8 的 `processor_alignment_diagnostics.jsonl` 时，每条记录应满足：

- `visual_span_count == 8`；
- `video_metadata.independent_image_spans == true`；
- `video_metadata.target_source_frame_indices` 只包含真实终帧；
- target positions 全部位于 `image_t7` span 内。

实验 9--12 的 all-frame 额外验收项：

- `video_metadata.selected_target_span_labels` 为 `image_t0` 至 `image_t7`；
- 每个 span 的 `source_frame_indices` 与 `tracking_frame_indices` 都有明确映射；
- `negative_scope == all_visual`，正负 token 不相交，且二者并集等于全部视觉 token。

实验 13--16 还必须验收 processor 展开后的 prompt 顺序：

- 8 个独立 image span 均存在，且 `image_t0` 至 `image_t7` 顺序不变；
- 每个 image span 前都有对应 `OBSERVATION n` 文本；rubric 位于 `image_t7` 之后；
- prompt SHA 在 RoboReward 与 Qwen 配置之间完全一致；
- 13/15 满足上述 all-frame token 契约；14/16 只在 `image_t7` 内形成互补的
  target / non-target 集合。

## 汇总指标与 ranking overlap

逐实验记录（MAE、总体/suc/fail/task 准确率、分布和同视频 pairwise 分档）：

```bash
python mydata_bench/write_exp_records.py \
  --experiments-root results/mydata_bench/experiments_v2_corssmodel \
  --metadata /home/dais/workspace/data/mydata_v2/new/metadata.jsonl \
  --ranking-metadata /home/dais/workspace/data/mydata_v2/new/ranking_data.jsonl
```

四个 attention 实验的 top-8/32/64 重合度：

```bash
python mydata_bench/write_ranking_overlap.py \
  --experiments-root results/mydata_bench/experiments_v2_corssmodel \
  --experiment attention_03_roboreward_text_images \
  --experiment attention_04_roboreward_images_text \
  --experiment attention_07_qwen_text_images \
  --experiment attention_08_qwen_images_text \
  --output-json results/mydata_bench/experiments_v2_corssmodel/ranking_overlap.json \
  --output-md results/mydata_bench/experiments_v2_corssmodel/ranking_overlap.md
```

实验 9--12 单独生成 all-frame overlap，避免覆盖上述 last-frame 汇总：

```bash
python mydata_bench/write_ranking_overlap.py \
  --experiments-root results/mydata_bench/experiments_v2_corssmodel \
  --experiment attention_09_roboreward_text_images_all_frames \
  --experiment attention_10_roboreward_images_text_all_frames \
  --experiment attention_11_qwen_text_images_all_frames \
  --experiment attention_12_qwen_images_text_all_frames \
  --output-json results/mydata_bench/experiments_v2_corssmodel/ranking_overlap_all_frames.json \
  --output-md results/mydata_bench/experiments_v2_corssmodel/ranking_overlap_all_frames.md
```

实验 13--16 的 GRM 式交错输入也写入独立 overlap：

```bash
python mydata_bench/write_ranking_overlap.py \
  --experiments-root results/mydata_bench/experiments_v2_corssmodel \
  --experiment attention_13_roboreward_interleaved_all_frames \
  --experiment attention_14_roboreward_interleaved_last_frame \
  --experiment attention_15_qwen_interleaved_all_frames \
  --experiment attention_16_qwen_interleaved_last_frame \
  --output-json results/mydata_bench/experiments_v2_corssmodel/ranking_overlap_interleaved.json \
  --output-md results/mydata_bench/experiments_v2_corssmodel/ranking_overlap_interleaved.md
```

和 v2 既有实验一致，本轮 attention cohort 仍是未经人工审核的自动 grounding
exploratory 集合，不能将描述性结果写成已确认的普遍因果效应。
