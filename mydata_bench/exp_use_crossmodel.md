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
- 为单独检验输入表示变化，四个 attention 配置都延续原 last-frame 实验：target
  bbox 只映射到最后一张独立图像，`negative_scope=target_span`，其它 ranking、top-k、
  bias 和 all-query 设置与 v2 保持一致。
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

验收 `processor_alignment_diagnostics.jsonl` 时，每条记录应满足：

- `visual_span_count == 8`；
- `video_metadata.independent_image_spans == true`；
- `video_metadata.target_source_frame_indices` 只包含真实终帧；
- target positions 全部位于 `image_t7` span 内。

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

和 v2 既有实验一致，本轮 attention cohort 仍是未经人工审核的自动 grounding
exploratory 集合，不能将描述性结果写成已确认的普遍因果效应。
