# Qwen3-VL 基线评测

`qwen_eval` 将两种不可直接混用的输入/输出协议明确分开。

| `qwen_eval.protocol` | 模型输入 | 必需原始输出 | 指标解释 |
|---|---|---|---|
| `roborewardbench_native` | 任务/评分标准文本后接未改动 MP4 | `ANSWER: <1-5>` | 原生离散有序指标 |
| `robo_dopamine_forward` | Robo-Dopamine 官方前向提示词与 8 张端点图像 | `<score>[+-]NN%</score>` | 有符号分数先截断为 progress，再映射到 1--5 |

第二种是受控的输入/输出消融，不是 RoboRewardBench 官方的离散基线。因此其
`metrics.json` 会写入 `adapter_metric=true`。两种模式都不会将 `reward` 或
`gpt5_mini_check` 传入模型输入。

## 命令

```bash
conda activate robo-dopamine

# 全量 benchmark：原生离散视频协议。
python rewardbench/run_qwen_eval.py run \
  --config rewardbench/configs/full_qwen3vl_8b_roborewardbench_native.yaml
python rewardbench/run_qwen_eval.py score \
  --run-dir rewardbench/qwen_eval/outputs/full_qwen3vl_8b_roborewardbench_native

# 冻结的 111-ID SAM3 单人审核 cohort：同样使用原生协议。
python rewardbench/run_qwen_eval.py run \
  --config rewardbench/configs/reward1_sam3_eligible_qwen3vl_8b_roborewardbench_native.yaml
python rewardbench/run_qwen_eval.py score \
  --run-dir rewardbench/qwen_eval/outputs/reward1_sam3_formal111_qwen3vl_8b_roborewardbench_native

# 可选：全量 benchmark 的 Robo-Dopamine 输入/输出消融。
python rewardbench/run_qwen_eval.py run \
  --config rewardbench/configs/full_qwen3vl_8b_robo_dopamine_forward.yaml
python rewardbench/run_qwen_eval.py score \
  --run-dir rewardbench/qwen_eval/outputs/full_qwen3vl_8b_robo_dopamine_forward

# 可选：111-ID cohort 的 Robo-Dopamine 输入/输出消融。
python rewardbench/run_qwen_eval.py run \
  --config rewardbench/configs/reward1_sam3_eligible_qwen3vl_8b_robo_dopamine_forward.yaml
python rewardbench/run_qwen_eval.py score \
  --run-dir rewardbench/qwen_eval/outputs/reward1_sam3_formal111_qwen3vl_8b_robo_dopamine_forward
```

使用 `--dry-run` 可在不加载 Qwen 权重的情况下检查配置和产物布局；后续真实运行会
自动替换这些 dry-run 记录。真实运行不会自动重试 `invalid` 记录；请先检查原因，再
按需传入 `--retry-failed`。

## reward=5 筛选

冻结的 527-ID、无标签源 cohort 位于
`rewardbench/cohorts/outputs/reward5_full_527/example_ids.json`，可手动执行：

```bash
python rewardbench/run_grounding.py parse \
  --config rewardbench/configs/reward5_full_sam3_screen.yaml
python rewardbench/run_grounding.py run --backend sam3 \
  --config rewardbench/configs/reward5_full_sam3_screen.yaml
python rewardbench/run_grounding.py audit \
  --run-dir rewardbench/grounding/outputs/reward5_full_sam3/sam3
```

不得把源数据的 reward=5 metadata 输入模型。只有审计完成并冻结无标签 eligible ID
列表后，才应创建最终的模型评测配置。

## 跨模型 attention ranking 与 steering

使用 carrot、bottle、cube 三条独立成功轨迹的最终人工标注端点进行每模型 ranking；按最后
prompt token 对目标 bbox 的 excess attention mass 排名，再以三源 normalized-Borda 共识取
top-8 heads（跳过第 0--1 层）。原生 MP4 attention 实验固定最多 8 帧，仅为避免完整 attention
矩阵过大，不改变已完成的 baseline。四个条件为 baseline、top-8→目标、top-8→等尺寸错误
视觉区域、低排名→目标；hook 仅作用于最后 prompt query。

GPU 0 运行 Qwen 原生、GPU 1 运行 Qwen 八图前向：

```bash
CUDA_VISIBLE_DEVICES=0 python rewardbench/run_qwen_attention.py rank --config rewardbench/configs/attention_qwen3vl_8b_native_rank.yaml
CUDA_VISIBLE_DEVICES=1 python rewardbench/run_qwen_attention.py rank --config rewardbench/configs/attention_qwen3vl_8b_forward_rank.yaml
CUDA_VISIBLE_DEVICES=0 python rewardbench/run_qwen_attention.py steer --config rewardbench/configs/attention_qwen3vl_8b_native_reward1.yaml
CUDA_VISIBLE_DEVICES=1 python rewardbench/run_qwen_attention.py steer --config rewardbench/configs/attention_qwen3vl_8b_forward_reward1.yaml
python rewardbench/run_qwen_attention.py score --config rewardbench/configs/attention_qwen3vl_8b_native_reward1.yaml
```

reward=5 将配置名中的 `reward1` 改为 `reward5`。八图前向仍是 adapter 指标，不能与原生
`ANSWER: <1-5>` 结果混合比较。完整结果见 `rewardbench/attention_steering_exp_record.md`。
# Legacy reference

This file was copied with rewardbench. For the new dataset, use
mydata_bench/exp_use.md and the mydata_*.yaml configurations; commands below
refer to historical rewardbench runs.
