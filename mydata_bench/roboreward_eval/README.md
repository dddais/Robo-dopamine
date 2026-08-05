# RoboReward-8B 评测与 attention steering

checkpoint-native 评测使用任务/评分标准文本后接原始 MP4，并要求 `ANSWER: <1-5>`。attention
ranking 与 Qwen 使用同一三条独立的 carrot/bottle/cube 成功轨迹，但每个 checkpoint 单独计算
head ranking，绝不复用 Qwen 或 GRM 的 head 列表。

```bash
CUDA_VISIBLE_DEVICES=2 python rewardbench/run_roboreward_attention.py rank --config rewardbench/configs/attention_roboreward_8b_native_rank.yaml
CUDA_VISIBLE_DEVICES=2 python rewardbench/run_roboreward_attention.py steer --config rewardbench/configs/attention_roboreward_8b_native_reward1.yaml
python rewardbench/run_roboreward_attention.py score --config rewardbench/configs/attention_roboreward_8b_native_reward1.yaml
```

reward=5 使用对应的 `attention_roboreward_8b_native_reward5.yaml`。attention 原生视频统一限制
为 8 帧；该限制只用于 attention tensor 的可控计算。条件、控制和结果见
`rewardbench/attention_steering_exp_record.md`。

## 论文 Overall（23-subset group-wise MAE）

论文的 `Overall (MAE)` 不是将所有样本混合后的 micro MAE，而是 23 个固定数据子集 MAE 的
等权平均。`score` 会额外写出 `paper_protocol_metrics.json`，并只在记录完整、每条都是严格
`ANSWER: <1-5>` 的原生离散结果且刚好覆盖全部 23 个子集时，将
`paper_metric_comparable` 标为 `true`。reward=1/5 审核 cohort 不会被错误地报告为论文 Overall。

```bash
CUDA_VISIBLE_DEVICES=2 python rewardbench/run_roboreward_eval.py run \
  --config rewardbench/configs/full_roboreward_8b_paper_protocol.yaml
python rewardbench/run_roboreward_eval.py score \
  --run-dir rewardbench/roboreward_eval/outputs/full_8b_paper_protocol
```

论文没有公开独立 benchmark evaluator；因此该实现是对已发表统计定义的可审计复现，而不是对
未公开官方代码的声称性复刻。配置保留 checkpoint 原生 MP4 processor，避免把论文数据构建的
1 FPS 描述误作推理输入规定。
