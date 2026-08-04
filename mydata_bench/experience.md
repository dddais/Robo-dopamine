# 跨模型 attention steering 工作记录

## 2026-07-29：现状审计与实施决策

- 目标模型为 RoboReward-8B 与 base Qwen3-VL-8B-Instruct；后者需同时支持
  `roborewardbench_native`（原始 MP4、`ANSWER: <1-5>`）和
  `robo_dopamine_forward`（8 张端点图、`<score>`）两种 I/O。
- `aligned_data/pick3suc_1_carrot`、`pick3suc_3_bottle`、`pick3suc_4_cube`
  均存在完整成功轨迹的 `test.mp4` 与三路相机视频；任务文本分别为“pick the carrot / bottle /
  white cube and put it on yellow plate”。已有 GRM 的三份 36×32 完整 ranking 产物，可作为
  格式和方法参考，但**不**直接复用其 head 排名到不同模型。
- 三个模型 checkpoint 都是 Qwen3-VL 架构：36 decoder layers、32 query heads、
  `spatial_merge_size=2`。因此可以保持 GRM 的核心做法：最后 prompt query 对目标 bbox
  视觉 token 的 excess attention mass 排名；steering 在 selected heads 的 pre-softmax
  attention mask 上对目标 token 加 bias、对其余视觉 token 减 bias，并保留目标/错误位置/
  低排名 head 控制。
- Qwen 前向格式可直接采用 GRM 的 8 图顺序和 token-span 对齐。两种原生 MP4 格式需要按
  `video_grid_thw` 和 `frames_indices` 构造视频 token span，并只对包含真实末帧的末个时间平面
  应用人工审核的末帧 bbox；这是避免将 endpoint bbox 错投到其它视频帧的关键实现点。
- ranking 使用上述三条成功轨迹，不从 reward benchmark 抽样；ranking 数据与 reward=1/5
  steering cohort 严格分离。GPU 仅允许使用 0、1、2。

## 2026-07-29：原生 token 对齐探针

使用 reward=5 人工审核 cohort 的一条样本，在 GPU 0--2 各运行一次真实 eager-attention
forward。三种 runtime 均成功返回 36×32 attention mass：

| 模型 / 输入格式 | 原生视觉 span | target bbox token 数 | 全视觉 token 数 |
|---|---|---:|---:|
| Qwen3-VL-8B / 原始 MP4 | 4 个时间 span，每个 300 token | 20 | 1,200 |
| Qwen3-VL-8B / Robo-Dopamine 八图 | 8 个图像 span | 6 | 535 |
| RoboReward-8B / checkpoint-native MP4 | 4 个时间 span，每个 88 token | 9 | 352 |

- 原生 MP4 metadata 的 `frames_indices` 为 `[0, 6, 11, 17, 23, 29, 34, 40]`，包含真实末帧
  40；`video_grid_thw` 的时间维为 4，因为每个视觉 token 时间平面合并两个视频帧。实现选择
  最后一个 contiguous video-token span，故人工审核的末帧 bbox 与输入语义一致。
- Qwen 原生 checkpoint 默认最多允许 768 帧，完整 attention 代价不可控；attention runtime
  显式采用 `attention_video_max_frames=8`，与 RoboReward checkpoint 默认一致。该限制仅用于
  新 attention 实验，不追溯改变已经完成的全量 baseline 评测。

## 2026-07-29：ranking 与 reward=1/5 steering 完成

- 三个真实 ranking 已完成，均为 carrot/bottle/cube 三源、36×32、排除前两层后的 1,088 heads。
  Qwen 原生 top-8 为 `(19,23),(4,25),(28,30),(10,1),(14,23),(21,3),(8,15),(5,11)`；Qwen 前向
  为 `(3,3),(3,29),(10,2),(8,11),(14,23),(4,25),(8,20),(7,28)`；RoboReward 为
  `(22,15),(19,23),(20,15),(21,16),(23,13),(19,17),(26,19),(21,15)`。
- 六个 steering 运行均完整：reward=1 为 111×4、reward=5 为 112×4，每个 steered record 的
  所有 selected layer 都有 `applied_calls>0`。少数粗视觉网格令 bbox 占满目标平面，故引入并
  明确记录等数量、不重叠的其它视觉 token 控制回退；未删除或隐瞒原始失败尝试。
- 指标与结论已写入 `attention_steering_exp_record.md`。前向结果保持 adapter 边界；原生 8 帧
  attention cap 不影响既有 full baseline。

## 2026-07-30：skip-8、全 query / 全阶段 steering 重跑

- 按新口径重新生成三套 ranking：`skip_early_layers=8`，每份 consensus 的 eligible heads 为
  896（layer 8--35）。新 top-8 与旧结果分目录保存，未覆盖旧的 skip-2 ranking。
- steering 改为 `steering_query_scope=all`。真实 records 的每个 selected layer 都满足
  `prefill_applied_calls=prefill_calls` 与 `decode_applied_calls=decode_calls`，且 prefill 的
  `applied_query_rows>1`；因此 mask 覆盖所有 prefill query 行及每一 decode query 行。
- 原生 video processor 现在强制要求 `frames_indices[-1]=total_num_frames-1`。最后 video token
  span（`video_t3`）才接受 endpoint bbox；当最终 temporal group 的内部 padding 令帧数不能整除
  token 时间平面数时，仍记录终帧而不伪造逐帧映射。
- reward=1（111 IDs）和 reward=5（112 IDs）的六组运行均 `formal_scoring_ready=true`。结果已
  追加到 `attention_steering_exp_record.md`。

## 2026-07-31：RoboRewardBench 论文 Overall 评分口径固化

- 论文 arXiv:2601.00675v2 的 Overall MAE 为 23 个 benchmark subset MAE 的无权平均，不是全样本
  micro MAE。已在 `roboreward_eval/paper_protocol.py` 固化这 23 个 subset 的顺序、公开 Table all
  的 RoboReward-8B 参考列和严格输入检查。
- 对完整 `full_8b_checkpoint_native_new` records 复评分：23/23 subset、2,831/2,831 有效，论文口径
  group-wise MAE 为 `0.6498847812`（显示为 0.650），论文 8B 报告 0.665，差 -0.015；micro MAE
  为 0.686，二者不可混用。
- 评分器只将完整的 23-subset 原生离散 `ANSWER: <1-5>` records 标为 `paper_metric_comparable=true`；
  reward=1/5 SAM3 审核 cohort 会明确拒绝生成论文 Overall。论文尚未公开独立 evaluator，因此该层是
  已公开统计定义的可审计复现，而非声称使用了作者未公开代码。
