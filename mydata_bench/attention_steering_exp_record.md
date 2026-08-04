# 跨模型 attention steering 实验记录

## 固定设计与完整性

- Ranking：carrot、bottle、cube 三条独立成功轨迹的最终 bbox；每模型 36×32 heads，排除第 0--1
  层，对三条 source ranking 做 normalized-Borda 共识，取 top-8。三个共识文件均有 1,088 个
  eligible heads 和三份完整 source artifact。
- Cohort：reward=1 为 SAM3 单人审核后冻结的 111 IDs；reward=5 为冻结的 112 IDs。所有模型
  输入 JSONL 均不含 reward 标签；评分阶段才回连标签。
- 条件：baseline、candidate_target、candidate_wrong、low_rank_target。所有六组的 3×N 个
  steering 条件均记录到 hook `applied_calls>0`。默认错误区域是同目标视觉平面的等尺寸最远
  区域；当粗网格使其不可用时使用等数量、严格不重叠的其它视觉 token 回退，并在记录的
  `control_region` 字段标明。
- 原生 MP4 固定最多 8 帧；Qwen 前向为八图 `<score>` adapter，不能与原生离散指标合并。

## 结果

下表为 Exact / MAE；所有 six runs 的 `formal_scoring_ready=true`。前向列的离散分数由 progress
诊断性映射而来。

| 模型与协议 | cohort | baseline | target | wrong | low-rank target |
|---|---|---|---|---|---|
| Qwen 原生 | r1 (111) | 43.24% / 0.820 | 43.24% / 0.820 | 43.24% / 0.829 | 44.14% / 0.802 |
| Qwen 原生 | r5 (112) | 30.36% / 1.616 | 29.46% / 1.652 | 30.36% / 1.634 | 30.36% / 1.643 |
| Qwen 八图前向（adapter） | r1 (111) | 87.39% / 0.432 | 87.39% / 0.432 | 87.39% / 0.432 | 87.39% / 0.432 |
| Qwen 八图前向（adapter） | r5 (112) | 37.50% / 2.080 | 37.50% / 2.080 | 37.50% / 2.080 | 36.61% / 2.098 |
| RoboReward 原生 | r1 (111) | 36.94% / 1.027 | 36.94% / 1.027 | 36.94% / 1.027 | 36.94% / 1.027 |
| RoboReward 原生 | r5 (112) | 27.68% / 1.304 | 27.68% / 1.295 | 26.79% / 1.321 | 27.68% / 1.312 |

本配置的单步 `last_prompt` steering 效应总体很小；只有 Qwen 原生 reward=1 的低排名控制出现
轻微改善，且未在其它 cohort/模型中复现。因此这些结果证明实现与控制链路可运行，但不支持把
该强度和 query scope 下的差异解释为稳健的性能增益。

## 2026-07-30：skip-8 与全程 attention mask 重跑

本节是对上述 `skip-2 + last_prompt` 运行的独立替代实验，输出目录名带
`skip8_allq`，不得和旧表混合。ranking 排除 layer 0--7，只从 896 个 layer 8--35 heads 中
选 top-8。所有 steering 配置为 `steering_query_scope=all`：mask 对 selected ranking heads 的
所有 prefill query 行和每个 decode query 行均生效。六份真实 record 的逐层 diagnostics 全部
满足 prefill/decode 的 applied calls 等于对应 calls。

原生视频 endpoint bbox 只映射到 processor 采样序列中含真实终帧的最后一个 video token span。
三条 ranking 视频的终帧核验分别为 carrot `665/666`、bottle `869/870`、cube `672/673`
（“终帧索引 / 总帧数”）；最后 token span 均为 `video_t3`。

| 模型与协议 | cohort | baseline Exact / MAE | target Exact / MAE | wrong Exact / MAE | low-rank Exact / MAE |
|---|---|---|---|---|---|
| Qwen 原生 | r1 (111) | 43.24% / 0.820 | 27.03% / 1.036 | 35.14% / 0.883 | 38.74% / 0.919 |
| Qwen 原生 | r5 (112) | 30.36% / 1.616 | 42.86% / 1.241 | 22.32% / 2.018 | 31.25% / 1.607 |
| Qwen 八图前向（adapter） | r1 (111) | 87.39% / 0.432 | 86.49% / 0.432 | 87.39% / 0.414 | 81.08% / 0.703 |
| Qwen 八图前向（adapter） | r5 (112) | 37.50% / 2.080 | 35.71% / 2.062 | 33.04% / 2.134 | 33.04% / 2.464 |
| RoboReward 原生 | r1 (111) | 36.94% / 1.027 | 42.34% / 0.856 | 36.04% / 1.063 | 38.74% / 0.982 |
| RoboReward 原生 | r5 (112) | 27.68% / 1.304 | 45.54% / 1.107 | 25.89% / 1.402 | 29.46% / 1.295 |

全程 target steering 在 Qwen 原生 reward=5、RoboReward 原生 reward=1/5 上提升明显，同时
同 head 的 wrong-region control 往相反方向移动，符合目标区域选择具有效应的预期；但 Qwen
原生 reward=1 和八图 adapter 的表现不同，说明效应有 cohort/协议依赖性。八图结果仍只能作为
adapter 消融，不能和原生离散模型做统一排名。

## skip-8 ranking head 的跨模型重合度

以下比较使用三个单源 `head_ranking.json` 在过滤 layer 0--7 后的前 8 个 head，而不是原始文件
未经筛选的前 8 行。重合度为 Jaccard：`|交集| / |并集|`；每一集合大小固定为 8。

| 数据 | Qwen 原生 top-8（layer, head） | RoboReward 原生 top-8（layer, head） | 交集 | 重合度 |
|---|---|---|---|---:|
| carrot | `(28,30),(19,23),(19,17),(28,19),(21,3),(10,1),(19,14),(13,24)` | `(22,15),(19,23),(26,31),(20,1),(19,17),(19,28),(26,26),(31,19)` | `(19,17),(19,23)` | 2/14 = 14.29% |
| bottle | `(28,30),(19,23),(8,15),(10,1),(14,23),(34,12),(10,13),(28,31)` | `(22,15),(20,15),(19,23),(21,16),(26,19),(23,13),(19,21),(10,13)` | `(10,13),(19,23)` | 2/14 = 14.29% |
| cube | `(21,16),(19,23),(21,25),(20,15),(22,15),(21,26),(23,13),(24,13)` | `(22,15),(21,25),(21,26),(20,15),(19,23),(25,5),(26,19),(21,16)` | `(19,23),(20,15),(21,16),(21,25),(21,26),(22,15)` | 6/10 = 60.00% |

Qwen 原生与 RoboReward 原生在 carrot/bottle 的重合仅为 2 个头，但 cube 共享 6 个中后层 head，
说明两种 checkpoint 对 cube 端点视觉证据的 attention 偏好高度相近，而这不是三数据都共有的
模式。三个数据唯一持续重复的跨模型 head 是 `(19,23)`。

| 数据 | Qwen 八图前向 top-8（layer, head） | 与 Qwen 原生的交集 / 重合度 | 与 RoboReward 原生的交集 / 重合度 |
|---|---|---|---|
| carrot | `(14,23),(10,2),(8,11),(8,20),(10,1),(14,3),(10,14),(11,21)` | `(10,1)` / 1/15 = 6.67% | 无 / 0% |
| bottle | `(8,11),(14,23),(10,2),(8,20),(8,22),(15,25),(10,1),(9,3)` | `(10,1),(14,23)` / 2/14 = 14.29% | 无 / 0% |
| cube | `(10,2),(8,11),(14,23),(8,20),(31,11),(10,1),(9,3),(10,14)` | 无 / 0% | 无 / 0% |

八图前向 ranking 集中在 layer 8--15（cube 另有 `(31,11)`），与两种原生 MP4 ranking 的重合很低；
这与八图提示词、八个视觉 span 和原生视频 temporal-token 输入不同相一致。因此八图前向的
ranking/steering 应作为独立 adapter 协议解读，不能假定其 head 与原生模型共享。

### 三源 Borda 共识 top-8

| 协议 | 共识 top-8 | 与 Qwen 原生交集 | 与 RoboReward 原生交集 |
|---|---|---|---|
| Qwen 原生 | `(19,23),(28,30),(10,1),(21,3),(14,23),(8,15),(19,21),(19,17)` | — | `(19,17),(19,23)`（2/14 = 14.29%） |
| Qwen 八图前向 | `(8,11),(10,2),(14,23),(8,20),(10,1),(9,3),(10,14),(14,3)` | `(10,1),(14,23)`（2/14 = 14.29%） | 无（0%） |
| RoboReward 原生 | `(22,15),(19,23),(20,15),(23,13),(21,16),(19,17),(21,15),(26,19)` | `(19,17),(19,23)`（2/14 = 14.29%） | — |

## 2026-07-31：target-span 修复与 GRM raw-mean-12 同构重跑

本节取代旧 Qwen adapter 的因果解释，并作为新的主跨模型 exploratory 结果。旧 adapter
ranking/steering 将 `after_cam_high` 错绑定到 before image span，数值无效；旧 skip-8 章节仅保留
为历史记录。

- Qwen adapter 目标现固定为第 6 个 image span，并同时检查它的 path 等于终态 cam-high 图。
- Ranking 与 GRM 对齐为每任务 12 个 progressive samples、last-prompt raw bbox mass、逐任务算术
  平均、三任务 normalized Borda、`skip_early_layers=2`。Qwen/RoboReward native 共用完全相同的
  截断 MP4。
- 三个 protocol 的 processor alignment 均 36/36 通过；六个 steering run 均四条件完整、
  `invalid_count=0`、`formal_scoring_ready=true`。
- 完整数值、cluster CI、跨模型 ranking/effect 比较及解释边界见
  `results/attention/exp_record_summary_0730.md` 和
  `results/attention/cross_model_rawmean12_comparison.json`。

| 模型 / protocol | cohort | baseline Exact / MAE | target Exact / MAE | cluster target shift（95% CI） |
|---|---|---|---|---|
| Qwen adapter | r1 | 87.39% / 0.432 | 81.98% / 0.604 | +0.0441 `[-0.0212,+0.1126]`（归一化连续 score） |
| Qwen adapter | r5 | 37.50% / 2.080 | 33.93% / 1.795 | +0.0797 `[+0.0099,+0.1486]`（归一化连续 score） |
| Qwen native | r1 | 46.85% / 0.721 | 41.44% / 0.748 | +0.0270 `[-0.0901,+0.1532]`（1–5 分） |
| Qwen native | r5 | 44.64% / 1.205 | 56.25% / 1.027 | +0.1622 `[+0.0090,+0.3153]`（1–5 分） |
| RoboReward native | r1 | 36.94% / 1.027 | 31.53% / 1.045 | +0.0180 `[-0.1081,+0.1351]`（1–5 分） |
| RoboReward native | r5 | 27.68% / 1.304 | 43.75% / 1.071 | +0.2162 `[+0.0811,+0.3604]`（1–5 分） |

结论：三套新 own-head ranking 都没有复现 GRM 的 r1 负向纠错；两个 native 离散模型在 r5
上改善，adapter 的 r5 平均序数误差改善但 Exact 下降。Qwen native 与 RoboReward native 的
全 ranking Spearman 为 0.850，但 top-8 只重合 2/8；当前证据支持“全局顺序相关、截断 head
身份不稳定”，不支持固定跨 checkpoint circuit。
