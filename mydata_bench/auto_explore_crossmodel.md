# 跨模型 attention steering 自动研究记录

> 状态：进行中  
> 启动时间：2026-08-13（Asia/Shanghai）  
> 研究约束：严格遵循 `exp_plan_crossmodel_auto_explore.md`；不执行 git 操作；不删除或覆盖既有数据/结果；代码与配置只做增量新增；模型输入不含 reward 标签，标签仅在评分阶段回连。

## 1. 研究问题与判定标准

本轮围绕两个主线：

1. 为 RoboReward-8B 和 Qwen3-VL-8B 找到相对各自 baseline 明显改善、并优于 wrong-region/low-rank 对照的 steering 配置，同时解释 GRM 效果更稳定的原因。
2. 在固定 attention bias 之外提出增量式改良机制，使两种离散模型的 MAE、suc/fail accuracy 和总体 accuracy 稳定改善，并解释机制。

判定时同时检查：总体 MAE/exact、suc/fail exact、逐 task、预测分布、同视频不同 instruction 的 pairwise 差值、candidate-vs-wrong/low-rank 特异性、运行完整性以及以 video SHA 为 cluster 的区间估计。探索性自动 grounding 结果不外推为普遍因果结论。

## 2. 环境与执行状态盘点

- 数据：`/home/dais/workspace/data/mydata_v2/new`；自动 grounding cohort：`results/mydata_bench/cohorts/auto_grounded_v2`。
- 既有跨模型输出：`results/mydata_bench/experiments_v2_corssmodel`（沿用原计划中的拼写）。
- GPU 1、2 正运行既有 GRM official 任务，启动已约 20 小时；本研究不干扰它们，只使用空闲 GPU 0。
- `auto_explore.md` 启动时为空；跨模型目录中已有 16 组 2026-08-10 完成的 baseline/steering 真实运行，可先作高价值复盘，避免重复耗算力。
- 现有模型实验环境为 `robo-dopamine`；模型实验命令和 processor alignment 契约见 `exp_use_crossmodel.md`。

## 3. 已有真实实验的首轮复盘

### 3.1 一个已满足 RoboReward 主线目标 1 的强配置

配置：`attention_13_roboreward_interleaved_all_frames`，即 GRM 式交错输入、8 个独立 image span、all-frame target/non-target 互补 bias、all-query、own-model ranking、top-32 heads、swap bias 6。

完整 846 条样本、10 个条件均无 invalid，`formal_scoring_ready=true`：

| 条件 | Exact | MAE | 相对 baseline 的聚类绝对误差变化 95% CI |
|---|---:|---:|---:|
| baseline | 23.52% | 1.5674 | — |
| candidate target, k=32 | **56.15%** | **0.9468** | **[-0.7768, -0.6070]** |
| wrong region, k=32 | 8.16% | 1.7246 | [0.0477, 0.1700] |
| low-rank target, k=32 | 19.15% | 1.6525 | [0.0301, 0.1263] |

配对区分度也明显增强：同视频 suc−fail 为 4 的比例从 4.42% 升至 27.07%，负差从 5.16% 降至 1.66%。candidate 修正 326 条、损害 50 条；wrong-region 只修正 29 条、损害 159 条。效应同时具备幅度、cluster CI 和空间/head 对照特异性，不是“任意扰动都有效”。

### 3.2 Qwen 的强候选及完整性问题

- 完整且严格解析的 image-sequence 配置中，`attention_08_qwen_images_text` 的 target k=64：Exact 22.81%→45.63%，MAE 1.4326→1.0449；wrong 为 21.39%/1.4527，low-rank 为 23.40%/1.4374。
- 完整且严格解析的 `attention_07_qwen_text_images` target k=64：Exact 28.25%→46.69%，MAE 1.5934→1.2506；wrong 为 24.35%/2.0863，low-rank 为 28.01%/1.6241。
- GRM 式交错配置 `attention_15/16` 的 target 条件也有大幅改善，但分别有 72/80 条严格输出格式失败（模型输出裸 `5` 而非 `ANSWER: 5`），故只能作为 exploratory，不能作为正式完成证据。

### 3.3 初步机制线索

1. **干预强度不是越大越好**：RoboReward interleaved all-frame 在 k=32 达峰，k=64 明显回落；Qwen 多数配置则在 k=64 更好。固定 top-k 不具有跨模型可移植性。
2. **时序覆盖很关键**：RoboReward interleaved all-frame k32 强增益，而同 prompt 的 last-frame k32 反而 MAE 变差；支持“完整视频的时序证据稀释单终帧 instruction-grounded 信号”的假设。
3. **prompt/media 顺序改变基线与剂量响应**：RoboReward images→text 的 baseline 已较强，all-frame k64 过度干预；text→images/interleaved 的低基线从 k32 得益更大。
4. **GRM 更稳定的结构性原因候选**：GRM 只比较 before/after 三视图、输出连续 progress，目标视觉 token 与 readout 距离短；离散模型包含 8 帧、自回归 rubric 生成和 KV-cache decode，attention bias 同时改变视觉路由、语言生成与离散阈值跨越，因此 head 截断和剂量更敏感。



## 5. 主线目标 1：最终配置与证据

### 5.1 RoboReward-8B

最终配置采用既有真实运行 `attention_13_roboreward_interleaved_all_frames` 的 `candidate_target_k32`：

- 输入：8 个独立 image span，文本与图像按 GRM 风格交错；
- temporal scope：all frames；每帧 tracking bbox 对应 token 加 `+6`，同帧其余视觉 token 加 `-6`；
- query scope：prefill 与 decode 的所有 query；
- ranking：本模型 raw attention mass，跳过前 8 层，取 top-32；
- 完整性：846/846 样本、10/10 条件完整，invalid=0，`formal_scoring_ready=true`；processor alignment 已通过；
- 指纹：config `bf69367e...cd9c`，model `e9bd763c...3821`，ranking SHA `679a7a00...bf7b`。

| 指标 | baseline | target k32 | wrong k32 | low-rank k32 |
|---|---:|---:|---:|---:|
| 总体 Exact | 23.52% | **56.15%** | 8.16% | 19.15% |
| MAE | 1.5674 | **0.9468** | 1.7246 | 1.6525 |
| suc Exact | 48.88% | **51.49%** | 5.22% | 37.69% |
| fail Exact | 11.76% | **58.30%** | 9.52% | 10.55% |

target 相对 baseline 的任务分层、video-cluster bootstrap 绝对误差变化均值为 -0.6912，95% CI `[-0.7768,-0.6070]`；修正/损害为 326/50。543 个有效 suc/fail pair 中，差值 4 从 24（4.42%）升至 147（27.07%），负差从 28（5.16%）降至 9（1.66%）。因此该配置同时满足 MAE、总体/suc/fail exact 和 pairwise，并明显优于空间与 head 对照。

### 5.2 Qwen3-VL-8B

最终配置采用严格完整的真实运行 `attention_08_qwen_images_text` 的 `candidate_target_k64`：

- 输入：8 个独立 image span，images→text；
- temporal scope：last frame，target/non-target bias 只在终帧 span 内互补；
- query scope：all；swap bias 6；
- ranking：本模型 raw attention mass，跳过前 8 层，取 top-64；
- 完整性：846/846 样本、10/10 条件完整，invalid=0，`formal_scoring_ready=true`；processor alignment 已通过；
- 指纹：config `55eb81c2...3f65`，model `5c5bf5cf...3b57`，ranking SHA `5c841e88...bf29`。

| 指标 | baseline | target k64 | wrong k64 | low-rank k64 |
|---|---:|---:|---:|---:|
| 总体 Exact | 22.81% | **45.63%** | 21.39% | 23.40% |
| MAE | 1.4326 | **1.0449** | 1.4527 | 1.4374 |
| suc Exact | 55.97% | **59.70%** | 38.06% | 58.58% |
| fail Exact | 7.44% | **39.10%** | 13.67% | 7.09% |

target 的 video-cluster 绝对误差变化 95% CI 为 `[-0.4643,-0.3300]`。543 个有效 pair 中，差值 4 从 20（3.68%）升至 169（31.12%），负差从 39（7.18%）降至 24（4.42%）。该配置同样满足全部主线 1 指标，并优于 wrong/low-rank。

### 5.3 为什么早期判断是“效果不稳定”，而当前能找到强配置

早期搜索主要落在 top-8、last-frame 或 native temporal span 上；完整矩阵显示关键变量之间存在强交互，而不是单个开关：

1. **head 剂量依模型而变**：RoboReward interleaved-all 在 k32 达峰，k64 Exact 回落到 23.29%；Qwen 两种 image-sequence 顺序则多在 k64 达峰。top-8 不是可迁移默认值。
2. **RoboReward 需要全时序目标覆盖**：同一 interleaved prompt 下，all-frame k32 的 MAE 为 0.9468，而 last-frame k32 为 1.6182（比 baseline 1.5674 更差）。这直接支持 temporal-prior/稀释假设。
3. **Qwen 更依赖终帧与 prompt adjacency**：images→text 把 rubric/readout 放在视觉 token 之后，终帧 target 对后续语言 token 因果可见；k64 提供了比 k32 更宽的中层路由覆盖。
4. **ranking 顶端不稳定、宽集合更稳定**：跨模型 top-8 交集通常仅 25%–50%，但 interleaved top-32 交集达到 75%–78%，top-64 达 76.6%–85.9%。因此只取极少 head 容易受 prompt/checkpoint 排名噪声影响。



## 7. GRM、RoboReward、Qwen 差异的原理解释

### 7.1 证据、推断与边界

**直接证据：**

- GRM incremental-after k32 的描述性 endpoint accuracy（0.2/0.8）从 57.21% 升至 70.09%，fail 从 82.35% 升至 97.58%；连续/离散 MAE 也下降。
- GRM 的 4 组 v2 运行均显示 target shift，且 bbox attention mass increase rate 为 1.0。
- 但现有 GRM `target_head_specific_causal_effect_supported=false` 的直接原因是 formal gate 标记 `exploratory_unaudited_auto_grounding`；并非可以宣称已通过人工审核的正式因果验证。
- 离散模型的效果对 top-k、媒体顺序和 temporal scope 呈强交互；改为 8 个独立 image span 并未自动消除不稳定。

**最符合证据的机制推断：**

1. **readout 距离与输出几何。** GRM 对 before/after 的连续 progress 直接读出；attention routing 的小幅连续变化可直接体现在 score。RoboReward/Qwen 要经过自回归语言生成并跨 1–5 离散边界，小变化可能无效，大变化又会越界。
2. **时序稀释。** GRM 只比较两个时刻、三视图，目标状态证据集中；8 帧视频有冗余时间先验。RoboReward all-frame 的巨大增益和同 prompt last-frame 的失败是该解释的最直接消融支持。
3. **causal mask 与 prompt adjacency。** 自回归模型只有后置 query 能读视觉 key。images→text/交错协议让 rubric/readout token 位于视觉证据之后；纯 text→images 的部分语言 token 无法回看未来 image，降低 instruction-to-vision 对齐的可控性。
4. **head ranking 是协议条件化的。** 同一模型仅改变输入顺序，top-8 overlap 可低至 12.5%；但 top-32/64 更稳定。这解释了 GRM ranking 方案直接迁移 top-8 时的脆弱性。


### 7.2 不能由本轮推出的结论

- 不能把自动 grounding cohort 的结果写成经人工审核的正式普遍因果效应。
- 不能声称 video tubelet 是唯一原因；独立 images 能改善可控性，但 prompt 顺序/top-k/时序覆盖同样关键。
- 不能声称一个固定 top-k 或 prompt 对所有模型最优。






## 9. 最终结论

两个主线目标在当前自动 grounding cohort 上均已完成：

1. **主线 1：** RoboReward 的 GRM-style interleaved/all-frame/top-32/bias-6 和 Qwen 的 images→text/last-frame/top-64/bias-6 均在 846 条完整真实推理上显著改善 MAE、总体/suc/fail Exact 与 pairwise，且明显优于 wrong-region/low-rank。

