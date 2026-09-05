# v2 原生视频、GRM 与 attention steering 实验总结

## 1. 实验范围与统计口径

本总结对应 [exp_plan.md](./exp_plan.md)，覆盖 `results/mydata_bench/experiments_v2/` 中的全部 **7 个 baseline 目录和 12 个 attention 目录**，包括后续新增的官方 GRM incremental 结果。结构参照 [跨模型总结](./exp_plan_crossmodel_summary.md)。统计于 2026-09-05 从原始 JSON/JSONL 重新核对；本次只新增总结，不重跑模型，也不改写已有实验记录。

- **Baseline**：完整数据集 1213 条，407 suc / 806 fail。
- **Attention**：自动 grounding cohort 846 条，268 suc / 578 fail；按既定计划接受自动 tracking 结果，不增加人工审核门槛。steering 必须与同实验、同 cohort 的 baseline 比较。
- **原生视频模型**：RoboReward-8B、Qwen3-VL-8B-Instruct；离散输出 1–5，MAE 为 `mean(abs(prediction-label))`，suc=5、fail=1，准确率要求完全相等。
- **GRM**：Robo-Dopamine-GRM-2.0-8B-Preview；离散 MAE 沿用项目四边界 `0.125/0.375/0.625/0.875`，另列连续 ordinal MAE `mean(abs(1+4*progress-label))`。准确率报告 `0.125/0.875` 与 `0.2/0.8` 两套端点阈值，中间值算不正确；预测分布的五个等宽区间另行统计，不能与 MAE 分档混用。
- **干预**：top-k=8/32/64、bias=6、all-query；比较 candidate-target、wrong-region 和 low-rank-head。GRM 不带 `_k8` 的三个条件是 top-8 别名，不是独立重复实验。
- **输入口径**：原生视频 attention 上限为 8 帧，保留二帧 temporal tubelet，最后帧干预实际映射到包含终帧的 tubelet；GRM 干预 high-camera 的 after 或 before+after 图像。完整 baseline 与 attention 的采样、后端及生成设置也可能不同，不能把差异全部归因于 cohort 筛选。

**必须区分旧版与官方 incremental。** 旧 `baseline_06_grm_incremental` 和不带 `_official` 的 attention 12/13 只评估 `terminal-20 → terminal` 的单个局部 hop。官方版从视频起点按 20 帧间隔逐 hop 推理并递推累计，每个条件独立累计，最终 progress 裁剪到 `[0,1]`；官方 attention 作用于所有 hop。旧目录只作为 local-hop 历史诊断，不用于整段进度结论。修正依据见 [experience.md](./experience.md) 的“GRM incremental 口径修正”和 [exp_use.md](./exp_use.md) 第 12 节。

## 2. Baseline：输入顺序与进度定义影响明显

### 2.1 原生视频离散模型

| 实验 / 输入 | MAE | 总准确率 | suc 准确率 | fail 准确率 |
| --- | --- | --- | --- | --- |
| [01 RoboReward，text → video](../results/mydata_bench/experiments_v2/baseline_01_roboreward_text_video/exp_record.md) | 2.1187 | 23.08% | 68.80% | 0.00% |
| [02 RoboReward，video → text](../results/mydata_bench/experiments_v2/baseline_02_roboreward_video_text/exp_record.md) | 1.1072 | 60.35% | 54.30% | 63.40% |
| [03 Qwen，text → video](../results/mydata_bench/experiments_v2/baseline_03_qwen_text_video/exp_record.md) | 1.6966 | 31.49% | 84.03% | 4.96% |
| [04 Qwen，video → text](../results/mydata_bench/experiments_v2/baseline_04_qwen_video_text/exp_record.md) | 1.6059 | 35.28% | 83.78% | 10.79% |

RoboReward 从 text → video 改为 video → text 后，MAE 从 2.1187 降至 1.1072，总准确率从 23.08% 升至 60.35%；fail 准确率从 0 升至 63.40%，但 suc 从 68.80% 降至 54.30%。Qwen 的顺序收益较小：MAE 1.6966 → 1.6059，总准确率 31.49% → 35.28%，suc 基本不变，fail 仍只有 10.79%。因此输入顺序十分重要，但不同模型收益幅度不同，也不保证两类同时受益。

### 2.2 GRM：正式 episode progress 与历史 local-hop

下表每个准确率单元格依次为 **总 / suc / fail**。

| 实验 | 离散 MAE | 连续 ordinal MAE | 准确率 0.125/0.875 | 准确率 0.2/0.8 |
| --- | --- | --- | --- | --- |
| [05 GRM forward](../results/mydata_bench/experiments_v2/baseline_05_grm_forward/exp_record.md) | 1.9802 | 1.9673 | 5.28% / 15.48% / 0.12% | 10.47% / 30.96% / 0.12% |
| [06 GRM local-hop（历史）](../results/mydata_bench/experiments_v2/baseline_06_grm_incremental/exp_record.md) | 1.3899 | 1.4491 | 40.81% / 1.47% / 60.67% | 54.74% / 1.47% / 81.64% |
| [06 official GRM incremental](../results/mydata_bench/experiments_v2/baseline_06_grm_incremental_official/exp_record.md) | 2.1871 | 2.1807 | 12.45% / 36.61% / 0.25% | 24.15% / 71.50% / 0.25% |

官方 incremental 相比 forward 提高了 suc 端点准确率，但离散 MAE 反而从 1.9802 升到 2.1871，fail 准确率仍接近零。旧 local-hop 的较低 MAE 和较高总体准确率伴随 suc 仅 1.47%，且评估对象不同，不能据此声称“incremental 整体优于 forward”。

## 3. Attention steering 结果

### 3.1 RoboReward 与 Qwen：最佳 MAE 不等于两类同时改善

每组选择 **candidate-target 中 n=846 的最低 MAE 点**；箭头左侧为本实验同 cohort baseline。

| 实验 / 干预 | k | MAE | 总准确率 | suc 准确率 | fail 准确率 |
| --- | --- | --- | --- | --- | --- |
| [06 RoboReward，text → video / 最后帧](../results/mydata_bench/experiments_v2/attention_06_roboreward_last_frame/exp_record.md) | 32 | 2.1206 → 1.3487 | 24.94% → 23.76% | 78.73% → 68.28% | 0.00% → 3.11% |
| [07 RoboReward，text → video / 全帧](../results/mydata_bench/experiments_v2/attention_07_roboreward_all_frames/exp_record.md) | 32 | 2.1206 → 1.2092 | 24.94% → 25.65% | 78.73% → 63.81% | 0.00% → 7.96% |
| [08 Qwen，text → video / 最后帧](../results/mydata_bench/experiments_v2/attention_08_qwen_last_frame/exp_record.md) | 64 | 1.4586 → 1.0319 | 29.20% → 41.96% | 89.55% → 73.13% | 1.21% → 27.51% |
| [09 Qwen，text → video / 全帧](../results/mydata_bench/experiments_v2/attention_09_qwen_all_frames/exp_record.md) | 64 | 1.4586 → 1.1608 | 29.20% → 35.22% | 89.55% → 67.54% | 1.21% → 20.24% |
| [14 RoboReward，video → text / 最后帧](../results/mydata_bench/experiments_v2/attention_14_roboreward_last_frame/exp_record.md) | 32 | 0.8333 → 0.8097 | 69.15% → 64.42% | 68.28% → 59.70% | 69.55% → 66.61% |
| [15 RoboReward，video → text / 全帧](../results/mydata_bench/experiments_v2/attention_15_roboreward_all_frames/exp_record.md) | 8 | 0.8333 → 0.7920 | 69.15% → 70.92% | 68.28% → 70.90% | 69.55% → 70.93% |

1. **原生视频六组实验中，只有 RoboReward 实验 15 的 k=8 同时降低 MAE、提高总/suc/fail 准确率。** MAE 下降 0.0414，总准确率提高 1.77 个百分点，suc/fail 分别提高 2.61/1.38 个百分点；这是小幅、单个配置的描述性改善。
2. RoboReward 的 text → video 在 k=32 时 MAE 大幅下降，但 fail 主要从高分转为 2 分，而不是正确的 1 分。实验 07 的 MAE 下降 0.9113，总准确率却只提高 0.71 个百分点，suc 下降 14.93 个百分点。
3. Qwen 两组在全部 k 下都降低 MAE，但 suc 准确率均低于 baseline；最后帧 k=64 比全帧 k=64 更好。全帧并不是普适改进。
4. RoboReward video → text 已有较强 baseline，强干预容易破坏它：实验 15 从 k=8 增至 k=64 后，MAE 升至 1.3570，总准确率降至 31.68%。实验 14 最低 MAE 点 k=32 的两类准确率也都下降。

### 3.2 GRM 正式协议：forward 最一致，官方 incremental 依赖干预范围

下表同样按完整 candidate-target 的最低离散 MAE 选点。准确率为 **总 / suc / fail**，均与同 cohort baseline 作差；13 official 的 k=64 只有 844 条，未进入完整样本最佳点选择。

| 实验 | k | 离散 MAE | 连续 ordinal MAE | 准确率 0.125/0.875 | 准确率 0.2/0.8 |
| --- | --- | --- | --- | --- | --- |
| [10 GRM forward / after](../results/mydata_bench/experiments_v2/attention_10_grm_forward_after/exp_record.md) | 64 | 1.8381 → 0.7270 | 1.8259 → 0.8309 | 4.85% / 14.93% / 0.17% → 72.10% / 42.91% / 85.64% | 9.69% / 29.85% / 0.35% → 75.89% / 50.75% / 87.54% |
| [11 GRM forward / before+after](../results/mydata_bench/experiments_v2/attention_11_grm_forward_before_after/exp_record.md) | 64 | 1.8381 → 0.7589 | 1.8259 → 0.8302 | 4.85% / 14.93% / 0.17% → 69.03% / 33.58% / 85.47% | 9.69% / 29.85% / 0.35% → 72.58% / 41.79% / 86.85% |
| [12 official GRM incremental / after](../results/mydata_bench/experiments_v2/attention_12_grm_incremental_after_official/exp_record.md) | 8 | 2.0118 → 0.6501 | 1.9946 → 0.6763 | 12.06% / 33.58% / 2.08% → 73.05% / 46.64% / 85.29% | 20.80% / 56.72% / 4.15% → 78.84% / 60.45% / 87.37% |
| [13 official GRM incremental / before+after](../results/mydata_bench/experiments_v2/attention_13_grm_incremental_before_after_official/exp_record.md) | 8 | 2.0118 → 0.9066 | 1.9946 → 0.9448 | 12.06% / 33.58% / 2.08% → 49.41% / 30.97% / 57.96% | 20.80% / 56.72% / 4.15% → 62.53% / 44.40% / 70.93% |

- **GRM forward 的 after、before+after 两组，在 k=8/32/64 下都同时降低离散 MAE 与连续 ordinal MAE，并在两套阈值下提高总/suc/fail 准确率。** 这是一组内跨 k 的一致趋势，尚不代表跨随机种子的重复验证。
- forward after 的 k=64 最低 MAE 为 0.7270，但 k=8 的 suc 更好：严格阈值下为 63.06%，宽阈值下为 71.27%，高于 k=64 的 42.91%/50.75%。因此最优 k 取决于是否更重视 suc/fail 平衡，不能只按总体准确率选择。
- **官方 incremental after 的 k=8 是正式 GRM 配置中最低完整样本 MAE 点：0.6501，连续 ordinal MAE 0.6763。** 两套阈值下 suc/fail 都提升。k=32 的宽阈值 suc 为 56.72%，与 baseline 持平；k=64 则略升至 57.46%。不能写成“所有 k、所有阈值均严格提高 suc”。
- 官方 incremental before+after 的 k=8 虽降低 MAE，却降低两套阈值的 suc；**k=32 才在完整样本上同时改善两类**：离散 MAE 2.0118 → 0.9291，严格阈值总/suc/fail 为 52.36%/55.97%/50.69%，宽阈值为 65.48%/58.96%/68.51%。
- 13 official 的 k=64 已有 844 条成功结果，离散 MAE 0.7002，严格/宽阈值总体准确率为 73.70%/79.15%；缺 2 条 fail，暂列不完整探索点，不能直接当作 846 条完整队列的最佳结果。

### 3.3 全部 top-k 的 MAE 与双类改善情况

“同时改善 k”要求同 cohort 下 MAE 严格下降、suc 与 fail 准确率均严格提高；GRM 要求两套阈值都满足。不完整条件不计入。

| 实验 | baseline | k=8 | k=32 | k=64 | 同时改善 k |
| --- | --- | --- | --- | --- | --- |
| 06 | 2.1206 | 1.5213 | 1.3487 | 1.5071 | 无 |
| 07 | 2.1206 | 1.4574 | 1.2092 | 1.3168 | 无 |
| 08 | 1.4586 | 1.2849 | 1.1915 | 1.0319 | 无 |
| 09 | 1.4586 | 1.1927 | 1.3499 | 1.1608 | 无 |
| 14 | 0.8333 | 0.8617 | 0.8097 | 1.0686 | 无 |
| 15 | 0.8333 | 0.7920 | 0.8440 | 1.3570 | 8 |
| 10 | 1.8381 | 0.8641 | 0.8061 | 0.7270 | 8, 32, 64 |
| 11 | 1.8381 | 0.7730 | 1.0165 | 0.7589 | 8, 32, 64 |
| 12 official | 2.0118 | 0.6501 | 0.6903 | 0.7116 | 8, 64 |
| 13 official | 2.0118 | 0.9066 | 0.9291 | 0.7002（n=844） | 32 |

### 3.4 历史 local-hop 结果

为完整覆盖原目录，保留两个旧实验的最低 MAE 点。它们不代表官方 episode progress，也不与上述正式 incremental 合并。

| 实验 | k | 离散 MAE | 准确率 0.125/0.875（总/suc/fail） | 准确率 0.2/0.8（总/suc/fail） |
| --- | --- | --- | --- | --- |
| [12 GRM local-hop / after（历史）](../results/mydata_bench/experiments_v2/attention_12_grm_incremental_after/exp_record.md) | 32 | 1.3109 → 1.1797 | 46.57% / 2.99% / 66.78% → 69.74% / 10.82% / 97.06% | 57.21% / 2.99% / 82.35% → 70.09% / 10.82% / 97.58% |
| [13 GRM local-hop / before+after（历史）](../results/mydata_bench/experiments_v2/attention_13_grm_incremental_before_after/exp_record.md) | 32 | 1.3109 → 0.8392 | 46.57% / 2.99% / 66.78% → 77.42% / 46.27% / 91.87% | 57.21% / 2.99% / 82.35% → 78.25% / 46.27% / 93.08% |

## 4. 配对区分度、控制组与预测分布

### 4.1 同视频不同 instruction 的配对结果

完整 cohort 有 578 条 fail，其中 35 条对应的 suc 不在 cohort 内，故完整条件的有效配对为 **543 对**；不是 578 对，也不是 543 个独立视频。下面仍使用各组完整样本最低 MAE 点。离散模型“强正差”为 suc−fail=3 或 4；GRM 为进度差 ≥50%，两者定义不同，不能直接横向排名。

| 实验 | k | 有效对 | 负差比例 | 强正差比例 |
| --- | --- | --- | --- | --- |
| 06 | 32 | 543 | 4.60% → 2.76% | 11.23% → 39.78% |
| 07 | 32 | 543 | 4.60% → 1.10% | 11.23% → 47.88% |
| 08 | 64 | 543 | 2.58% → 3.31% | 58.38% → 55.99% |
| 09 | 64 | 543 | 2.58% → 3.68% | 58.38% → 46.78% |
| 14 | 32 | 543 | 9.76% → 8.29% | 69.24% → 66.30% |
| 15 | 8 | 543 | 9.76% → 4.60% | 69.24% → 67.03% |
| 10 | 64 | 543 | 11.97% → 10.87% | 5.71% → 56.54% |
| 11 | 64 | 543 | 11.97% → 8.84% | 5.71% → 60.22% |
| 12 official | 8 | 543 | 18.97% → 8.47% | 11.23% → 65.93% |
| 13 official | 8 | 543 | 18.97% → 12.52% | 11.23% → 58.56% |

GRM 的较大正差增加尤其明显，官方 incremental after k=8 的负差也显著减少（描述性幅度）。但平均误差下降不保证每一项排序指标改善：Qwen 最后帧 k=64 的负差比例由 2.58% 增至 3.31%，强正差比例略降；RoboReward 实验 15 k=8 虽减少负差，强正差比例也略降。需要同时看负差、平局、正差幅度与端点分类，不能用一个指标替代全部判断。

### 4.2 Wrong-region 与 low-rank-head 控制组

| 实验 / k | n（各条件） | baseline MAE | target MAE | wrong MAE | low-rank MAE |
| --- | --- | --- | --- | --- | --- |
| 15 / 8 | 846 | 0.8333 | 0.7920 | 0.8830 | 0.8452 |
| 08 / 64 | 846 | 1.4586 | 1.0319 | 1.3605 | 1.5272 |
| 10 / 64 | 846 | 1.8381 | 0.7270 | 2.1868 | 1.8712 |
| 11 / 64 | 846 | 1.8381 | 0.7589 | 2.3404 | 1.8913 |

| 12 official / k=8，共同 838 条 | baseline | target | wrong | low-rank |
| --- | --- | --- | --- | --- |
| 离散 MAE | 2.0203 | 0.6480 | 1.4905 | 1.9177 |
| 准确率 0.125/0.875 | 11.93% | 73.51% | 36.28% | 18.97% |
| 准确率 0.2/0.8 | 20.53% | 79.36% | 45.35% | 25.42% |

这些代表点的 target MAE 均优于两种控制，支持区域/head 选择具有作用。官方 incremental 控制表已重新限制到四条件共同成功的 838 条，避免把缺少 8 条 suc 的 wrong 组与完整 target 直接比较。

但控制组并非总是不变：12 official 的 wrong 本身也明显优于 baseline；Qwen 全帧 k=32 的 wrong 总准确率 32.62%，甚至高于 target 的 31.09%，同时 MAE 更差、suc 降到 17.16%。RoboReward 实验 14 k=64 的 low-rank MAE 0.7612，也优于 target 的 1.0686。因此这些控制支持部分配置的选择性，尚不能证明所有收益都来自正确目标证据增强。

### 4.3 预测分布与 task 差异

- RoboReward text → video / 全帧 k=32：fail 输出 5 的数量从 290/578 减到 45/578，但 **398/578（68.86%）输出 2**，真正输出 1 的只有 46/578。由此解释“MAE 大幅下降而 fail 精确准确率仅 7.96%”。
- Qwen 最后帧 k=64：fail 输出 1 从 7 增至 159，输出 5 从 171 减至 49；与此同时 suc 输出 5 从 240/268 减至 196/268，说明两类之间存在权衡。
- GRM forward after k=64：按五个等宽进度区间统计，fail 落入 `[0,20%)` 从 2/578 增至 501/578，suc 落入 `[80%,100%]` 从 80/268 增至 136/268；但也有 70 条 suc 落入最低区间。分布明显分离，但没有解决所有成功样本。
- 官方 incremental after k=8：suc 落入最高区间从 152/268 增至 162/268，fail 落入最低区间从 24/578 增至 505/578。此处 `[0,20%)` 不包含恰为 0.2 的值，和 `progress<=0.2` 的准确率计数可能不同。

下表统计各最低 MAE 点的 **task 总准确率** 相对 baseline 提高/持平/下降的 task 数，共 28 个 task。GRM 采用 `0.2/0.8` 阈值；这不代表各 task 的 suc/fail 都同时提升，也不是显著性检验。

| 实验 | k | 提高 | 持平 | 下降 |
| --- | --- | --- | --- | --- |
| 06 | 32 | 10 | 5 | 13 |
| 07 | 32 | 10 | 6 | 12 |
| 08 | 64 | 18 | 4 | 6 |
| 09 | 64 | 15 | 3 | 10 |
| 14 | 32 | 6 | 6 | 16 |
| 15 | 8 | 14 | 7 | 7 |
| 10 | 64 | 28 | 0 | 0 |
| 11 | 64 | 28 | 0 | 0 |
| 12 official | 8 | 27 | 0 | 1 |
| 13 official | 8 | 26 | 1 | 1 |

两组 GRM forward 在全部 28 个 task 的总体准确率上均提高，12 official 在 27 个提高；RoboReward 实验 15 的小幅总体收益则伴随 7 个 task 下降。完整 task 准确率、预测分布和逐档 pairwise 计数见上表链接的各实验 `exp_record.md`。

## 5. Ranking head 结果

各模型均为 36 层 × 32 heads，排除前 8 层后比较 896 个位置。head 编号按原文件从 0 开始；跨模型位置重合不等价于功能完全相同。下面从各实验当前 ranking JSON 读取，**包含既有 ranking_overlap.md 未覆盖的两个 `_official` 版本**。

<details>
<summary>全部 attention 实验 top-8</summary>

| 实验 | top-8 heads |
| --- | --- |
| 06 | L22H15, L19H23, L19H17, L21H25, L22H5, L20H15, L21H16, L19H10 |
| 07 | L22H15, L19H23, L19H17, L21H25, L22H5, L21H16, L20H15, L19H28 |
| 08 | L21H16, L19H23, L21H25, L20H15, L21H26, L22H15, L21H29, L21H18 |
| 09 | L21H16, L19H23, L21H25, L20H15, L22H15, L21H29, L21H26, L19H31 |
| 10 | L19H23, L22H15, L20H4, L19H0, L19H10, L19H16, L19H15, L18H30 |
| 11 | L19H23, L19H10, L18H30, L19H16, L19H0, L22H15, L20H4, L19H21 |
| 12 local-hop | L20H4, L19H16, L19H0, L19H10, L19H23, L19H2, L19H21, L21H26 |
| 13 local-hop | L20H4, L19H23, L19H16, L19H0, L19H10, L19H21, L19H15, L18H30 |
| 12 official | L19H16, L19H10, L20H4, L19H0, L19H23, L19H31, L19H28, L20H2 |
| 13 official | L19H16, L19H10, L19H23, L19H0, L20H4, L18H30, L19H21, L20H2 |
| 14 | L20H15, L21H27, L19H3, L20H13, L19H0, L21H25, L19H28, L22H15 |
| 15 | L20H15, L21H27, L19H16, L19H3, L12H18, L19H28, L20H13, L18H30 |

</details>

代表性重合度如下，数值为 `|top-k 交集|/k`，不是 Jaccard。GRM 与原生视频模型的输入协议不同，跨模型列只是位置描述。

| 实验 A | 实验 B | top-8 | top-32 | top-64 |
| --- | --- | --- | --- | --- |
| 06 | 08 | 62.50% | 68.75% | 73.44% |
| 07 | 09 | 62.50% | 71.88% | 70.31% |
| 06 | 10 | 37.50% | 37.50% | 54.69% |
| 08 | 10 | 25.00% | 40.62% | 59.38% |
| 06 | 07 | 87.50% | 90.62% | 84.38% |
| 08 | 09 | 87.50% | 90.62% | 82.81% |
| 06 | 14 | 37.50% | 50.00% | 57.81% |
| 14 | 15 | 62.50% | 71.88% | 68.75% |
| 10 | 11 | 87.50% | 87.50% | 90.62% |
| 10 | 12 official | 62.50% | 68.75% | 71.88% |
| 12 official | 13 official | 75.00% | 90.62% | 89.06% |
| 12 local-hop | 12 official | 62.50% | 87.50% | 84.38% |
| 13 local-hop | 13 official | 87.50% | 81.25% | 85.94% |

相同模型与输入顺序下，最后帧/全帧的 top-8 通常较稳定：RoboReward text → video、Qwen 和 GRM forward 均为 87.50%。但 RoboReward 仅改变图文顺序，最后帧 top-8 重合就降到 37.50%。这说明 ranking 同时依赖模型和输入协议；相近 head 集合也不能保证相同 steering 收益。GRM 官方累计版已重新 ranking，不应直接套用旧 local-hop 的重合度结论。

## 6. 完整性与有效性边界

- **完整性**：7 个 baseline 均有 1213 条成功结果；6 个原生视频 attention 均完整覆盖 846 条 × 10 条件；GRM forward 和历史 local-hop 各有 846 条 × 13 个记录条件，其中包含三个 top-8 别名。
- **官方 incremental 缺失**：12/13 official 各有 8 条 suc 在至少一个 hop 无法构造等大小、不重叠的 wrong region，因此每个 wrong 条件缺 8 条；计入别名时各有 32 条 `missing_control` 记录。13 official 另有 2 条样本级 `invalid`，原因为 GRM score 文本严格解析失败，其 k=64 target/low-rank 仅 844 条、wrong 为 836 条。样本级错误记录没有 condition，不能误当成 baseline 失败；baseline 和 k=8/32 target 仍完整。
- **Ranking 非独立验证**：36 条 ranking 清单中 34 条 grounding 可用，34 条与评测 cohort 重叠，且发现集以 suc 为主。因此“跨 k 一致”不等于独立 hold-out 泛化。
- **样本与配对依赖**：fail 占 cohort 的 68.32%，总体准确率易被 fail 主导；同一视频可对应多个 instruction，543 个 pair 不是独立采样单位。未在本总结中把普通逐样本方差当作独立视频不确定性。
- **选择与统计证据**：最低 MAE 点为观察结果后的描述性选择；本总结没有新增多重比较校正、独立随机种子或 held-out 检验。原 GRM `attention_metrics.json` 的进度效应、CI、gate 与本总结的“两类端点准确率同时改善”不是同一估计目标，不能直接互换。
- **协议可比性**：forward、官方 incremental、历史 local-hop 不可混合统计；原生视频与 GRM 的视角、采样、prompt、离散/连续输出定义也不同。完整 baseline 与 attention 内置 baseline 仅在各自协议下解释，不能用其绝对指标作单因素因果比较。
- **核对方式**：按 `(example_id, condition)` 取 append-only 最后一条，随后仅统计 `status=ok`；样本级无 condition 错误独立保留。重新关联 metadata 计算标签、task 和 source suc，复算所有总览与配对指标，并与已有 native metrics、GRM baseline metrics 交叉一致。GRM attention 的描述性 MAE/阈值准确率直接由原始 progress 复算；不把 intervention mass 增加视为判别提升。

## 7. 结论

本轮最扎实的发现是：**attention steering 在 GRM forward 上跨 k 的收益一致，官方 incremental after 也有很强收益；在 RoboReward/Qwen 原生视频上则强烈依赖输入顺序和 k，经常以牺牲 suc 为代价换取 fail 或 MAE 改善。**

优先保留和复现的配置为：

1. **GRM forward / after**：k=64 适合最低 MAE 与较高总体准确率，k=8 保留更好的 suc/fail 平衡；before+after 的三档 k 也均优于自身 baseline，但增加 before 干预并不普遍更优。
2. **GRM 官方 incremental / after / k=8**：846 条完整队列离散 MAE 0.6501、宽阈值总准确率 78.84%，两类均提高；共同 838 条控制比较也支持 target 优于 wrong 和 low-rank。
3. **RoboReward / video → text / 全帧 / k=8**：当前原生视频实验中唯一同时改善 MAE、suc 与 fail 的配置，但收益较小且 k=32/64 不保持，需独立验证。

Qwen 原生视频最后帧 k=64 值得作为 MAE 改善的参考，但尚未达到 suc/fail 同时提高；官方 incremental before+after k=32 是完整样本的双类改善点，k=64 需先补全失败记录再复核。旧 local-hop 结果不能作为官方 incremental 有效性的依据。

所有逐实验统计见 [experiments_v2](../results/mydata_bench/experiments_v2/) 中的 `exp_record.md`。既有非官方版完整 head 重合矩阵见 [ranking_overlap.md](../results/mydata_bench/experiments_v2/ranking_overlap.md)，官方版代表性重合度已在本文补充。
