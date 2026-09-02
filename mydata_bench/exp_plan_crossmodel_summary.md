# 跨模型 image-sequence 与 attention steering 实验总结

## 1. 实验范围与口径

本轮实验对应 [`exp_plan_crossmodel.md`](./exp_plan_crossmodel.md)，考察 RoboReward-8B 与
Qwen3-VL-8B 在以下变化下的表现：

- 将原生 video 输入改为均匀采样的 8 张独立图像；
- 比较 `text → images`、`images → text` 和 GRM 式交错输入；
- 对最后一帧或全部 8 帧进行 target / non-target attention steering；
- 使用按当前模型和输入协议重新得到的 ranking，比较 top-k = 8、32、64，并设置
  wrong-region 和 low-rank-head 控制组。

Baseline 在完整数据集上评估，共 1213 条（407 suc / 806 fail）。Attention 实验使用自动
grounding 成功的 846 条 cohort（268 suc / 578 fail）。因此 steering 的变化必须与各自
`exp_record.md` 中同 cohort 的 baseline 比较，不能直接与 1213 条的 baseline 数字作差。

## 2. Baseline：输入顺序具有显著影响

| 模型 | 输入顺序 | MAE | 总准确率 | suc 准确率 | fail 准确率 |
| --- | --- | ---: | ---: | ---: | ---: |
| RoboReward-8B | text → images | 1.5565 | 14.34% | 34.15% | 4.34% |
| RoboReward-8B | images → text | **1.0965** | **53.83%** | **43.98%** | **58.81%** |
| Qwen3-VL-8B | text → images | 1.8145 | **26.05%** | **75.43%** | 1.12% |
| Qwen3-VL-8B | images → text | **1.6125** | 17.89% | 40.79% | **6.33%** |

RoboReward 明显偏好 `images → text`：MAE 降低 0.4600，总准确率提高 39.49 个百分点，
且 suc/fail 更平衡。Qwen 的结论不一致：`images → text` 的 MAE 更低，但 `text → images`
因更倾向输出高分而具有更高的 suc 和总体精确准确率，fail 准确率仍接近于零。说明输入
顺序和 causal mask 确实会强烈改变输出分布，但不存在跨模型统一的最优顺序。

## 3. Attention steering 结果

下表为每个实验中 **样本完整（n=846）且 candidate-target 的最低 MAE 点**。箭头左侧是
该协议的同 cohort baseline，右侧是 steering 后结果。

| 实验 | 模型与输入 / 干预范围 | k | MAE | 总准确率 | suc 准确率 | fail 准确率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 03 | RoboReward，text → images / 最后一帧 | 32 | 1.4965 → 1.3452 | 17.49% → 20.45% | 48.13% → 23.51% | 3.29% → 19.03% |
| 04 | RoboReward，images → text / 最后一帧 | 32 | 0.8203 → 0.6974 | 64.78% → 68.09% | 59.33% → 48.51% | 67.30% → 77.16% |
| 07 | Qwen，text → images / 最后一帧 | 64 | 1.5934 → 1.2506 | 28.25% → 46.69% | 84.33% → 75.37% | 2.25% → 33.39% |
| **08** | **Qwen，images → text / 最后一帧** | **64** | **1.4326 → 1.0449** | **22.81% → 45.63%** | **55.97% → 59.70%** | **7.44% → 39.10%** |
| 09 | RoboReward，text → images / 全帧 | 32 | 1.4965 → 0.9716 | 17.49% → 52.25% | 48.13% → 45.52% | 3.29% → 55.36% |
| 10 | RoboReward，images → text / 全帧 | 32 | 0.8203 → 0.7423 | 64.78% → 64.54% | 59.33% → 47.01% | 67.30% → 72.66% |
| 11 | Qwen，text → images / 全帧 | 8 | 1.5934 → 1.3463 | 28.25% → 24.94% | 84.33% → 66.42% | 2.25% → 5.71% |
| 12 | Qwen，images → text / 全帧 | 8 | 1.4326 → 1.4279 | 22.81% → 13.71% | 55.97% → 38.81% | 7.44% → 2.08% |
| **13** | **RoboReward，交错输入 / 全帧** | **32** | **1.5674 → 0.9468** | **23.52% → 56.15%** | **48.88% → 51.49%** | **11.76% → 58.30%** |
| 14 | RoboReward，交错输入 / 最后一帧 | 64 | 1.5674 → 1.5000 | 23.52% → 33.10% | 48.88% → 33.58% | 11.76% → 32.87% |
| 15 | Qwen，交错输入 / 全帧 | 8 | 1.7057 → 1.3322 | 30.97% → 26.60% | 94.78% → 75.75% | 1.38% → 3.81% |
| 16 | Qwen，交错输入 / 最后一帧 | 8 | 1.7057 → 1.6312 | 30.97% → 30.61% | 94.78% → 90.67% | 1.38% → 2.77% |

### 主要发现

1. **只有实验 08 的 k=64 和实验 13 的 k=32 同时改善 MAE、总体准确率、suc 准确率与
   fail 准确率。** 其中实验 13 最突出：MAE 降低 0.6206，总准确率提高 32.63 个百分点，
   并且不是通过牺牲某一 split 获益。
2. 多数配置虽然降低 MAE、提高 fail 准确率，却同时降低 suc 准确率。这表明固定强度的
   steering 往往首先造成整体分数下移；由于 cohort 中 fail 占 68.3%，只看 MAE 或总体
   准确率会高估方法的稳定性。
3. k 的效果明显非单调。实验 13 在 k=32 达到最优，但 k=64 的总体准确率几乎回到
   baseline；实验 14 则只有 k=64 有中等改善。因此不存在可跨输入协议复用的统一 k。
4. 全帧干预不是普遍优于最后一帧：它在 RoboReward 的 `text → images` 和交错输入上更强，
   但在 RoboReward 的 `images → text` 上反而不如最后一帧；Qwen 的若干全帧条件还存在
   不完整输出，不能正式排序。
5. GRM 式交错输入与 causal-mask 假设只得到**部分支持**。RoboReward 的交错全帧 k=32
   产生本轮最清晰的结果，但同一协议的 k=8/k=64、最后一帧版本以及 Qwen 均未稳定复现。
   因此 causal masking / temporal span 不是造成跨模型差异的唯一原因。

### Pairwise 与控制组证据

- 实验 13 k=32 中，同视频 suc−fail 的负差比例由 5.16% 降至 1.66%，强正差（3 或 4）
  由 18.42% 升至 46.04%。同 k 的 wrong-region（MAE 1.7246、准确率 8.16%）和
  low-rank-head（MAE 1.6525、准确率 19.15%）均明显变差，支持该结果具有区域和 head
  选择性。
- 实验 08 k=64 中，负差比例由 7.18% 降至 4.42%，强正差由 49.54% 升至 54.88%；同 k
  的 wrong-region 和 low-rank-head 均不如 target 条件。
- 但这种特异性并不普遍：部分 Qwen 全帧实验中 wrong-region 也有明显改善，说明强 bias
  可能同时改变全局输出校准，而不只是增强目标区域证据。

综上，当前方法证明了 **attention head 与空间区域选择可以有效影响模型判断**，但尚未达到
“在两个模型、多个输入协议和 k=8/32/64 下稳定同时改善 MAE 与 suc/fail 准确率”的目标。
它目前更像强烈依赖模型、输入顺序、干预时域和剂量的探索性方法，而不是可直接跨模型迁移
的稳定方案。

## 4. Ranking head 结果

高频出现的 head 包括 `L20H15`、`L21H25`、`L21H16`、`L19H28` 和 `L23H13`，但 top-8
仍对输入顺序和干预范围敏感。交错协议的 ranking 最稳定：同模型全帧/最后一帧的 top-8
重合率为 62.5%（RoboReward）和 75.0%（Qwen），top-64 重合率为 89.06% 和 82.81%；
交错协议跨模型 top-32/64 重合率也达到 75.0%–87.5% / 76.56%–85.94%。相比之下，普通
独立图像输入中 RoboReward 两种顺序的 top-8 仅重合 12.5%。这说明较大的候选 head 集合
具有一定跨协议共性，但最头部的少量 head 不够稳定。

<details>
<summary>各实验 top-8</summary>

| 实验 | top-8 heads |
| --- | --- |
| 03 RoboReward text → images / last | L25H5, L22H15, L22H5, L31H19, L21H16, L22H26, L24H10, L21H25 |
| 04 RoboReward images → text / last | L21H16, L20H15, L19H0, L20H4, L19H28, L22H2, L21H29, L21H27 |
| 07 Qwen text → images / last | L21H25, L21H26, L21H16, L25H5, L21H19, L21H29, L20H15, L21H3 |
| 08 Qwen images → text / last | L24H13, L21H25, L20H15, L23H13, L21H16, L21H18, L21H29, L24H11 |
| 09 RoboReward text → images / all | L22H15, L19H28, L19H31, L21H16, L22H5, L19H17, L20H15, L19H10 |
| 10 RoboReward images → text / all | L19H28, L21H16, L19H16, L20H15, L22H15, L19H0, L18H30, L19H31 |
| 11 Qwen text → images / all | L21H25, L21H16, L19H31, L19H23, L22H15, L20H15, L19H28, L21H29 |
| 12 Qwen images → text / all | L24H13, L21H16, L12H18, L20H15, L21H25, L23H13, L27H16, L19H28 |
| 13 RoboReward interleaved / all | L19H28, L20H15, L19H31, L20H13, L26H31, L23H13, L19H23, L24H13 |
| 14 RoboReward interleaved / last | L24H13, L20H15, L20H4, L23H13, L19H28, L21H25, L21H26, L20H13 |
| 15 Qwen interleaved / all | L20H15, L19H23, L21H25, L19H28, L21H31, L21H27, L21H18, L23H13 |
| 16 Qwen interleaved / last | L21H25, L20H15, L21H26, L21H18, L23H13, L21H31, L21H27, L23H30 |

</details>

## 5. 有效性边界

- RoboReward 的 6 组 attention 实验以及 Qwen 实验 07/08 均完成全部 846 条样本和 10 个
  条件。Qwen 实验 11/12/15/16 分别有 11/37/72/80 个严格解析失败，导致后续条件级联
  缺失：实验 11/12 到 k=32 仍完整，k=64 不完整；实验 15/16 只有 baseline 与
  candidate-target-k8 完整。缺失条件的数字只能作描述性参考，不能作为正式比较。
- Ranking 原有 36 条发现样本，grounding 后有效 34 条；这 34 条与评测 cohort 重叠，并非
  独立 hold-out，可能使 head 选择效果偏乐观。
- Attention cohort 是自动 grounding 筛选集，且 suc/fail 比例不平衡；结论不能直接外推到
  完整数据集或其他任务。
- 每个配置同时探索了 3 个 k 和多个控制条件，未做多重比较校正。实验 08 k=64 和实验 13
  k=32 应在独立 ranking / hold-out cohort 上复现后再升级为确认性结论。
- 本轮没有在同一张汇总表内提供严格匹配的 native-video 对照，因此可以确认独立 image
  span 的实现与干预有效，但不能仅凭本轮结果定量断言 video → image 本身优于原生 video。

## 6. 结论

最值得保留并复现的两个配置是：

1. **RoboReward-8B：交错 8-image 输入、全帧 target/non-target steering、top-32**；
2. **Qwen3-VL-8B：images → text、最后一帧 steering、top-64**。

实验 13 表明，交错输入与全时域目标增强可以在 RoboReward 上显著改善指令一致性与
suc/fail 平衡；实验 08 说明 Qwen 也能获得同方向收益。但整体上，单纯把 video 拆成 image、
改变图文顺序或扩大干预到全部帧，都不能稳定解决跨模型迁移问题。下一步应优先在独立
hold-out ranking 上复现上述两点，并进行按模型/协议校准的 bias 与 k 剂量搜索，同时把
“suc 与 fail 均提升”作为必要验收条件，而不是只优化 MAE 或总体准确率。

详细结果见各实验目录中的 `exp_record.md`，ranking 重合度见
[`ranking_overlap.md`](../results/mydata_bench/experiments_v2_corssmodel/ranking_overlap.md)、
[`ranking_overlap_all_frames.md`](../results/mydata_bench/experiments_v2_corssmodel/ranking_overlap_all_frames.md)
和 [`ranking_overlap_interleaved.md`](../results/mydata_bench/experiments_v2_corssmodel/ranking_overlap_interleaved.md)。
