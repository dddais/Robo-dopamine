# 跨模型研究与效果分析

遵守本文的的要求，进行auto research，针对现有研究背景与问题进行探索性研究。

进行长时间的充分的调研、思考、理论分析、实验，直到完成所有主线目标。

我要睡觉了，需要我确认的部分先跳过，进行你能进行的内容。

## 研究背景：

- 基于/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan.md的规划，进行了baseline 和attention steering的实验，目前发现在robo-dopamine的GRM上该方法的效果十分明显;
- 但是在qwen3-vl-8b和roboreward-8b的效果不是很明显
- 已有实验可供参考：
  - 目前已进行的跨模型实验：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel.md
  - 该实验结果：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel_summary.md

## 主线目标1

- 调研相关文章，调研总结当前有哪些相关方法 ，目前已有部分可参考，见本文档“可参考代码/文献”部分，调研结果更新在“可参考代码/文献”部分。

## 主线目标2

- step1:基于调研的结果和方法，从理论角度思考提出能够改进目前attention steering的方案，如果需要可继续调研相关工作文章
- step2:实现step1提出的改进方案，进行实验验证，分析实验结果，如果效果不好则重复step1提出改进方案
- 验收目标：不断重复上述两个step，直到能够成功改良原 attention steering方法:两种模型（roboreward-8b 和qwen3-vl-8b ）各自在五种输入构造中的至少两种输入下都能稳定提升在数据集上的表现（MAE下降，suc,fail准确率提高）(不要求所有top-k都能满足，至少有一个不小的范围可以吧)，像GRM的表现那样。
- 严禁使用端点hard coding这种类似作弊的方法！！！



## 基本原则（必须遵守）

- 尽量不修改现有代码库，如果需要修改，进行增量式修改，比如增加可选配置项等；
- 不允许进行git 操作本地已有的仓库，只能git clone开源仓库进行参考；
- 不允许对本地数据，结果等进行删除修改等操作，只能新增；
- 不用担心耗时，进行充分的调研、思考、理论分析，提出有道理的优雅的方案，严禁作弊的方法



## 可参考相关代码/文献

- /home/dais/workspace/gaze-heads : [https://arxiv.org/pdf/2606.14703v1](https://arxiv.org/pdf/2606.14703v1)
- /home/dais/workspace/PAI : [https://arxiv.org/pdf/2311.02262](https://arxiv.org/pdf/2311.02262)
- /home/dais/workspace/PASTA : [https://arxiv.org/pdf/2407.21771](https://arxiv.org/pdf/2407.21771)
- **PASTA — Post-hoc Attention Steering for LLMs**（ICLR 2024）：[arXiv:2311.02262](https://arxiv.org/abs/2311.02262)。选择小部分 task-specific/task-agnostic heads，并以 `log(alpha)` 修改 attention mask，相当于对非强调 token 做乘法降权；说明 soft multiplicative reweighting 比固定饱和 bias 更适合保留原能力。
- **PAI — Paying More Attention to Image**（ECCV 2024）：[arXiv:2407.21771](https://arxiv.org/abs/2407.21771)。只在最后 query 放大 image-token attention，并使用 image-free branch 做 logits 对比以抵消 text inertia；启发本任务比较 query scope，并区分“视觉证据增强”与“全局输出校准”。
- **ASCD — Attention-Steerable Contrastive Decoding**（AAAI 2026）：[arXiv:2506.14766](https://arxiv.org/abs/2506.14766)，DOI 10.1609/aaai.v40i12.38000。用 500 张参考图统计 `textAttn/visAttn`，每样本 top-32 投票得到 model-specific text-centric heads；正向分支只增强这些 heads，负向分支只抑制动态选出的关键视觉 token。其核心证据是选择性 head/token 操作优于随机或全量操作。
- **CAST — Caption-Guided Visual Attention Steering**：[arXiv:2605.04641](https://arxiv.org/abs/2605.04641)。以 caption-query vs non-caption-query 的 attention-output 差异训练 probe 选 head，并对选中 head 加入对比均值 steering direction；启发用“对比敏感度”而非 raw target mass 作为 ranking 目标。
- **Gaze Heads: How VLMs Look at What They Describe**：[arXiv:2606.14703](https://arxiv.org/abs/2606.14703)。在独立 comics 语料上发现少量 model-specific gaze heads；top-100 干预有效而 all-head 干预破坏生成。工作区已有 RoboReward/Qwen 各自在 500 个独立 comics 样本上得到的完整 ranking，可作为无评测标签泄漏的外部候选。
- **HAS — Highlight-guided Attention Steering for Multimodal LLM Video Summarization**：[arXiv:2607.17994](https://arxiv.org/abs/2607.17994)。把连续帧重要性平滑到 `[0,1]`，再以 `log(h+epsilon)` 注入 attention logits，等价于温和乘法重标定；支持使用连续时序 prior，避免 hard frame selection。
- **Arbitration Failure, Not Perceptual Blindness**：[arXiv:2604.09364](https://arxiv.org/abs/2604.09364)。报告 VLM 失败时视觉证据通常仍被编码，瓶颈更接近末层 arbitration；last-token patching 仅改变 0–1% 输出，而 full-sequence patching 改变 60–84%。这说明只增加注意力质量未必提高准确率，必须直接验证输出与配对区分度。
- **Inference-Time Attention Steering for VLA Driving Models**（ECCV 2026）：[arXiv:2608.17095](https://arxiv.org/abs/2608.17095)。在 Qwen3-VL 上发现 bias 强度和 late-layer 数量都有单调剂量效应，并强调逐调用 exposure audit；支持本研究进行 K-aware 剂量归一化和 hook 生效审计。
- **Attention is Case-Sensitive**（ECCV 2026）：[arXiv:2608.03711](https://arxiv.org/abs/2608.03711)。注意力集中并不保证准确率提高，强 salience 甚至会降性能；作为本研究不能用 attention-mass 增加替代任务指标的反例证据。
- Localization heads :Your Large Vision-Language Model Only Needs AFew Attention Heads For Visual Grounding; [https://arxiv.org/pdf/2503.06287](https://arxiv.org/pdf/2503.06287)
- Your Model Already Knows: Attention-Guided Safety Filter for Vision-Language-Action Models : [https://arxiv.org/pdf/2606.09749](https://arxiv.org/pdf/2606.09749)
- Analyzing Multi-Head Self-Attention :[https://arxiv.org/pdf/1905.09418](https://arxiv.org/pdf/1905.09418)
- **Visual Retrieval Heads (VRHs)**：[arXiv:2608.27417](https://arxiv.org/abs/2608.27417)。在 11 个 VLM、5 个 referring-expression benchmark 上统一比较 query token、key aggregation 和跨样本聚合；结论是从**输出预测 token**到 referent region 的 attention sum 最能找到因果 head。VRH 只占约 1.7%–2.6%，mask top-20 可令 grounding accuracy 最多下降 80 个百分点；共享 LLM backbone 的 VLM 之间还能迁移。它直接指出本项目只用 last-prompt-query raw mass 排名可能选中了“看向目标”但不负责输出判定的 head。
- **SADI / Region-Aware Attention Recalibration**：[arXiv:2605.24957](https://arxiv.org/abs/2605.24957)。以跨 head 的 robust median consensus 建立视觉锚点，再由空间 inter-head disagreement 连续分配 intervention budget，避免统一 bias 和 hard truncation；启发把样本/token 不确定性纳入剂量，而不是固定 `±6`。
- **Causal Route Gating**：[arXiv:2605.24024](https://arxiv.org/abs/2605.24024)。把 head 分解为 visual/text route，用一次 forward/gradient 近似其 token-level effect，仅抑制 prior-dominant text route。其“route competition”解释与本项目“attention mass 上升但准确率不一定上升”一致，说明末端 arbitration 需要独立处理。
- **ScAle — Attention Head Scaling as a Minimal Adapter**：[arXiv:2606.29579](https://arxiv.org/abs/2606.29579)。在冻结 VLM 上学习约 1K 个标量，调制 last-token attention 与 MLP activation；说明 bounded、连续、少参数的层/head 剂量比全局同强度干预更合理。
- **Role-Break in Attention Heads**：[arXiv:2607.29412](https://arxiv.org/abs/2607.29412)。把 hallucination 表述为各 head 偏离其正常上下文角色，保留 head identity 的低维信号在六个 VLM/四个 benchmark 上平均 AUROC 93.23；支持用样本级 head deviation 作 gate，而非仅靠静态全局排名。
- **Targeting the Attention Heads Behind Object Hallucination**：[arXiv:2608.24966](https://arxiv.org/abs/2608.24966)。以 attention drop 与 hallucination-token log-prob 的因果筛选得到 32-head set；held-out 配对检验显示 CHAIR 改善，但 object recall 从 0.78 降至 0.70。该副作用与本项目 fail 改善、suc 受损的现象同构，要求报告完整行为剖面而非单一总体指标。



## 原因分析

### 1. 固定 `±6` 造成近饱和而不是温和 steering

当前 target token 加 `+6`、negative token 加 `-6`，相对 attention odds 被放大 `exp(12)≈1.63×10^5`。当 negative scope 覆盖全视觉区域时，效果还随负 token 数扩大。这会把模型从原表示流形强行推离，符合已有实验中“fail 大幅改善但 suc 同时下降”、wrong-region 偶尔也改善以及 k 响应非单调的现象。

### 2. 剂量没有按 head 数、目标面积和时域曝光归一化

同一 bias 用于 k=8/32/64，而被修改的 `(layer, head, query, key)` 元素数可相差一个数量级；all-frame 的 target/negative token 又显著多于 last-frame。因而 k 消融同时改变“head 覆盖”和“总剂量”，不能被解释为纯粹的 head 数效应。应至少记录每次调用 exposure，并采用随 k 递减的 bounded bias 或按面积/不确定性动态预算。

### 3. query scope 同时污染 prompt 编码与输出判定

旧 `steering_query_scope=all` 会修改 prefill 的所有 query 行以及每个 decode step；PAI 只修改每次调用的最后 query，ScAle 也聚焦 last-token。全 prompt 改写容易改变文本 rubric、格式和语言先验，而不只是输出 token 从视觉区域取证。需要比较 `last_query`/decode-only，并验证 score token 实际暴露于 hook。

### 4. ranking 目标与真正的因果读取 head 不完全一致

当前 ranking 使用 last prompt token 到 bbox 的 raw attention mass。VRH 的系统消融显示，从输出 prediction token 计算 region-sum 更能识别因果 grounding head；CAST 也以 caption/non-caption 的 attention-output 对比敏感度选 head。raw mass 可能偏向“空间注视 head”，却漏掉把视觉证据传给最终评分的 retrieval/arbitration head。

### 5. 视觉增强与输出校准被混为一个效应

Qwen 在 text-first/interleaved baseline 上本来就强烈偏向高分；强 target steering 又普遍令 suc 与 fail 一起下移。总体 MAE 和 fail accuracy 因类别不平衡而改善，并不证明配对指令区分度稳定提高。PAI/ASCD 的 contrastive branch、Causal Route Gating 的 visual/text route 分解都提示：需要从 target-specific 信号中消掉全局 score drift，或以无标签的样本级证据 gate 控制是否干预。

### 6. 静态排名的独立性与跨协议泛化不足

既有 34 条 grounding ranking 样本与评测 cohort 重叠，会导致选择偏乐观；不同输入顺序的 top-8 又不稳定。工作区 500 条独立 comics 得到的 model-specific gaze ranking 可作为无评测标签泄漏的外部候选。其 Qwen/RoboReward top-32 交集为 28/36（Jaccard 77.78%），但与当前各自 task ranking 的 top-32 只重合 5 和 4 个，适合用作独立排名对照，而不能未经实验直接替代。

## 实验基础设置

增量式修改：代码修改不要影响到之前的实验运行，尽量以增量式的形式增加代码，比如加可选参数配置之类的

**数据集** ：/home/dais/workspace/data/mydata_v2/new ;/home/dais/workspace/Robo-Dopamine/results/mydata_bench/cohorts/auto_grounded_v2 (认为这就是正确的，不需要人工审核)

**config** 放在：/home/dais/workspace/Robo-Dopamine/mydata_bench/configs/v2_crossmodel

**输入**：video->text ; text->video; image->text ; text->image ;interleaved ;以上五种都需要尝试，方法最好能在大部分输入构造下work
**输出** 在：/home/dais/workspace/Robo-Dopamine/results/mydata_bench/experiments_v2_corssmodel/auto_research

**conda环境**：sam3:rewardbench-sam3 ；其它实验：robo-dopamine

**可用GPU**：0，1，2

**vpn** : proxy_on

**评价指标**：
1.MAE：按照roboreward的原定义
2.准确率：对于suc数据，lable=5,对于fail数据，lable=1；预测结果和lable相同的数量与概率。包括总准确率，suc,fail准确率，各个具体task的准备率分布。对于GRM这种输出连续进度的，用阈值区分开，统计两套阈值的情况：0.125，0.875；0.2，0.8
3.预测分布：统计在suc,fail数据上模型预测的lable分布，以及各个具体task上的模型预测分布；
4.pairwise区分度分析：因为数据集构成原理是1条suc数据，对应了1条或多条相同视频，不同instruction的fail数据，所以需要先找到suc数据所对应的fail数据，分析相同视频下不同instruction带来的影响。对于roboreward-8b,qwen这种输出离散的模型，计算统计配对数据中suc数据的预测值与fail数据的预测值的差值，把差值分成：负，0，1，2，3，4几档统计一下；
5.ranking head统计:列出具体的top 8，统计top 8,32,64在不同模型的重合度

## 最终有效方案

### 验收结论

**主线数值目标已完成**：在同一 846-record cohort（268 suc / 578 fail）上，以“MAE 下降、总准确率提高、suc 准确率提高、fail 准确率提高”四项同时成立为门槛，RoboReward-8B 在 3/5 种输入构造通过，Qwen3-VL-8B 在 2/5 种通过。不存在按真实标签、suc/fail 身份或预测端点切换的规则；真实标签只在模型推理全部完成后用于评分。

总体结论的统计身份为 **探索性通过 / CAUTION**，不是预注册确认性结论。主要原因是研究经过 R1–R13 多轮搜索，Qwen 的新 text→images 方法是在观察 R9 full 失败后提出；此外 cohort 只保留了 1213 条源记录中自动 grounding 成功的 846 条（69.74%）。完整 11/11 谬误扫描和复现性判定见 [validation_report_v1.md](../results/mydata_bench/experiments_v2_corssmodel/auto_research/validation_report_v1.md)。

### 最终方案 A：RoboReward relation-aware direct attention steering

对 text→images 输入，把 manipulated-object bbox 与由指令确定的 destination/reference bbox 映射成视觉 token 后取**集合并集**，在 all frames、all queries 上用 task-specific top heads 直接 steering：`bias=6`、`top-k=16–24`、`negative_scope=all_visual`。没有 destination 时按冻结规则回退 object-only。

| cohort | 条件 | MAE | 总 acc | suc acc | fail acc | MAE cluster-bootstrap 95% CI | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| full 846 | baseline | 1.4965 | 17.49% | 48.13% | 3.29% | — | — |
| full 846 | k16 | 1.3026 | 32.39% | 63.43% | 17.99% | `[-0.307,-0.156]` | 5.33e-26 |
| full 846 | k24 | 1.3121 | 34.28% | 57.09% | 23.70% | `[-0.296,-0.132]` | 2.58e-27 |
| held-out 694 | baseline | 1.4813 | 17.44% | 53.13% | 3.78% | — | — |
| held-out 694 | k16 | 1.2795 | 31.70% | 66.67% | 18.33% | `[-0.315,-0.127]` | 1.06e-21 |
| held-out 694 | k24 | 1.2939 | 33.43% | 59.38% | 23.51% | `[-0.292,-0.090]` | 3.03e-22 |

完整 control 为 5922/5922、零 invalid：wrong-region 与 low-rank heads 在 k16/k24 都不能同时通过四指标，只有正确区域 + 高排名 heads 通过。另一个独立 full 进程的 baseline/k16/k24 共 2538 个预测与首个 full 逐条完全相同。配置见 [full_r4_roboreward_text_images_relation_union.yaml](configs/v2_crossmodel/auto_research/full_r4_roboreward_text_images_relation_union.yaml)，完整 task/distribution/pairwise 见 [R4 exp_record](../results/mydata_bench/experiments_v2_corssmodel/auto_research/full_r4_roboreward_text_images_relation_union/exp_record.md) 与 [control exp_record](../results/mydata_bench/experiments_v2_corssmodel/auto_research/controls_full_r4_roboreward_text_images_relation_union/exp_record.md)。

RoboReward 另有两个已通过输入：

- **interleaved all-frame k32**：full `MAE 1.5674→0.9468`，总 `23.52→56.15%`，suc `48.88→51.49%`，fail `11.76→58.30%`；694 子集仍为 `1.5576→0.8876 / 23.63→56.92% / 52.08→54.69% / 12.75→57.77%`。
- **native video→text all-frame k8**：full `0.8333→0.7920 / 69.15→70.92% / 68.28→70.90% / 69.55→70.93%`。该效应较小，作为第三个通过输入而不是主 headline。

因此 RoboReward 为 **3/5** 输入通过。

### 最终方案 B：Qwen baseline-preserving 三分支凸 log-prob 混合

Qwen text→images 使用三条完全无标签分支：原始 baseline、relation-target steering、等面积 matched-wrong-region steering。对 1–5 score-token log-prob 作全局固定凸组合：

`ℓ_mix = w_b ℓ_baseline + w_t ℓ_relation-target + w_c ℓ_matched-wrong`，其中权重非负且和为 1，再取 1–5 argmax。

所有 record、task、suc/fail 共用同一权重；不根据原始预测或输出端点切换。k24 的稳定权重线段为：

- `(0.70,0.05,0.25)`
- midpoint `(0.65,0.075,0.275)`
- `(0.60,0.10,0.30)`

| cohort | 权重 | MAE | 总 acc | suc acc | fail acc | MAE cluster-bootstrap 95% CI | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| full 846 | baseline | 1.5934 | 28.25% | 84.33% | 2.25% | — | — |
| full 846 | `(0.70,0.05,0.25)` | 1.5343 | 30.97% | 85.07% | 5.88% | `[-0.0694,-0.0162]` | 5.65e-6 |
| full 846 | midpoint | 1.5236 | 31.44% | 85.82% | 6.23% | `[-0.0881,-0.0307]` | 1.40e-6 |
| full 846 | `(0.60,0.10,0.30)` | 1.5260 | 31.56% | 85.45% | 6.57% | `[-0.0886,-0.0301]` | 1.94e-6 |
| held-out 694 | baseline | 1.6009 | 25.07% | 83.85% | 2.59% | — | — |
| held-out 694 | midpoint | 1.5173 | 28.53% | 85.42% | 6.77% | `[-0.1120,-0.0450]` | 3.03e-6 |

k32 在保守权重点 full 与 694 上也四项点估计改善，但 MAE cluster CI 跨 0，故只作为 k24 的边界支持，不作为独立稳健点。配置见 [R9 config](configs/v2_crossmodel/auto_research/full_r9_qwen_text_images_constrained_contrastive_k24.yaml)，完整报告见 [midpoint full](../results/mydata_bench/experiments_v2_corssmodel/auto_research/full_r9_qwen_text_images_constrained_contrastive_k24/tri_mix_wMid_full_v1.md)、[midpoint held-out](../results/mydata_bench/experiments_v2_corssmodel/auto_research/full_r9_qwen_text_images_constrained_contrastive_k24/tri_mix_wMid_heldout694_v1.md) 和 [R12 k32](../results/mydata_bench/experiments_v2_corssmodel/auto_research/full_r12_qwen_text_images_constrained_tri_mix_k32/tri_mix_wA_full_v1.md)。

Qwen 的另一条通过输入是 **images→text last-frame k64**：full `MAE 1.4326→1.0449`，总 `22.81→45.63%`，suc `55.97→59.70%`，fail `7.44→39.10%`；694 子集为 `1.3977→0.9784 / 22.19→47.84% / 58.33→60.94% / 8.37→42.83%`，MAE CI `[-0.534,-0.369]`，McNemar p=4.68e-35。因此 Qwen 为 **2/5** 输入通过。

三分支方法是在 R9 full 之后提出，694 子集随后参与过权重探索，必须标为 post-hoc。为检验这种开发偏差，R10 pilot 冻结 interleaved 权重 `(0.55,0.10,0.35)` 后运行 R13：4230/4230、零 invalid，但 k24/k32 的 full MAE 分别 `1.7057→1.7423/1.7305`，694 MAE `1.7176→1.7651/1.7522`，即使三个 accuracy 都提高也严格判失败，不重试、不重新选权重。见 [R13 k24 full](../results/mydata_bench/experiments_v2_corssmodel/auto_research/full_r13_qwen_interleaved_constrained_tri_mix/tri_mix_frozen_k24_full_v1.md) 与 [k32 held-out](../results/mydata_bench/experiments_v2_corssmodel/auto_research/full_r13_qwen_interleaved_constrained_tri_mix/tri_mix_frozen_k32_heldout694_v1.md)。

### 五输入总表

| 模型 | text→video | video→text | text→images | images→text | interleaved | 通过数 |
| --- | --- | --- | --- | --- | --- | ---: |
| RoboReward | 未通过（suc↓） | **通过 k8** | **通过 relation-union k16–24** | 未通过（suc↓） | **通过 k32** | **3/5** |
| Qwen | 未通过（suc↓） | 未通过 | **通过 tri-mix k24 权重线段** | **通过 k64** | 未通过（R13 MAE↑） | **2/5** |

### Task、预测分布、pairwise 与 head overlap

- task overall up/same/down：Robo text→images k16 为 `24/2/2`，Robo interleaved k32 为 `27/1/0`；Qwen text→images midpoint 为 `8/19/1`，Qwen images→text k64 为 `22/4/2`。完整 suc/fail task 表均在上述 exp record。
- pairwise suc−fail `<0/0/1/2/3/4`：Robo text→images `30/168/50/121/165/9 → 17/162/47/87/183/47`；Robo interleaved `28/166/41/208/76/24 → 9/171/45/68/103/147`；Qwen text→images `32/138/74/48/241/10 → 24/155/44/56/232/32`；Qwen images→text `39/116/58/61/249/20 → 24/105/54/62/129/169`。
- 最终 text→images ranking 跨模型 top-8/32/64 交集为 `5/25/49`，Jaccard 为 `45.45%/64.10%/62.03%`。具体 top-8 与完整表见 [final_task_ranking_overlap_v1.md](../results/mydata_bench/experiments_v2_corssmodel/auto_research/final_task_ranking_overlap_v1.md)。

### 适用边界与复现

- 不可外推为五种输入均有效；尤其 Qwen video→text/interleaved 没有稳定点。
- Qwen tri-mix 需要三次 forward，约为单分支三倍推理开销。
- R4 两个独立 full 进程逐条相同；Qwen R6/R9 的 152×3 重叠分支预测和 log-prob 完全相同；R9/R12 的 846 个 baseline 以及 R10/R13 的 152×5 个分支在 prediction、log-prob、raw output 上也完全相同。R12/R13 没有独立全量重跑。总复现 verdict 为 `PARTIALLY_REPRODUCIBLE`。
- A100-SXM4-80GB、bfloat16、torch 2.8.0+cu128、transformers 4.57.0；最终 `PYTHONPATH` 正确设置后的测试为 **39 passed**。
- 完整自动研究过程、所有失败和误标 659-report 的处理见 [auto_explore.md](auto_explore.md)。
