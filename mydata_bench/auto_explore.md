# Cross-model attention steering 自动研究记录

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-09-04
- Verification Status: ANALYZED
- Version Label: auto_explore_final_v1

## 目标与不可变约束

- 主目标：RoboReward-8B 与 Qwen3-VL-8B 各自在五种输入构造中至少两种同时稳定改善 MAE、总准确率、suc 准确率和 fail 准确率。
- 禁止以真实标签、suc/fail 身份或输出端点规则控制干预；模型前向阶段不读取标签。
- 数据、旧配置和旧结果只读；新代码保持旧配置默认语义，新产物只写入 `auto_research/`。
- 正式比较只允许同一 cohort、同一输入协议的 paired baseline。

## 2026-09-04：既有证据审计

既有完整 cohort 为 846 条（268 suc / 578 fail）。原方法对所有 k 使用 `swap_bias=6`，并在选中目标 token 加 `+6`、负区域 token 加 `-6`，目标相对负区域的注意力 odds 被乘以 `exp(12)≈1.63e5`。这不是温和 steering，且总干预量随 head 数和负 token 数增加。

已达到“四指标同时改善”的单点只有：

| 模型 | 输入构造 | 配置 | 结果 |
| --- | --- | --- | --- |
| RoboReward-8B | interleaved | all-frame, k=32, bias=6 | MAE 1.5674→0.9468；总 acc 23.52%→56.15%；suc 48.88%→51.49%；fail 11.76%→58.30% |
| Qwen3-VL-8B | image→text | last-frame, k=64, bias=6 | MAE 1.4326→1.0449；总 acc 22.81%→45.63%；suc 55.97%→59.70%；fail 7.44%→39.10% |

因此每个模型还缺至少一种输入构造。失败模式高度一致：强 steering 往往使整体预测下移，改善占多数的 fail，却损害 suc；k 的响应非单调。

## 文献/代码核验与第一轮假设

- PASTA（arXiv:2311.02262v2）通过向 attention logits 加 `log(alpha)` 做乘法重标定，官方示例 `alpha=0.01`，并强调少量 task-specific / task-agnostic heads。
- PAI（arXiv:2407.21771v1）代码只修改每次 attention 调用的最后 query 行，并以 image-free classifier-free guidance 抑制 text inertia。
- Gaze Heads（arXiv:2606.14703v1）与 localization-head 工作支持稀疏、model-specific head 干预；全量 head 干预会损害输出。
- ASCD（arXiv:2506.14766v3）与 CAST（arXiv:2605.04641v2）支持以对比敏感度/模型特定统计选 head，而不只用 raw target mass。

第一轮假设 H1：主要问题不是视觉证据不可达，而是干预剂量与 query scope 失配。采用仅增强目标 token（无负区域压制）、只改每次调用最后 query、并对 k 使用递减 bias，可减少全局分数下移，同时保留 instruction-conditioned target 增益。

## 增量实现

- 新增 `steering_query_scope: last_query`：prefill 与 decode 均只修改最后一行；旧 scope 不变。
- 新增 `swap_bias_by_top_k`：允许 K-aware 剂量；未配置时仍使用原 `swap_bias`。
- 新增 `control_conditions`：探索阶段可只跑 candidate target；默认仍包含 wrong-region 和 low-rank-head。
- 新增任务分层、严格 suc/fail 配对的 pilot cohort：152 条（76 对，覆盖 28 个 task）。该选择只用于探索，不替代 846 条正式验收。
- 新增单元测试；首轮定向测试结果：2 passed。

## 实验运行日志

### Pilot R1（待运行）

共同设置：top-k={16,32,48}，bias={2.0,1.5,1.2}，`last_query`，positive-only (`negative_scope=none`)，无探索期控制分支。三个待筛选输入为 RoboReward text→images、Qwen text→images、Qwen interleaved。

### Pilot R1 结果（完成，152/152，exit code 0）

| 模型/输入 | 条件 | MAE | 总 acc | suc acc | fail acc |
| --- | --- | ---: | ---: | ---: | ---: |
| RoboReward text→images | baseline | 1.5658 | 17.76% | 35.53% | 0.00% |
|  | k16 / k32 / k48 | 1.5395 / 1.5592 / 1.5592 | 15.79% / 16.45% / 16.45% | 31.58% / 32.89% / 32.89% | 0 / 0 / 0 |
| Qwen text→images | baseline | 1.5592 | 42.76% | 85.53% | 0.00% |
|  | k16 / k32 / k48 | 1.5921 / 1.5855 / 1.5658 | 38.82% / 39.47% / 39.47% | 77.63% / 78.95% / 78.95% | 0 / 0 / 0 |
| Qwen interleaved | baseline | 1.6513 | 46.71% | 93.42% | 0.00% |
|  | k16 / k32 / k48 | 1.6447 / 1.6118 / 1.6447 | 46.71% / 46.71% / 46.71% | 93.42% / 93.42% / 93.42% | 0 / 0 / 0 |

判定：H1 不成立。Qwen interleaved k32 的 paired absolute-error change 为 -0.0395，cluster bootstrap 95% CI [-0.0658,-0.0132]，但没有任何 fail 到达 exact label=1，准确率验收不改善。温和 target-only 只能轻微移动中间分数，不能解决两端判定；Qwen text→images 还显著损害 exact accuracy（k16 McNemar p=0.03125）。

### 第二轮假设 H2：关系联合区域

任务不是“是否看见被操作物”，而是“物体与 destination/receptacle 是否形成完成关系”。现有 parser/grounding 有意只保留 manipulated object；对称 target-span steering 会把 plate/bowl 等 destination 当作 non-target 压制。H2 将从指令语义（不读标签）解析 destination，分别映射 object bbox 与 destination bbox 到视觉 token 后取集合并集，不使用包围两者的大矩形；再进行 last-query steering。对没有 destination 的指令保持原 target-only 行为。

### Pilot R2：关系联合区域（完成，三组均 608/608，exit code 0）

新增确定性 placement-reference parser，并用 SAM3 对 pilot 的最后帧 destination 做独立 grounding。v2 结果为 152/152：150 `ok`，2 `no_detection` 按预注册规则回退 target-only；模型前向文件不含标签。共同干预为 last-frame、`target_reference_union`、`last_query`、`negative_scope=target_span`，top-k={8,16,32}，bias={6,5,4}。运行记录确认 union 使用离散 bbox token 集合并集，且正负 token 不重叠。

| 模型/输入 | 条件 | MAE | 总 acc | suc acc | fail acc |
| --- | --- | ---: | ---: | ---: | ---: |
| RoboReward text→images | baseline | 1.5658 | 17.76% | 35.53% | 0.00% |
|  | k8 / k16 / k32 | 1.6250 / 1.6250 / 1.6645 | 15.13% / 17.11% / 16.45% | 26.32% / 30.26% / 28.95% | 3.95% / 3.95% / 3.95% |
| Qwen text→images | baseline | 1.5592 | 42.76% | 85.53% | 0.00% |
|  | k8 / k16 / k32 | 1.5921 / 1.5461 / 1.5395 | 40.13% / 40.13% / 38.82% | 78.95% / 80.26% / 77.63% | 1.32% / 0 / 0 |
| Qwen interleaved | baseline | 1.6513 | 46.71% | 93.42% | 0.00% |
|  | k8 / k16 / k32 | 1.6250 / 1.6842 / 1.6645 | 42.11% / 46.71% / 48.68% | 84.21% / 93.42% / 96.05% | 0 / 0 / 1.32% |

判定：H2 不满足验收。Qwen interleaved k32 同时提高总/suc/fail exact accuracy，但 MAE 反向增加；Qwen text→images 的 k16/k32 仅改善 MAE而损害 suc；RoboReward 虽出现少量 fail exact correction，但整体与 suc 明显恶化。关系对象缺失确是原设计的一项结构性问题，但静态 task raw-mass ranking 仍不能形成稳定的 instruction-conditioned readout。

### 第三轮假设 H3：独立 gaze-head ranking

保持 R2 的输入、区域、query scope、negative scope 与剂量完全不变，只把 head ranking 替换为两模型各自在 500 条独立 comics 上得到的 gaze ranking。该 ranking 与评测 cohort 无交集，Qwen/Robo top-32 高度重合，但与 task raw-mass top-32 分别只重合 5/4 个，因而能隔离检验“静态 task ranking 选错输出读取 head”的解释。

### Pilot R3 结果（完成，三组均 608/608，exit code 0）

| 模型/输入 | 条件 | MAE | 总 acc | suc acc | fail acc |
| --- | --- | ---: | ---: | ---: | ---: |
| RoboReward text→images | baseline | 1.5658 | 17.76% | 35.53% | 0.00% |
|  | k8 / k16 / k32 | 1.5658 / 1.5855 / 1.5855 | 15.13% / 15.79% / 14.47% | 30.26% / 31.58% / 28.95% | 0 / 0 / 0 |
| Qwen text→images | baseline | 1.5592 | 42.76% | 85.53% | 0.00% |
|  | k8 / k16 / k32 | 1.5921 / 1.5987 / 1.6316 | 41.45% / 38.16% / 38.16% | 82.89% / 76.32% / 76.32% | 0 / 0 / 0 |
| Qwen interleaved | baseline | 1.6513 | 46.71% | 93.42% | 0.00% |
|  | k8 / k16 / k32 | 1.6118 / 1.5921 / 1.6184 | 42.76% / 42.11% / 40.13% | 85.53% / 84.21% / 80.26% | 0 / 0 / 0 |

判定：H3 不成立。外部 gaze heads 可以轻微降低部分 MAE，但没有产生任何 fail exact correction，并系统性损害 suc；通用“看向视觉”的 head 并不等同于任务输出阶段的关系判定 head。

### 第四轮假设 H4：关系区域与已验证强路径结合

旧全量结果表明 all-query/all-frame、bias=6 确实能让 fail prediction 跨过 exact 端点，但经常因全局下移损害 suc。H4 不再改变 ranking，而是在这条已有因果效应的路径上加入 relation union，并细查 k={16,24,32} 邻域：若 destination 信息能减少 suc 的误伤，应在保留 fail 改善的同时恢复 suc。该轮是机制组合实验，探索期仍不运行控制分支。

### Pilot R4 结果（Robo/Qwen text→images 完整；Qwen interleaved 非正式）

| 模型/输入 | 条件 | MAE | 总 acc | suc acc | fail acc |
| --- | --- | ---: | ---: | ---: | ---: |
| RoboReward text→images | baseline | 1.5658 | 17.76% | 35.53% | 0.00% |
|  | k16 / k24 / k32 | 1.4079 / 1.3947 / 1.4605 | 35.53% / 38.16% / 34.21% | 55.26% / 51.32% / 47.37% | 15.79% / 25.00% / 21.05% |
| Qwen text→images | baseline | 1.5592 | 42.76% | 85.53% | 0.00% |
|  | k16 / k24 / k32 | 1.6053 / 1.5855 / 1.7303 | 44.74% / 44.08% / 45.39% | 88.16% / 86.84% / 90.79% | 1.32% / 1.32% / 0 |
| Qwen interleaved* | baseline | 1.6513 | 46.71% | 93.42% | 0.00% |
|  | k16 / k24 / k32* | 1.8067 / 1.8667 / 1.9000 | 48.00% / 50.67% / 51.33% | 95.95% / 98.65% / 100% | 1.32% / 3.95% / 3.95% |

`*` Qwen interleaved 有 2 条 suc 在三个 candidate 条件均只输出裸 `5`，不符合冻结的 `ANSWER: <1-5>` contract；因此 candidate n=150、`formal_scoring_ready=false`，这些数字仅用于失败模式诊断。确定性重跑不会修复格式性失败，故不重试、不用于验收。

判定：H4 为 RoboReward text→images 找到首个新的稳健候选，k=16/24/32 全部同时改善 MAE、总准确率、suc accuracy 和 fail accuracy，下一步进入 846 条全量复验。对 Qwen，关系 union 确实保护甚至提高 suc，但同时令 fail 普遍升分，因此 MAE恶化；需要把 target-specific effect 与共有的 score drift 分离。

### 第五轮假设 H5：target-vs-control 对比解码

依据 PAI/ASCD 的正负分支思想，对同一无标签输入分别运行 instruction-grounded target/reference 区域与等面积 disjoint control 区域。记录严格 score token 对 1–5 的 log-prob，并比较 `positive + α(positive−control)` 与 `baseline + α(positive−control)`；共有的上/下分漂移在差分中抵消，只有区域特异、指令条件化的证据被放大。真实 suc/fail 标签仍只在两个分支全部推理完成后由 scorer 读取。新增 `record_discrete_label_logprobs` 默认关闭，旧实验语义不变；定向测试通过。

### Pilot R5 结果（Qwen text→images 完整；interleaved 非正式）

- text→images：1064/1064 均 `ok`，即 152 条 ×（baseline + 3 个 k 的 target/control）。baseline score-token argmax 与原生 greedy 输出逐条一致，排除了取 token 位置错误。
- interleaved：1034 `ok`、10 个不同样本各 1 条 `sample_failure`，均为模型仅输出裸 `5`；不重试、不作正式比较。
- text→images 对 α={0.25,0.5,1,2,4,8,16,32,64}、k={8,16,32} 和两种 anchor 进行完整网格评估。没有任一点同时改善四指标。较强 α 只能让 fail accuracy 从 0 提到 2.63%–5.26%，但 suc accuracy 从 85.53% 降到约 63%–66%，MAE与总体 accuracy也恶化。

判定：H5 不成立。last-query 的 target-vs-wrong residual 仍主要表达“区域显著性变化”，没有形成成功/失败关系的可分方向。

### 第六、七轮假设：强正分支对比与关系残差插值

- H6 在 H4 已验证能保护 Qwen suc 的 all-query/all-frame relation-union 正分支上加入 matched wrong-region 分支，检查对比差分能否消除 fail 的共同升分。
- H7 不再用任意 wrong region，而用同 head、同剂量、同时间范围的 **target-only** 分支作为控制。`control + α(union−control)` 在 α∈[0,1] 是原始 target-only（倾向整体降分）与 relation-union（保护 suc、但倾向升分）的概率空间插值；它不读取标签，也不根据预测端点切换，仅隔离 destination 对输出的增量效应。新增 `candidate_target_only` 为可选控制，旧默认不变。

### Pilot R6/R7 结果（两轮均 1064/1064、零 invalid）

H6 在完整的 k={16,24,32} × α 网格中出现两个相邻通过点，均为 k24、`control_plus_delta`：

| 方法 | MAE | 总 acc | suc acc | fail acc |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.5592 | 42.76% | 85.53% | 0.00% |
| H6 k24 α=0.50 | 1.5000 | 47.37% | 86.84% | 7.89% |
| H6 k24 α=0.75 | 1.5461 | 44.74% | 88.16% | 1.32% |

H7 只有 k24、α=0.8 一个孤立通过点：MAE 1.5329、总 acc 44.74%、suc 86.84%、fail 2.63%；相邻 α 不通过，且效应弱于 H6。因此 H7 作为机制对照保留，不进入正式复验。

判定：选择 H6 的 k24，并在 full cohort 上预先固定 α={0.50,0.75}；完整推理仍保留 k16/24/32 以检查 k 稳健性。因为 152 条 pilot 用于选参，正式报告除 846 条总体外必须单列排除 pilot 后的 694 条 held-out，防止开发集重用造成乐观偏差。

### 第八轮：补齐原生 video→text 输入

既有实验已覆盖两模型的 image→text、text→image、interleaved，以及 Qwen text→video、RoboReward text→video/video→text；尚缺 Qwen video→text attention intervention。R8 使用冻结的原生 MP4 protocol、`video_then_text` 顺序与 Qwen native ranking，测试 all-frame/all-query relation union 的 k={16,24,32}，以补齐五种输入构造的探索矩阵。该轮只作为输入覆盖/泛化检验，不替代 H6 的预注册全量复验。

R8 为 608/608、零 invalid。baseline：MAE 1.5395、总 acc 33.55%、suc 64.47%、fail 2.63%。k16/k24/k32 的 MAE 分别为 1.6974/1.7434/1.7303，总 acc 38.82%/38.82%/30.92%，suc 77.63%/77.63%/61.84%，fail 均为 0。判定：relation union 在 Qwen 原生 video→text 上同样造成整体升分，未满足验收；但至此五种输入构造均已有基线/attention 探索证据，不能把最终结论外推为普适于所有顺序。

## 首个 846 条正式复验：RoboReward text→images（成功）

全量 destination grounding 为 846/846：838 `ok`、8 `no_detection` 按冻结规则回退 target-only，目的地均从指令解析为 `plate`。正式 inference 为 3384/3384、零 invalid、`formal_scoring_ready=true`。

| 条件 | MAE | 总 acc | suc acc | fail acc | absolute-error change cluster 95% CI | McNemar p |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| baseline | 1.4965 | 17.49% | 48.13% | 3.29% | — | — |
| k16 | 1.3026 | 32.39% | 63.43% | 17.99% | [-0.307,-0.156] | 5.33e-26 |
| k24 | 1.3121 | 34.28% | 57.09% | 23.70% | [-0.296,-0.132] | 2.58e-27 |
| k32 | 1.4835 | 30.14% | 51.87% | 20.07% | [-0.154,0.011] | 1.73e-16 |

预测分布（label 1→5）从 baseline `{1:19,2:316,3:277,4:44,5:190}` 变为 k16 `{1:116,2:323,3:139,4:13,5:255}`、k24 `{1:155,2:318,3:97,4:28,5:248}`、k32 `{1:132,2:313,3:87,4:48,5:266}`。k16/k24 的误差改善在按 task 分层、按 video cluster 的 10,000 次 bootstrap 中 CI 全部低于 0；k32 虽四个点估计均改善，但误差 CI 跨 0，故最终稳健范围优先写作 k=16–24。

结论：RoboReward 已达到第二个输入候选。结合既有 interleaved k32 全量成功点，RoboReward 部分满足“两种输入”主线门槛；仍需补控制实验并等待 Qwen 正式复验。

## 第九轮：受约束 Qwen H6 全量复验（H6 被否证）

未约束的 R6 全量运行最终得到 5919 行：5918 `ok`、1 个 `sample_failure`。失败样本为 `fail/ljx_lfz_task_1_3/17`，在完成 k16 后的后续生成中输出长解释而不是严格 `ANSWER: <1-5>`；因此 k24/k32 各缺 1 条。该运行按规则没有自动重试，也不作正式验收。

R9 改用 trie 受约束解码，只允许协议合法字符串，同时固定 pilot 预选的 k24。推理为 2538/2538（846 × baseline/relation-target/matched-wrong）、零 invalid；R9 与未约束 pilot R6 的 152 个重叠样本在三个对应分支上的预测与 1–5 score-token log-prob 均逐值完全一致，说明约束只消除了格式失败，没有改变原本合法输出。

预固定 H6 两个点在 full 与排除 pilot 后的 694 held-out 都失败：

| cohort | 方法 | MAE | 总 acc | suc acc | fail acc |
| --- | --- | ---: | ---: | ---: | ---: |
| full 846 | baseline | 1.5934 | 28.25% | 84.33% | 2.25% |
|  | α=0.50 | 1.7092 | 36.17% | 82.09% | 14.88% |
|  | α=0.75 | 1.9267 | 28.37% | 84.33% | 2.42% |
| held-out 694 | baseline | 1.6009 | 25.07% | 83.85% | 2.59% |
|  | α=0.50 | 1.7550 | 33.72% | 80.21% | 15.94% |
|  | α=0.75 | 2.0101 | 24.78% | 82.81% | 2.59% |

α=0.50 的 exact accuracy/McNemar 虽改善，但 MAE 与 suc 明显恶化；α=0.75 的绝对误差反而显著恶化。对三种既有公式和 α=0–2 的后验网格也没有四指标通过点。结论：两分支 H6 的 pilot 成功没有跨 cohort 泛化，不能作为最终方案。

## 第十轮：三分支凸 log-prob 混合

R9 失败暴露的结构约束是：relation-target 分支保护 suc 但整体偏高，matched-wrong 分支改善 fail 但严重损害 suc；只在两者之间插值会完全丢掉稳定的 baseline。H10 将三个分支的 1–5 score-token log-prob 作固定凸组合：

`ℓ_mix = w_b ℓ_baseline + w_t ℓ_relation-target + w_c ℓ_matched-wrong`，其中 `w_b+w_t+w_c=1` 且全部非负，再取 1–5 argmax。

这是 geometric/product-of-experts 式的三路校准：baseline 保留原判断，少量 relation-target 提供关系证据，matched-wrong 抵消 text-first 的高分漂移。所有样本、task、suc/fail 使用同一组固定权重；模型前向不读标签，也没有按预测端点切换的规则。

在 k24 上，权重线段 `(0.70,0.05,0.25) → (0.60,0.10,0.30)` 的两个端点与中点 `(0.65,0.075,0.275)` 均在 152 pilot、694 held-out 与 full 846 上同时改善四指标：

| cohort | 权重 `(baseline,target,wrong)` | MAE | 总 acc | suc acc | fail acc | abs-error change 95% CI | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| full 846 baseline | — | 1.5934 | 28.25% | 84.33% | 2.25% | — | — |
| full 846 | (0.70,0.05,0.25) | 1.5343 | 30.97% | 85.07% | 5.88% | [-0.0694,-0.0162] | 5.65e-6 |
| full 846 | (0.65,0.075,0.275) | 1.5236 | 31.44% | 85.82% | 6.23% | [-0.0881,-0.0307] | 1.40e-6 |
| full 846 | (0.60,0.10,0.30) | 1.5260 | 31.56% | 85.45% | 6.57% | [-0.0886,-0.0301] | 1.94e-6 |
| held-out 694 baseline | — | 1.6009 | 25.07% | 83.85% | 2.59% | — | — |
| held-out 694 | (0.70,0.05,0.25) | 1.5317 | 28.10% | 84.38% | 6.57% | [-0.0856,-0.0248] | 5.72e-6 |
| held-out 694 | (0.65,0.075,0.275) | 1.5173 | 28.53% | 85.42% | 6.77% | [-0.1120,-0.0450] | 3.03e-6 |
| held-out 694 | (0.60,0.10,0.30) | 1.5202 | 28.67% | 84.90% | 7.17% | [-0.1129,-0.0447] | 4.65e-6 |

中点 full 的 corrected/harmed 为 30/3，held-out 为 26/2。其 full pairwise suc−fail bins 从 baseline `<0/0/1/2/3/4 = 32/138/74/48/241/10` 改为 `25/约158/约40–57/约49–57/约229–238/29–34`（精确到每个权重的表见对应记录）；负差减少，差值 4 明显增加。完整 task accuracy、suc/fail 与 task 预测分布、pairwise 表见 R9 目录的 `tri_mix_wA_*`、`tri_mix_wMid_*`、`tri_mix_wB_*` Markdown/JSON。

需要明确：三分支方法是在观察 R9 full 的两分支失败后提出，权重范围也经历探索，因此 694 子集不能被描述成完全未触碰的确认集。它是强、透明的探索性证据，而不是预注册确认性结果。为减少这一 look-elsewhere 风险，后续另在 interleaved pilot 上冻结一个跨 k 权重，再做从未用于该权重选择的 846 条全量复验。

## R10/R11：新输入泛化 pilot

- R10 为受约束 interleaved all-query/all-frame relation-target vs matched-wrong，1064/1064、零 invalid。原 H6 的三类两分支公式没有任何四指标通过点；但三分支权重 `(0.55,0.10,0.35)` 在 k24 与 k32 都通过 pilot 四指标，已冻结进入 R13 全量。
- R11 为受约束原生 video→text relation-union vs target-only，1064/1064、零 invalid。三类两分支公式没有通过点；三分支凸网格只在 k24 的 `(0,0.15,0.85)` 出现单一通过点，k16/k32 均没有，判为不稳定，不进入全量。

## RoboReward 全量控制与 held-out

R4 控制实验为 5922/5922、零 invalid、`formal_scoring_ready=true`。另一个独立进程完整重复了 baseline 与 candidate-target k16/k24；三条件各 846 个预测逐条完全一致，构成确定性同环境复现。

| 条件 | MAE | 总 acc | suc acc | fail acc |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.4965 | 17.49% | 48.13% | 3.29% |
| target k16 | 1.3026 | 32.39% | 63.43% | 17.99% |
| wrong-region k16 | 1.4504 | 7.21% | 8.58% | 6.57% |
| low-rank k16 | 1.5579 | 16.31% | 47.01% | 2.08% |
| target k24 | 1.3121 | 34.28% | 57.09% | 23.70% |
| wrong-region k24 | 1.3806 | 15.96% | 8.21% | 19.55% |
| low-rank k24 | 1.5804 | 14.66% | 41.04% | 2.42% |

wrong-region 有时能降低 MAE，但无法同时保护 suc/总体准确率；low-rank 在两个 k 都不通过。只有正确区域 + 高排名 head 同时满足四指标，支持区域与 head 选择性。

排除用于选 k 的 152 pilot 后，RoboReward R4 在 694 held-out 上仍有连续 k=16–24 范围通过：baseline MAE/总/suc/fail 为 `1.4813/17.44%/53.13%/3.78%`；k16 为 `1.2795/31.70%/66.67%/18.33%`，abs-error CI `[-0.315,-0.127]`、McNemar p `1.06e-21`；k24 为 `1.2939/33.43%/59.38%/23.51%`，CI `[-0.292,-0.090]`、p `3.03e-22`。

补充敏感性分析也确认既有两个成功点在同一 694 子集保持四指标方向：RoboReward interleaved k32 的 MAE `1.5576→0.8876`、总 `23.63%→56.92%`、suc `52.08%→54.69%`、fail `12.75%→57.77%`；Qwen images→text k64 的 MAE `1.3977→0.9784`、总 `22.19%→47.84%`、suc `58.33%→60.94%`、fail `8.37%→42.83%`。后者的 MAE CI `[-0.534,-0.369]`、McNemar p `4.68e-35`。

## 工程验证

- 全套测试第一次因 `conda run` 未把仓库根目录加入模块搜索路径而在 collection 阶段报 `ModuleNotFoundError`，没有执行测试；显式设置 `PYTHONPATH` 后为 **39 passed**。
- 新增的固定 direct held-out scorer 与 full 运行自带指标逐项完全一致；三分支报告器在 pilot 上与既有两分支 scorer 的原公式逐项完全一致。
- 控制 scorer 曾因异步命令状态判断被重复启动一次；第二次写出的 `steering_metrics.json` 与第一次 SHA-256 完全相同（`ac10b38...4484536`），`steering_invalid.json` 也完全相同。原始 5922 行推理文件未被修改。

## R12：Qwen text→images 的 k32 边界复验

R12 使用与 R9 相同的受约束输出协议、task-specific ranking、relation-target 与 matched-wrong 三分支，只把 top-k 固定为 32，并复用 k24 已保留权重线段的保守端点 `(baseline,target,wrong)=(0.70,0.05,0.25)`。推理为 2538/2538（3 条件 × 846）、全部 `ok`、零重复 `(example_id,condition)`、每条件恰好 846 条且全部记录 1–5 score-token log-prob；原始行不含 `label`、`target_label`、`ground_truth` 或 `split` 字段。

| cohort | 方法 | MAE | 总 acc | suc acc | fail acc | abs-error change cluster CI | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| full 846 | baseline | 1.5934 | 28.25% | 84.33% | 2.25% | — | — |
| full 846 | k32 `(0.70,0.05,0.25)` | 1.5638 | 32.15% | 86.57% | 6.92% | `[-0.0399,0.0265]` | 3.61e-8 |
| held-out 694 | baseline | 1.6009 | 25.07% | 83.85% | 2.59% | — | — |
| held-out 694 | k32 `(0.70,0.05,0.25)` | 1.5677 | 29.25% | 85.94% | 7.57% | `[-0.0602,0.0248]` | 1.31e-7 |

full corrected/harmed 为 36/3，held-out 为 31/2；full pairwise bins 从 `<0/0/1/2/3/4 = 32/138/74/48/241/10` 变为 `24/171/21/72/221/34`。四项点估计在 full 与 held-out 均通过，exact accuracy 的配对改善很强，但按 task 分层、video cluster 重采样的 MAE CI 跨 0。因此 k32 只作为 k24 稳定结果的支持性边界，不单独声称稳健 MAE 改善；Qwen text→images 的主要证据仍是 k24 权重连续线段及其 CI 全低于 0。

报告生成时曾误把 `auto_explore_split_v1/pilot_ids.json`（187 IDs）当成 R1–R11 使用的 152-ID pilot 文件，产生文件名为 `tri_mix_wA_heldout694_v1.*`、实际 `analysis_id_count=659` 的附加报告。发现后没有删除或覆盖该产物，而是用正确的 `cohorts/auto_research_pilot_v1/example_ids.json` 新增 `tri_mix_wA_heldout694_v2.*`；前一文件明确弃用，任何结论只引用 v2。

## R13：Qwen interleaved 冻结权重全量验证（失败）

R13 在 R10 的 152-record pilot 上先冻结 `(baseline,target,wrong)=(0.55,0.10,0.35)`，再对 k={24,32} 运行全量。推理为 4230/4230（5 条件 × 846）、全部 `ok`、零重复键、每条件 846 条、零缺失 score-token log-prob，原始推理行中同样没有标签或 split 字段。

| cohort | 方法 | MAE | 总 acc | suc acc | fail acc | abs-error change cluster CI | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| full 846 | baseline | 1.7057 | 30.97% | 94.78% | 1.38% | — | — |
| full 846 | k24 frozen mix | 1.7423 | 32.51% | 97.01% | 2.60% | `[-0.0067,0.0602]` | 9.77e-4 |
| full 846 | k32 frozen mix | 1.7305 | 34.99% | 96.64% | 6.40% | `[-0.0240,0.0465]` | 5.40e-9 |
| held-out 694 | baseline | 1.7176 | 27.52% | 95.31% | 1.59% | — | — |
| held-out 694 | k24 frozen mix | 1.7651 | 28.96% | 97.40% | 2.79% | `[0.0051,0.0811]` | 6.35e-3 |
| held-out 694 | k32 frozen mix | 1.7522 | 31.84% | 96.88% | 6.97% | `[-0.0180,0.0682]` | 6.94e-8 |

两个 k 都提高总、suc、fail exact accuracy，却使 full 与 held-out MAE 恶化，故严格判为失败，不因 exact p-value 很小而通过。k24/k32 full 的 corrected/harmed 分别为 14/1、36/2；pairwise 负差确实从 8 降至 2/1、差值 4 从 4 增至 10/27，但 fail 的 label-5 数也从 250 增至 264/274，解释了“端点命中更多、平均误差反而更大”的表面矛盾。R13 没有重试或后验改权重，是 pilot 过拟合和单指标误判的负向确认。

## 最终五输入覆盖矩阵

以下每格使用该输入中最有代表性的冻结/已记录条件；“通过”要求 MAE、总 accuracy、suc accuracy、fail accuracy 四项同时改善。

| 模型 | 输入构造 | baseline `MAE/总/suc/fail` | 代表条件 `MAE/总/suc/fail` | 判定 |
| --- | --- | --- | --- | --- |
| RoboReward | text→video | `2.1206/24.94/78.73/0.00%` | all-frame k32 `1.2092/25.65/63.81/7.96%` | 未通过：suc 下降 |
| RoboReward | video→text | `0.8333/69.15/68.28/69.55%` | all-frame k8 `0.7920/70.92/70.90/70.93%` | **通过** |
| RoboReward | text→images | `1.4965/17.49/48.13/3.29%` | relation-union k16 `1.3026/32.39/63.43/17.99%` | **通过；k16–24 稳健** |
| RoboReward | images→text | `0.8203/64.78/59.33/67.30%` | k32 `0.6974/68.09/48.51/77.16%` | 未通过：suc 下降 |
| RoboReward | interleaved | `1.5674/23.52/48.88/11.76%` | all-frame k32 `0.9468/56.15/51.49/58.30%` | **通过** |
| Qwen | text→video | `1.4586/29.20/89.55/1.21%` | all-frame k64 `1.1608/35.22/67.54/20.24%` | 未通过：suc 下降 |
| Qwen | video→text | R8 pilot `1.5395/33.55/64.47/2.63%` | relation-union k24 `1.7434/38.82/77.63/0.00%`；R11 也无稳定点 | 未通过 |
| Qwen | text→images | `1.5934/28.25/84.33/2.25%` | tri-mix k24 midpoint `1.5236/31.44/85.82/6.23%` | **通过；连续权重线段稳定** |
| Qwen | images→text | `1.4326/22.81/55.97/7.44%` | last-frame k64 `1.0449/45.63/59.70/39.10%` | **通过** |
| Qwen | interleaved | `1.7057/30.97/94.78/1.38%` | R13 frozen k32 `1.7305/34.99/96.64/6.40%` | 未通过：MAE 恶化 |

最终计数为 RoboReward **3/5**、Qwen **2/5**，均达到“至少两种输入构造”的数值验收目标。Qwen 的第二条 text→images 证据属于后验探索，不能升级为预注册确认性证据；这一限定不改变四指标门槛已经在 full、pilot-excluded 694、连续权重线段上同时满足的事实。

## 最终行为剖面

### Task 方向

| 方案 | overall task up/same/down | suc task up/same/down | fail task up/same/down |
| --- | --- | --- | --- |
| RoboReward text→images k16 | 24/2/2 | 见完整记录 | 见完整记录 |
| RoboReward interleaved k32 | 27/1/0 | 见完整记录 | 28/0/0 |
| Qwen text→images k24 midpoint | 8/19/1 | 5/22/1 | 3/25/0 |
| Qwen images→text k64 | 22/4/2 | 13/7/8 | 17/11/0 |

这说明总体通过不等于每个 task 都提高；Qwen text→images 的增益集中在少数 task，但没有 fail task 变差。完整 28-task accuracy 与每 task 预测分布保存在各方案的 `exp_record.md` / `tri_mix_*.md`。

### suc/fail 预测分布

- RoboReward text→images k16：suc 从 `{1:0,2:38,3:73,4:28,5:129}` 变为 `{1:12,2:40,3:43,4:3,5:170}`；fail 从 `{1:19,2:278,3:204,4:16,5:61}` 变为 `{1:104,2:283,3:96,4:10,5:85}`。
- Qwen text→images k24 midpoint：suc 从 `{1:0,2:5,3:15,4:22,5:226}` 变为 `{1:1,2:5,3:17,4:15,5:230}`；fail 从 `{1:13,2:261,3:60,4:76,5:168}` 变为 `{1:36,2:253,3:71,4:46,5:172}`。
- RoboReward interleaved k32 与 Qwen images→text k64 的完整 split/task 分布见原 `exp_record.md`，并在 694 pilot-excluded scorer 中保持四指标方向。

### Pairwise suc−fail bins

| 方案 | baseline `<0/0/1/2/3/4` | candidate `<0/0/1/2/3/4` |
| --- | --- | --- |
| RoboReward text→images k16 | `30/168/50/121/165/9` | `17/162/47/87/183/47` |
| RoboReward interleaved k32 | `28/166/41/208/76/24` | `9/171/45/68/103/147` |
| Qwen text→images k24 midpoint | `32/138/74/48/241/10` | `24/155/44/56/232/32` |
| Qwen images→text k64 | `39/116/58/61/249/20` | `24/105/54/62/129/169` |

四个最终方案都减少负差并增加差值 4，说明改善不只是类别边际准确率变化，也增强了同视频不同指令的区分度。

## Head overlap

最终 text→images task rankings 的 top-8 为：

- RoboReward：`L22H15,L19H28,L19H31,L21H16,L22H5,L19H17,L20H15,L19H10`
- Qwen：`L21H25,L21H16,L19H31,L19H23,L22H15,L20H15,L19H28,L21H29`

跨模型 top-8/32/64 交集分别为 5/8、25/32、49/64，Jaccard 分别为 45.45%、64.10%、62.03%。共享较多但排序不同，支持 model-specific ranking，而不支持把一个模型的顺序直接硬迁移给另一个模型。完整表见 `final_task_ranking_overlap_v1.md`。

## 最终方法、边界与复现性结论

1. **RoboReward relation-aware direct steering**：在 text→images 中把 manipulated object 与从指令解析/grounding 的 destination token 取集合并集，all-frame/all-query、task ranking、bias=6、k=16–24。wrong-region 和 low-rank 控制均不能同时通过四指标，支持区域与 head 选择性。
2. **Qwen baseline-preserving tri-mix**：对 baseline、relation-target、matched-wrong 三个分支的 1–5 score-token log-prob 作全局固定非负凸组合，再 argmax。text→images 的 k24 权重线段 `(0.70,0.05,0.25)→(0.60,0.10,0.30)` 全部通过；推荐报告中点而非宣称唯一最优点。该方法需要三次分支推理。
3. **无端点作弊**：所有 forward 只接收同一条无标签输入及其 instruction-derived 区域；权重不随样本、task、suc/fail、预测端点或真实标签变化。标签只在推理完成后的 scorer 中使用。
4. **外推边界**：结论限于成功自动 grounding 的 846/1213（69.74%）记录、A100/bfloat16 环境与当前模型版本；video→text 和 Qwen interleaved 没有稳定通过，不能声称五输入普适。
5. **统计身份**：RoboReward 新方案证据较强；Qwen text→images 是透明的后验探索性结果。R13 的冻结跨 k interleaved 复验失败，因此总体置信度为 `CAUTION`，未来需要全新 cohort/benchmark 做真正确认。
6. **复现性**：RoboReward 两个独立 full 进程的 2538 个对应预测完全一致；Qwen R6/R9 的 152×3 重叠分支预测及 log-prob 完全一致。R12/R13 未独立全量重跑，故总 verdict 为 `PARTIALLY_REPRODUCIBLE`。

补充交叉运行审计：R9 与 R12 的 846 条 baseline 在 native prediction、1–5 log-prob、raw output 上均零差异；R10 与 R13 的 152×5=760 个重叠 interleaved 分支也在三类字段上零差异。这证明 R13 的失败不是约束解码或运行漂移造成，而是冻结权重确实没有跨 cohort 泛化。

11/11 statistical fallacy scan、多重比较说明、环境与复现性表见 `results/mydata_bench/experiments_v2_corssmodel/auto_research/validation_report_v1.md`。最终测试为 **39 passed**。
