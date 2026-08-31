# 跨模型 Attention Steering 自动探索日志

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-16T00:00:00+08:00
- Integrity Pass Date: 2026-08-17T03:11:21+08:00
- Verification Status: ANALYZED
- Version Label: crossmodel_auto_explore_v2
- Upstream Plan: `mydata_bench/exp_plan_crossmodel_auto_explore.md`
- Upstream Dependencies: frozen development/screening/held-out partition `4d65d16add4129b3afc0b16f24495189d47580e5688b7167b4921c0cd528fee2`

## 0. 研究目标与不可违反的边界

本日志严格以 `exp_plan_crossmodel_auto_explore.md` 为研究协议。主线目标是：

1. 补全相关文献、开源实现与方法谱系；
2. 从理论和实验上改良现有 attention steering；
3. 最终使 RoboReward-8B 与 Qwen3-VL-8B **分别**在五种输入构造中的至少两种输入下，top-k=8/32/64 均相对同输入 baseline 同时满足：MAE 降低、suc 准确率提高、fail 准确率提高。

禁止端点 hard coding、标签泄漏、按评测标签选干预、跨模型/跨输入复用 ranking 等作弊做法。既有数据、配置和结果只读；新配置与新结果使用独立名称和目录。只使用计划授权的 GPU 0、1、2。自动 grounding cohort 未经人工审核，因此所有结论均限定为 exploratory，不作普遍因果表述。

### 操作化验收标准

- 每个候选输入必须以该输入、该模型的 `baseline` condition 为配对基线；不能拿不同 prompt 或完整 1213 样本 baseline 横向替代。
- top-k=8、32、64 三个条件都必须在同一冻结 cohort 上有完整有效样本，并同时满足三项严格方向性不等式：`ΔMAE<0`、`ΔAcc_suc>0`、`ΔAcc_fail>0`。
- “稳定提升”除方向性外，还检查 episode/video 聚类 bootstrap 95% CI、跨 task 异质性、预测分布、同视频 pairwise 区分度以及干预特异性对照；若仅总体方向改善但 CI 跨 0，标为“方向达标但证据不足”，不冒充强验证。
- 候选方法与超参数只能由 ranking/开发资料或无标签/成功轨迹证据确定；最终评测 cohort 的 suc/fail 标签不能用于逐样本干预或端点选择。

## 1. 启动审计（2026-08-16）

### 1.1 运行环境

- 仓库实际解析路径：`/mnt/public1/dais/workspace/Robo-Dopamine`（用户路径 `/home/dais/workspace/Robo-Dopamine` 指向同一位置）。
- GPU：4×A100 80GB；启动时 0、1、2、3 均空闲。按新计划仅使用 0、1、2。
- Conda：`robo-dopamine`、`rewardbench-sam3` 均存在。
- 共享盘剩余约 608GB，使用率 99%。后续优先复用既有 ranking/grounding，先小规模筛选，再扩展全量，避免重复生成无价值大产物。
- 启动时未发现本项目仍在运行的 RoboReward/Qwen/attention 实验进程。

### 1.2 已有证据基线

`results/mydata_bench/experiments_v2_corssmodel/` 已包含 4 个完整集 baseline 与 12 个 attention 实验；attention cohort 为 846 条（268 suc、578 fail）。五种输入构造可对应为 native `text→video`、`video→text`，独立图像 `text→images`、`images→text`，以及 `interleaved`。当前 crossmodel 目录覆盖后三种；native 两种结果位于 `experiments_v2/`，后续统一复核。

现有结果的关键事实：

- RoboReward `images→text` baseline 已较强（cohort MAE 0.8203，suc/fail 准确率 59.33%/67.30%）。last-frame steering 的 k=8/32/64 均降低 MAE，但三个 k 都牺牲 suc 准确率；不满足主线标准。
- RoboReward `text→images` all-frame 在 k=8/32/64 均降低 MAE并大幅提高 fail 准确率，但 suc 准确率分别从 48.13% 变为 43.28%/45.52%/20.52%；仍不达标。
- RoboReward `interleaved` all-frame 的 k=32 很强（MAE 1.5674→0.9468，总准确率 23.52%→56.15%），但 k=8 fail 准确率下降，k=64 suc/fail 改善不足；只说明方法有可利用的作用窗口。
- Qwen `text→images` last-frame 的 k=32/64 显著降低 MAE并提高 fail 准确率，但 suc 准确率下降；k=8 同样未平衡提升。
- Qwen `interleaved` all-frame 在 k=32 有明显总体改善，但 k=8、k=64 仍存在 suc/fail 权衡，且部分非 target 条件有效样本少于 846，必须先处理完整性再比较。
- 整体呈现一致的“干预强度窗口”：固定 bias=6 时，k=8 经常过弱，k=32 常最有效，k=64 经常过强并把预测分布推向低分；这比“方法完全无效”更符合现有证据。

### 1.3 首轮机制假设（待文献与实验验证）

1. **干预剂量随 k 失配**：固定每 head bias 导致总干预容量随 head 数增大；k=8/32/64 实际不是同一方法强度的稳健性测试。应研究按 head 数、目标/负 token 数或原始 attention mass 归一化的剂量。
2. **正负区域不守恒**：`+6` 施加于少量 target token、`-6` 施加于大量非 target visual token，softmax 后的注意力分布变化依赖 token 数和原始 logit，可能造成不可控的全局重分配。可考虑质量守恒/目标 attention mass 校准，而不是固定 logit 常数。
3. **ranking 只测“看目标”而非“用目标判断指令一致性”**：当前 raw attention mass/Borda 更容易找到 object-localization heads；这些 head 不一定对 reward label 有因果贡献。需要 instruction-conditioned、contrastive 或 causal head ranking，并用独立 ranking 集选择。
4. **all-query 干预污染生成/推理过程**：对所有 query 持续加 bias 可能破坏 rubric、语言和答案 token 的信息流。需要 query gating（仅任务 token、决策前 query 或 attention sink 排除）以及渐进/层选择干预。
5. **时序证据与目标证据被混合**：8 帧模型可能主要按时间进展判断；目标区域增强既可能提升 instruction grounding，也可能压制场景/状态变化证据。需要分离 object identity、object state、temporal progress 三类 head 或按帧加权。
6. **自回归位置与 prompt 顺序导致表面差异**：causal mask 使视觉 token 与最终答案 query 的可达性、距离和语言前缀不同；interleaved 的 k=32 成功表明输入布局重要，但固定强 steering 仍非充分条件。

## 2. 相关方法调研与对当前问题的映射

本节只把可从论文主页、arXiv 元数据/正文或官方代码核验的内容当作证据。2026 年论文均仍按其当前公开版本描述，不把预印本结论写成已独立复现事实。

### 2.1 核心方法谱系

| 方法 | 经核验的核心机制 | 对当前研究的直接启示 |
| --- | --- | --- |
| [PASTA / Tell Your Model Where to Attend](https://arxiv.org/abs/2311.02262)（ICLR 2024） | 先在开发任务上逐 head profiling，再只对少量稳定有效的 head 重加权；官方默认 `alpha=0.01, scale_position=exclude`，即压低非强调 token，而非盲目提升全部 head。 | ranking 应从“attention 大”升级为“对任务输出有因果/开发集效用”；单边抑制和 head 稀疏性是必要消融。`alpha=0.01` 等价于对非目标未归一化 attention 乘 0.01（logit 约 `-4.61`），远弱于当前目标 `+6`、非目标 `-6` 形成的 12-logit 差。 |
| [PAI / Paying More Attention to Image](https://arxiv.org/abs/2407.21771)（ECCV 2024） | 发现 text inertia；按生成 token 的 image attention 选择 head/layer并增强视觉 attention，同时用纯文本分支做 CFG 式 logit 修正。 | 仅增强视觉并不等于增强“指令—目标一致性”；可借鉴对比抵消纯视觉进度先验，但需避免额外双分支成本与标签泄漏。 |
| [Visual Attention Redistribution (VAR)](https://arxiv.org/abs/2503.03321) | 识别高 attention、低贡献的 visual sink token；先选 image-centric heads，再把 sink 的一部分 attention budget 重分配给非 sink 视觉 token。 | 比固定 `±bias` 更优雅的方向是**质量守恒重分配**：只回收无效预算，不无限制造 target mass；同时需要排除 sink/低贡献 head。 |
| [Localization Heads](https://arxiv.org/abs/2503.06287) | 在冻结 LVLM 中以语义响应与低空间熵联合筛选少数 localization heads；论文报告极少 head 即可做 training-free grounding。 | 当前 raw bbox mass 主要找到“在哪里”的 head，不保证找到“是否按指令正确执行”的 head。可把低熵、对 query 变化的重定位能力与 reward 因果效用纳入 ranking。 |
| [Gaze Heads](https://arxiv.org/abs/2606.14703) | 用受控多区域 query 的**时序重定位相关性**发现少数 gaze heads；干预 top-100（少于 9% heads）可定向描述，而 random heads 无效、all-head 会破坏生成。论文入口的默认 recipe 是 prefill+decode（CLI `decode_only=False`）；另提供 decode-only 选项，并做 top-k、强度和对照消融。强度消融显示 δ=1 只部分移动 softmax，δ=10 后接近硬干预饱和。 | 与本项目最接近，但它控制“描述哪个区域”而非“输出奖励”。本项目应把 full/decode/prefill/last-query 拆开定位效应来源，并探索低于现有 6 的剂量；同时必须保留 wrong-region、low-rank 和 k-sweep，不能把 attention 转移直接当作任务提升。 |
| [ASCD](https://arxiv.org/abs/2506.14766) | 在解码中做正/负 attention steering：离线找文本偏置 head，并动态识别关键视觉 token；结合对比解码缓解幻觉。 | 支持“head 角色不同、正负干预对象不能对称粗暴处理”；但其目标是 hallucination，不可直接假定 reward 任务同样有效。 |
| [CAST](https://arxiv.org/abs/2605.04641) | 用 caption-query 对 attention head 做 probe，选 caption-sensitive heads，并在其输出方向上做 steering。 | 提供 instruction-conditioned head profiling 模板：用相同视觉、不同指令/问法的对比响应选 head，比 success-only raw mass 更贴合本任务。 |
| [CHASE / EmoMM](https://arxiv.org/abs/2605.01024) | 先检测模态冲突，再只对 conflict-critical heads 做 inter-/intra-modality bias。 | 干预应输入自适应：当视频和指令证据冲突时增强绑定，证据一致时少动，理论上可缓解 suc/fail 权衡；router 必须仅依赖模型内部无标签信号。 |
| [HAS](https://arxiv.org/abs/2607.17994) | 将连续、有界、平滑的 frame highlight 分布取 log 后作为 attention-logit bias，保留非关键帧而非硬删除。 | 对“temporal prior”假设最直接：将 8 帧统一强干预改为 query-conditioned 连续时序权重，避免所有帧 target token 等权。 |
| [Attention is Case-Sensitive](https://arxiv.org/abs/2608.03711) | 大小写能稳定改变 LLM/VLM attention，但 attention 集中与下游准确率并不单调一致，甚至可降性能。 | 是重要反证边界：attention 改了不代表 reward 判断更好，必须用输出指标、配对对照与因果 head 验证。 |

### 2.2 文献归纳出的候选改良路线

优先级按“现有代码可实现、理论针对性、计算成本”排序：

1. **Query-scope 分解**：分别测试 full、prefill、last-prompt 与 decode，区分“改写多模态表征”和“生成期重读视觉”的作用；Gaze Heads 正式 recipe 使用 full sequence，但其任务和近饱和强度与 reward 分类不同，不能直接照搬。
2. **单边或质量守恒 steering**：先比较 target boost-only、non-target suppress-only 与对称 `±bias`；若需代码增量，再实现以目标 attention mass/可回收 sink mass为预算的归一化重分配。
3. **k-normalized dose**：把强度定义为总干预预算而不是每 head 常数，例如 `b(k)=b_ref*sqrt(k_ref/k)` 的 L2 守恒族；必须用独立 screening split 冻结公式后再跑评测，不能按最终标签逐 k 调参。
4. **Instruction-conditioned causal ranking**：在独立 ranking 数据上，用同视频正确/反事实指令的输出差、attention 重定位、低熵与消融效应组合评分；避免只选 object-localization heads。
5. **Conflict/query adaptive gating**：用无标签内部信号（正确/反事实指令的 margin、视觉—文本 attention 比、预测熵）决定是否和多强地干预；一致样本尽量保持 baseline，冲突样本才增强绑定。
6. **连续 temporal steering**：从 tracking 区域的状态变化、指令相似度或 head attention 得到 8 帧软权重，使用 `log(weight)` 形式注入，避免 all-frame 等权和终帧硬编码。
7. **对比分支抵消进度先验**：借鉴 PAI，以真实指令分支减去 null/反事实指令分支的指令无关分量；该路线成本较高，放在前述单分支方法失败之后。

### 2.3 零新增推理诊断

- 现有 raw-mass top heads 高度集中在中后层（约 19–24 层），与 Gaze Heads/稀疏视觉 head 文献的“局部子集”现象一致，但这只是相关性。
- raw mass 与 excess mass 的 top-8 重合度通常为 5–8/8；raw mass 与 visual enrichment 仅约 1–4/8。说明 ranking 对“绝对看了多少”还是“在视觉预算内偏向目标”非常敏感，不能把 raw ranking 当作唯一真值。
- last-frame 样本的 target/non-target token 中位数约 2/46；all-frame 约 17/367（样本随 bbox 大小波动）。对称 `+6/-6` 作用在数量悬殊的集合上，且 8 帧干预规模约扩大 8 倍。
- 以 RoboReward interleaved k=32 的一个完整诊断样本为例，`all` scope 在每个被选层共作用 6 次（1 prefill + 5 decode），但 `applied_query_rows=786`；绝大多数被改写的是 prefill row，而非仅答案生成。这正是 S1 改成 decode-only 的直接证据。
- Qwen 原 all-frame/interleaved 实验的高 k 缺失不是 OOM，而是强干预后输出退化：常见错误为只生成裸 `1`/`5` 或冗长重复文本，无法解析规定的 `ANSWER: <1-5>`。由于当前异常会中断该样本后续条件，旧结果不同 condition 的 n 不相等，不能做无条件横比。

## 3. 自动研究执行队列

1. 复核 native video 两种输入、GRM 强效果和所有跨模型 ranking/processor diagnostics，构建统一比较表。
2. 阅读并核验 Gaze-Heads、PAI、PASTA 及相邻 activation/attention steering 文献和代码，提炼可直接映射到当前失败机制的方法。
3. 先做零新增推理的诊断：head overlap、层分布、attention mass、target/negative token 数、现有 bias-k 剂量响应、样本缺失与配对 bootstrap。
4. 基于诊断设计最小增量方法；优先使用新增配置和现有可选项。每个方法先在冻结的小型、标签不可见 screening cohort 上跑完整 k=8/32/64 与对照。
5. 仅将通过预设门槛的方法扩展到 846 条全量 cohort；失败不静默重跑，记录原因后调整为新的实验 ID。
6. 达标后做复现、11 类统计谬误扫描、task/pairwise/分布分析，并在本文末尾写最终结论。

## 4. 实验运行记录

> 后续每次运行追加：时间、实验 ID、精确命令、工作目录、GPU、超时/监控范围、退出码、输出文件、异常和验收结果。

### 4.1 Screening S1：query scope 与负区域（启动）

- 启动时间：2026-08-16T18:21:36+08:00
- 工作目录：`/home/dais/workspace/Robo-Dopamine`
- 类型：generic inference；hard timeout 3 小时；监控进程存活、GPU 利用率、`steering.shard-00.jsonl` 行数与文件增长。
- 选择规则：`stable_shard(video_sha256, 16)==0`，不读取 suc/fail 标签，预期约 1/16 冻结 cohort。每个样本仍运行 baseline、target/wrong/low-rank × k=8/32/64。

精确命令：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_17_roboreward_interleaved_decode_all_visual_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_qwen_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_18_qwen_interleaved_decode_all_visual_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_19_roboreward_interleaved_decode_boost_only_screen.yaml --shard-id 0 --num-shards 16
```

启动前检查：新增配置可被 `load_config` 读取；`test_crossmodel_image_sequence.py` 最终为 `5 passed`。初次把探索配置放在冻结目录顶层时，该测试因“正式配置必须恰好 16 个”出现 `1 failed, 4 passed`；已把探索配置移动到 `v2_crossmodel/auto_explore/`，未修改测试或正式配置，复测通过。

#### S1 完成状态与结果

- 完成时间：约 2026-08-16T18:27:00+08:00；三进程均 exit code 0。
- 输出：各自的 `steering.shard-00.jsonl` 与 `steering_manifest.shard-00.json`；每项 35 个样本 × 10 条件 = 350 条，350/350 `status=ok`，无解析失败。
- screening shard 构成：11 suc、24 fail。该构成仅在推理完成后用于评价，没有参与样本选择或干预。

candidate-target 摘要：

| 实验 | k | MAE（baseline→target） | suc acc | fail acc | 三指标同时通过 |
| --- | ---: | --- | --- | --- | --- |
| RoboReward decode + all-visual | 8 | 2.0000→2.0000 | 63.64%→63.64% | 4.17%→4.17% | 否 |
| 同上 | 32 | 2.0000→1.7714 | 63.64%→63.64% | 4.17%→4.17% | 否 |
| 同上 | 64 | 2.0000→1.6857 | 63.64%→63.64% | 4.17%→4.17% | 否 |
| Qwen decode + all-visual | 8 | 2.0857→2.0571 | 100.00%→100.00% | 0.00%→0.00% | 否 |
| 同上 | 32 | 2.0857→2.0286 | 100.00%→90.91% | 0.00%→0.00% | 否 |
| 同上 | 64 | 2.0857→1.9429 | 100.00%→81.82% | 0.00%→4.17% | 否 |
| RoboReward decode + boost-only | 8 | 2.0000→1.9714 | 63.64%→63.64% | 4.17%→4.17% | 否 |
| 同上 | 32 | 2.0000→1.8000 | 63.64%→63.64% | 4.17%→4.17% | 否 |
| 同上 | 64 | 2.0000→1.6857 | 63.64%→63.64% | 4.17%→4.17% | 否 |

解释：decode-only 明显提高输出协议稳定性，且 candidate heads 呈 k-dependent MAE 改善，wrong/low-rank 基本不变，说明干预有 head/region 特异性；但只在后续 decode step 改写视觉 key 的注意力不足以改变多数离散端点预测。下一步用 `last_prompt`，它与 ranking 的最后 prompt query 完全对齐，处于“all-query 过强、decode-only 过弱”的中间位置。

### 4.2 Screening S2：last-prompt 对齐干预（启动）

- 启动时间：2026-08-16T18:28:41+08:00
- 设计：沿用 S1 的 label-blind shard 0/16、ranking、bias 与 all-frame target；唯一变化是 `steering_query_scope=last_prompt`，并比较 `negative_scope=all_visual` 与 RoboReward `none`。
- hard timeout 3 小时；监控同 S1。

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_20_roboreward_interleaved_lastprompt_all_visual_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_qwen_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_21_qwen_interleaved_lastprompt_all_visual_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_22_roboreward_interleaved_lastprompt_boost_only_screen.yaml --shard-id 0 --num-shards 16
```

#### S2 完成状态与结果

- 三进程 exit code 0。RoboReward 两项均 350/350 `ok`；Qwen 写入 322 条，并有 4 个样本在某个 condition 裸输出 `5` 而触发 parser failure，导致该样本后续条件未运行。
- RoboReward 的 candidate-target k=8/32/64 在 MAE、suc/fail accuracy 上与 baseline **完全相同**，boost-only 亦同；last-prompt 对该模型不构成有效控制面。
- Qwen k=8 MAE 2.0857→2.0571，但端点准确率不变；在共同有效样本上 k=32 只有约 -0.0323 MAE，suc/fail accuracy 仍无改善；k=64 无改善。该 scope 既弱又仍可能破坏输出格式。
- 结论：强效应并非来自最后 prompt query，也不是 decode-only 足够产生；主要来源应是多个 prefill query row 的表征级改写。下一轮隔离 `prefill`，并把对称 bias 从 6 降至 2。`±2` 形成 4-logit 目标/非目标差，与 PASTA 默认单边 `log(0.01)≈-4.61` 的量级相近，理论上比现有 12-logit 差更温和。

### 4.3 Screening S3：prefill 来源与温和剂量（启动）

- 启动时间：2026-08-16T18:35:45+08:00
- 设计：RoboReward/Qwen 分别测试 `prefill + bias=2 + all_visual`；第三卡测试 RoboReward `all + bias=2`，直接分离 decode 追加效应。

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_23_roboreward_interleaved_prefill_bias2_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_qwen_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_24_qwen_interleaved_prefill_bias2_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_25_roboreward_interleaved_full_bias2_screen.yaml --shard-id 0 --num-shards 16
```

#### S3 完成状态与结果

- 三进程 exit code 0。RoboReward 两项 350/350 `ok`；Qwen 346 条且 1 个样本裸输出 `5` 后中断。
- RoboReward prefill-only：k=32 MAE 2.0000→1.7429，suc 63.64%→72.73%，但 fail 4.17%→0%；k=64 三项恶化。说明 prefill 表征改写能提高部分 suc，但缺少 decode 期视觉重读时 fail 端不平衡。
- Qwen prefill-only：k=32 MAE 2.0857→1.3714、fail 0→8.33%，但 suc 100→90.91%；k=64同类权衡。
- **RoboReward full+bias2 的 k=32 首次在 screening 同时通过三项方向标准**：MAE 2.0000→1.7429，suc 63.64%→72.73%，fail 4.17%→8.33%。wrong-region k=32 把 suc 降至 54.55%、fail 降至 0%，low-rank 基本无益，具备初步 region/head 特异性。
- 同一配置 k=8 只改善 MAE、端点准确率不动；k=64 则 suc/fail 同时下降。按预先提出的“总干预容量随 k 失配”假设，下一轮冻结 `b(k)=2*sqrt(32/k)`（以已通过的 k32/b2 为锚，保持 head-wise bias 向量 L2 norm），而不是按最终标签逐样本/逐条件调参。

### 4.4 Screening S4：L2 守恒剂量（第一批，启动）

- 启动时间：2026-08-16T18:41:35+08:00
- 每配置只含一个 k，因此每样本为 baseline + target/wrong/low-rank 共 4 条；仍使用相同 label-blind shard。

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_26_roboreward_interleaved_full_l2dose_k8_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_27_roboreward_interleaved_full_l2dose_k64_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_qwen_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_28_qwen_interleaved_full_l2dose_k32_screen.yaml --shard-id 0 --num-shards 16
```

#### S4 完成状态与结果

- 三进程 exit code 0。RoboReward k8/k64 各 140/140 `ok`；Qwen 139 条，1 个裸 `5` parser failure。
- RoboReward k8/b4：MAE 2.0000→1.8857，suc/fail accuracy 完全不变；未通过。
- RoboReward k64/b1.414：MAE 2.0000→2.0571，suc 63.64%→54.55%，fail 4.17%→0%；未通过且方向恶化。
- Qwen k32/b2/full：MAE 2.0857→1.3143，suc 100%→90.91%，fail 0→16.67%；仍为 suc/fail 权衡。
- 结论：L2 剂量守恒不足。raw ranking 的 top8 缺少足够任务因果 head，9–32 区间含关键 head，而 33–64 混入有害 head；必须改进排序准则，而非只缩放总强度。

### 4.5 Screening S5：替代 head ranking（启动）

- 启动时间：2026-08-16T18:46:43+08:00
- `visual_enrichment` 与 `excess_mass` 均为原始独立 ranking 阶段已生成的产物，不读取 evaluation 标签、不重用另一模型 ranking。

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_29_roboreward_interleaved_visual_enrichment_bias2_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_qwen_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_30_qwen_interleaved_visual_enrichment_bias2_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_31_roboreward_interleaved_excess_mass_bias2_screen.yaml --shard-id 0 --num-shards 16
```

#### S5 完成状态与结果

- 三进程 exit code 0，三项均 350/350 `ok`，无格式失败。
- RoboReward visual-enrichment：k=32 MAE 2.0000→1.6286、suc 63.64%→72.73%、fail 4.17%→8.33%，通过；k8 端点不动，k64 suc 下降。
- RoboReward excess-mass：k=32 MAE 2.0000→1.5714、suc 63.64%→72.73%、fail 4.17%→12.50%，通过且优于 visual-enrichment；wrong k32 将 suc/fail 降至 45.45%/0%，low-rank 无相同增益。k8 端点仍不动，k64 suc 下降至 45.45%。
- Qwen visual-enrichment：k8/32/64 都降低 MAE，但 fail 仍为 0%，k32/64 还使 suc 从 100%降至90.91%，不通过。
- 结论：替代 ranking 改善了 k32 的 head/region 特异性，但 top8 信号仍不足、top64 尾部仍有害。下一轮测试只 boost target、保留非目标场景证据，检验 suc 损失是否主要由大范围负 bias 引起。

### 4.6 Screening S6：full-scope boost-only 与 Qwen excess ranking（启动）

- 启动时间：2026-08-16T18:52:56+08:00

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_32_roboreward_interleaved_full_boost_only_bias6_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_qwen_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_33_qwen_interleaved_full_boost_only_bias6_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n robo-dopamine python mydata_bench/run_qwen_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_34_qwen_interleaved_excess_mass_bias2_screen.yaml --shard-id 0 --num-shards 16
```

#### S6 完成状态与结果

- 三进程 exit code 0。RoboReward boost-only 为 350/350 `ok`；Qwen boost-only 写入 313 条（307 `ok`、6 个 `sample_failure`）；Qwen excess-mass 写入 345 条（344 `ok`、1 个 `sample_failure`）。Qwen 的 7 次失败均为裸输出 `5`、不符合 `ANSWER: <1-5>` 协议；由于 runner 在单样本首次异常后不再运行后续条件，各 condition 的 n 不齐。
- 下表严格对每个 condition 与其 baseline 取共同有效 `example_id`；标签仅在推理完成后按 `suc/→5`、`fail/→1` 用于评价。

| 实验 | k | 共同 n（suc/fail） | MAE（baseline→target） | suc acc | fail acc | 三指标同时通过 |
| --- | ---: | ---: | --- | --- | --- | --- |
| RoboReward boost-only/bias6 | 8 | 35（11/24） | 2.0000→1.9143 | 63.64%→63.64% | 4.17%→4.17% | 否 |
| 同上 | 32 | 35（11/24） | 2.0000→1.1429 | 63.64%→63.64% | 4.17%→29.17% | 否（suc 未严格提高） |
| 同上 | 64 | 35（11/24） | 2.0000→1.4286 | 63.64%→45.45% | 4.17%→8.33% | 否 |
| Qwen boost-only/bias6 | 8 | 35（11/24） | 2.0857→1.6000 | 100.00%→90.91% | 0.00%→0.00% | 否 |
| 同上 | 32 | 30（8/22） | 2.1667→0.8333 | 100.00%→87.50% | 0.00%→22.73% | 否 |
| 同上 | 64 | 29（8/21） | 2.1034→1.0345 | 100.00%→62.50% | 0.00%→28.57% | 否 |
| Qwen excess-mass/bias2 | 8 | 35（11/24） | 2.0857→2.0571 | 100.00%→100.00% | 0.00%→0.00% | 否 |
| 同上 | 32 | 34（10/24） | 2.1471→1.5000 | 100.00%→90.00% | 0.00%→0.00% | 否 |
| 同上 | 64 | 34（10/24） | 2.1471→1.5000 | 100.00%→90.00% | 0.00%→4.17% | 否 |

RoboReward boost-only k32 的 region/head 特异性很强：candidate-target MAE 1.1429、fail 29.17%，而 wrong-region 对应 MAE 1.9714、suc 27.27%、fail 4.17%，low-rank 对应 MAE 2.0571、suc 54.55%、fail 4.17%。因此保留非目标视觉信息确实缓解了对称抑制的副作用，但仍不能使 suc **严格**超过 baseline；k8 仍无端点移动，k64 仍混入伤害 suc 的 head。Qwen boost-only 在 k32/k64 大幅提高 fail，却同步降低 suc，wrong-region 在高 k 也能显著改变输出且格式失败较多，表明强 bias 已超出区域特异的安全窗口。Qwen excess-mass/bias2 更稳定但仍无平衡端点增益。

结论：S6 否定“仅删除 non-target 负 bias 即可完成目标”。当前证据进一步支持 **head ranking 失配**：前 8 个 localization-like heads 不足以改变 reward 端点，9–32 中有任务有效 head，33–64 混入 harmful head。后续不再围绕同一 raw/enrichment/excess ranking 做逐 k 剂量搜索，而在独立 34 条 success ranking 集上构建 instruction-conditioned/causal head score；评测 shard 标签不进入 ranking、gating 或干预。

### 4.7 Causal head profiling C1：独立 success 轨迹（启动）

- 方法冻结于推理前：候选池为模型自身 raw-mass、excess-mass、visual-enrichment 各 top64 的并集（RoboReward 89 heads、Qwen 100 heads）；样本为原 ranking 阶段全部 34 条独立 success 轨迹，覆盖 8 个 subset。工具显式拒绝非 `suc/` ID 以及含 `reward/label/expected_reward/native_prediction/progress` 字段的输入。
- 对每个候选 head 单独施加 `bias=6`、`query_scope=all`、`negative_scope=none` 的 all-frame target boost；以 teacher-forced `ANSWER: ` 后 1–5 五个 token 的受限 softmax 计算连续 reward-5 margin。相同 head 再对等大小、互斥 wrong region 干预。核心量为 `target−baseline`（成功轨迹非破坏性/增益）和 `target−wrong`（空间特异性）。
- 预注册排序：先列 `target>0 且 spatial>0` 的 tier 0，再列 `target>=0` 的 tier 1，最后列其余被 profile 候选；tier 内按 task-balanced `(target + spatial)` 降序，以原 mass rank 仅作 tie-break。未被 profile 的原 raw-ranking heads 只追加为 tier 3 尾部，使完整 ranking 恢复到 896 heads，保证 top64 与 64-head low-rank control 可互斥；tier-3 永不进入 candidate top64。
- 该方法不是按 evaluation screening 结果选 head：S1–S6 的 35 条样本、标签与输出均不被 profiling 工具读取。连续 margin 只是独立成功轨迹上的开发信号，最终仍须由冻结 screening shard 的离散 MAE/suc/fail 三指标判定。
- 新增文件：`mydata_bench/auto_explore_causal_rank.py`、两个 `causal_rank_*_interleaved.yaml`。静态 compile 通过；正式 16-config 矩阵兼容测试 `PYTHONPATH=/home/dais/workspace/Robo-Dopamine pytest -q mydata_bench/tests/test_crossmodel_image_sequence.py` 为 `5 passed`。
- smoke：RoboReward raw-top1 × 34 条轨迹，34/34 `ok`，证实 forced-choice forward、target/wrong hook 与 append-only schema 有效；其输出保留为 `causal_profile.head-shard-00-of-89.jsonl`，不删除或并入正式聚合。

正式命令（hard timeout 3 小时/进程；仅 GPU 0、1、2）：

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_rank_roboreward_interleaved.yaml --head-shard-id 0 --num-head-shards 2
CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_rank_roboreward_interleaved.yaml --head-shard-id 1 --num-head-shards 2
CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_rank_qwen_interleaved.yaml --head-shard-id 0 --num-head-shards 2
```

Qwen head shard 1 将在首个 RoboReward shard 自然完成、释放 GPU 后启动；不抢占、不使用 GPU 3。

#### C1 RoboReward 完成与 S7 反证

- RoboReward profile 两分片分别为 1530 与 1496 条，合计 89 heads × 34 success trajectories = 3026，0 invalid。纯因果 artifact 保留为 `causal_ranking.json`：tier0（target>0、spatial>0）33 heads，tier1（target≥0）5 heads，tier2 51 heads；完整 fallback 后共 896 heads。
- 在接触 S7 标签前，由 C1 的“仅 38 个 target 非负 head”这一开发事实构造并冻结 `causal_safe_padding_ranking.json`：前 38 个因果安全 heads 后追加原 raw-mass ranks 807–832 的 26 个未 profile 近惰性 heads；top64 与末端 64-head control 均为 64 个唯一 heads、重合 0。该构造的 fingerprint 为 `18182be1910bee8cec7bb4a5c8c0c93747f1db2b89f30dea09e2c7ee5cc056ec`。
- S7 命令：

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_35_roboreward_interleaved_causal_safe_padding_bias6_screen.yaml --shard-id 0 --num-shards 16
```

- S7 exit code 0，350/350 `ok`。结果是明确反证：

| k | MAE（baseline→target） | suc acc | fail acc | 通过 |
| ---: | --- | --- | --- | --- |
| 8 | 2.0000→2.6286 | 63.64%→81.82% | 4.17%→0.00% | 否 |
| 32 | 2.0000→2.5714 | 63.64%→90.91% | 4.17%→0.00% | 否 |
| 64 | 2.0000→2.5143 | 63.64%→90.91% | 4.17%→0.00% | 否 |

wrong-region k8/32/64 的 suc 仅为 36.36%/27.27%/36.36%，low-rank 基本接近 baseline，说明 C1 heads 确实具有 target/head 特异性；但它们学到的是“看到成功轨迹目标时把 reward-5 margin 推高”，不是“区分完成与未完成”。因此 C1 成功修复了 raw top8 无作用与 top64 有害尾部，却产生更强的 **class-5 oversteering**；不能以 suc 改善冒充总体成功。

### 4.8 Causal temporal profiling C2：完整成功 vs 早期停滞（启动）

不使用 evaluation fail 样本，改为从每条独立成功轨迹生成受控的 early-stalled 输入：保留前两张图（t0、t1），后六个槽位重复 t1，并同步冻结 source indices/target tracking 对齐。它仍有 8 个独立 image spans 和相同指令/目标，但不含终点完成证据。

每个 head 新增 `temporal_margin_delta = Δmargin5(full success target boost) − Δmargin5(early-stalled target boost)`。C2 tier0 要求 full target、spatial、temporal 三者均正且 stalled effect≤0；tier1 要求 full target≥0、spatial>0、temporal>0；tier 内 score 冻结为 `target + spatial + 2×temporal`。若安全 head 少于 64，沿用在评测前构造的最低 raw-mass inert padding。该设计直接检验 temporal-prior 假设，并避免读取真实 fail 标签。

#### Screening ceiling 规则澄清（在 C2 评测前冻结）

label-blind shard 0/16 恰好使 Qwen interleaved baseline 在 11 条 suc 上达到 100% exact accuracy，故该 shard 上 `ΔAcc_suc>0` 在数学上不可能。该小 shard 仍用于剔除明显失败方案，但 Qwen screening 的必要条件改为：MAE 下降、fail accuracy 严格提高、suc accuracy 不下降；最终 846 条 cohort（其 suc baseline 非 100%）仍严格要求三指标同时改善，主线验收标准不变。这不是为了迁就某个候选结果，而是预先处理离散指标的 ceiling effect；RoboReward screening 仍要求三项严格改善。

#### 关键完整性纠正：ranking 视频与 846 cohort 并不独立

文件级交叉核验发现，原先沿用既有 manifest 命名而称为“independent ranking”的 34 个 success `example_id` **全部**包含在 846 evaluation cohort 中，且 34/34 video SHA-256 也重合。此前 ranking 只使用 success 轨迹的无 fail 输出 attention mass，未直接读取 evaluation 输出，但视频级独立性陈述不成立；必须纠正，不能继续把最终 846 结果称为完全 held-out confirmatory。

从此将这 34 个视频簇正式重分类为 ranking/development clusters；反复用于 S1–S8 的 shard0/16 含 11 个不同视频簇，与 ranking 34 簇重合为 0，也一并归为 screening/development。冻结的 confirmatory 集定义为排除这 45 个视频簇后的 697 条记录（223 suc、474 fail、254 个视频簇）。未来：

- 697 条 held-out 集承担无泄漏主结论与 bootstrap；
- 846 全量只作 transparent exploratory/descriptive 复算，不伪称独立验证；
- ranking 34 簇可合法用于开发 supervised/contrastive head profiler，但绝不回流到 697 held-out 排序或 gating；
- 最终报告同时给出开发簇、screening 簇和 held-out 簇 SHA/ID manifest，使该纠正可复核。

#### C2 完成：early-stalled proxy 未能代表真实 fail

- RoboReward：89 个候选 heads × 34 条 development success，共 3026 条，0 invalid；严格 tier0 14 个、tier1 10 个。安全 padding 后 top64 为 24 个 causal-safe heads + 40 个最低 raw-mass 未 profile heads，fingerprint `52230f99fdc440790b4354e9d6d23bca8c62ddddab3e8bed091cf3d4cd8f4b93`。
- Qwen：100×34=3400 条，0 invalid；tier0 16 个、tier1 9 个。安全 padding 后 top64 为 25+39，fingerprint `8dcd831fb5115cdae4b2e197fa73636e9804ffb4c710912a30e2df6c72bec906`。
- artifact 中早期写入的 `independent_success_*` 字段是纠正视频重合前的遗留描述；以本节上方冻结分区为准，这 34 个视频簇均属 development，不能称独立 held-out。

正式 profile/聚合命令为：

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_temporal_rank_roboreward_interleaved.yaml --head-shard-id 0 --num-head-shards 2
CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_temporal_rank_roboreward_interleaved.yaml --head-shard-id 1 --num-head-shards 2
CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_temporal_rank_qwen_interleaved.yaml --head-shard-id 0 --num-head-shards 2
CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_temporal_rank_qwen_interleaved.yaml --head-shard-id 1 --num-head-shards 2
```

### 4.9 Screening S8：C2 temporal causal ranking

RoboReward 依次冻结测试 boost-only bias=6 与 bias=2；两项均 350/350 `ok`，exit code 0：

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_36_roboreward_interleaved_causal_temporal_bias6_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_37_roboreward_interleaved_causal_temporal_bias2_screen.yaml --shard-id 0 --num-shards 16
```

| dose | k | MAE（baseline 2.0000→target） | suc acc（63.64%→） | fail acc（4.17%→） | 通过 |
| --- | ---: | ---: | ---: | ---: | --- |
| boost-only 6 | 8 | 2.2000 | 72.73% | 0.00% | 否 |
| 同上 | 32 | 2.4000 | 90.91% | 0.00% | 否 |
| 同上 | 64 | 2.3429 | 81.82% | 0.00% | 否 |
| boost-only 2 | 8 | 2.0000 | 63.64% | 4.17% | 否（完全不变） |
| 同上 | 32 | 2.2286 | 72.73% | 0.00% | 否 |
| 同上 | 64 | 2.2286 | 72.73% | 0.00% | 否 |

因此 early-stalled 的合成对照虽能发现“完整成功比早期停滞更推向 5”的 heads，却仍没有约束这些 heads 在真实 wrong-instruction fail 上朝 1 移动；它仍是更复杂的 class-5 profiler。RoboReward 已构成明确反证，故没有再把同一机制投入 Qwen screening 预算。

### 4.10 Causal contrastive profiling C3：同视频 success/fail 开发对

在已重分类为 development 的 34 个 ranking 视频簇内，从 cohort 为每个视频确定性选取 lexicographically first fail；32/34 有 fail counterpart，得到 32 个同视频 success/fail pairs。标签只用于离线定义 teacher-forced 正确端点 margin：success 奖励 target boost 朝 5，fail 奖励 target boost 朝 1，并同时要求 target 优于 equal-size wrong-region。标签不进入推理时 gating，且这 32 个视频簇均排除于冻结 697 held-out。

RoboReward fail profile 为 89×32=2848 条，0 invalid；得到 10 个严格双向安全 heads，补入 54 个最低 raw-mass heads。ranking fingerprint：`b9fe3301a90f2b63ebeec53b71ec7a262d9e688517417ccc8cd8825ab092488d`。top1 `(layer=19, head=10)` 的 success correct-margin delta +0.6438、fail correct-margin delta +0.2208、paired spatial +0.9708。

Qwen fail profile 为 100×32=3200 条，0 invalid；得到 24 个严格安全 heads，补入 40 个最低 raw-mass heads。ranking fingerprint：`bd4b1e8c9d9f735eb99df30db3a6275f559379c31ae58845f0e00a9e2d8ca6ca`。top1 `(layer=19, head=17)` 的 success/fail correct-margin delta 分别 +1.2539/+0.9664，paired spatial +1.4023。

正式命令：

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_contrastive_rank_roboreward_interleaved.yaml --head-shard-id 0 --num-head-shards 2
CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_contrastive_rank_roboreward_interleaved.yaml --head-shard-id 1 --num-head-shards 2
conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank aggregate-contrastive --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_contrastive_rank_roboreward_interleaved.yaml --num-head-shards 2
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_contrastive_rank_qwen_interleaved.yaml --head-shard-id 0 --num-head-shards 2
CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_contrastive_rank_qwen_interleaved.yaml --head-shard-id 1 --num-head-shards 2
conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank aggregate-contrastive --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_contrastive_rank_qwen_interleaved.yaml --num-head-shards 2
```

### 4.11 Screenings S9–S10：RoboReward contrastive ranking

S9 为 boost-only bias=6，S10 为对称 target +2 / 其他 visual −2；两项均 350/350 `ok`，exit code 0：

```bash
CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_38_roboreward_interleaved_contrastive_causal_bias6_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_39_roboreward_interleaved_contrastive_causal_symmetric_bias2_screen.yaml --shard-id 0 --num-shards 16
```

| 方法 | k | MAE（baseline 2.0000→target） | suc acc（63.64%→） | fail acc（4.17%→） | 通过 |
| --- | ---: | ---: | ---: | ---: | --- |
| boost-only 6 | 8 | 1.6571 | 72.73% | 0.00% | 否 |
| 同上 | 32 | 1.7429 | 72.73% | 0.00% | 否 |
| 同上 | 64 | 1.4857 | 72.73% | 4.17% | 否 |
| symmetric 2 | 8 | 1.8286 | 72.73% | 4.17% | 否 |
| 同上 | 32 | 1.7143 | 63.64% | 4.17% | 否 |
| 同上 | 64 | 1.5429 | 63.64% | 4.17% | 否 |

C3 首次使 RoboReward 三个 k 都稳定降低 MAE，且不再出现 success-only causal ranking 的严重 5-oversteering；但它没有让 fail exact accuracy 严格超过 baseline。S10 的预测分布显示 target k8/32/64 分别把若干 baseline 3/5 移到 2/4，因此连续 ordinal 判断在改善，只有一条 baseline fail 已为 1，其他 fail 尚未跨越 exact=1 边界。wrong-region 与 target 在 MAE 上仍有一定同向变化，提示“对 target 的局部强化”尚未完全等同于“完成状态识别”。不继续盲扫 uniform bias；后续若需要改进应让每个 head 的方向/权重显式继承 paired fail margin，而不是用输出端点调剂量。

### 4.12 Screening S11：Qwen contrastive ranking（启动）

在读取任一 S11 输出前同时冻结 boost-only bias=6 与 symmetric bias=2 两个敏感性族；均使用相同 Qwen ranking `bd4b…ca6ca`、label-blind shard0/16、top-k 8/32/64 及 target/wrong/low-rank 对照。按前述 ceiling 规则，screening gate 为 MAE 严格降低、fail exact 严格提高、suc exact 不下降；最终 held-out gate 不变。

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_qwen_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_40_qwen_interleaved_contrastive_causal_bias6_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_qwen_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_41_qwen_interleaved_contrastive_causal_symmetric_bias2_screen.yaml --shard-id 0 --num-shards 16
```

- 启动状态：配置解析及正式矩阵兼容测试通过，`5 passed`；每进程 hard timeout 3 小时，仅使用 GPU 0/1。

#### S11 完成与 S12 冻结

- boost-only bias=6：230 `ok`、15 `sample_failure`。所有 35 条 baseline 与 target-k8 有效，但 15 条在 wrong-k8 生成裸 `5` 而非 `ANSWER: 5`，循环按样本停止，故 k32/k64 只有偏置后的 20 条，不能作完整比较。完整 k8 为 MAE 2.0857→1.5143、suc 100.00%→90.91%、fail 0.00%→12.50%，仍不通过 ceiling gate。
- symmetric bias=2：330 `ok`、4 `sample_failure`，同为裸 `5` parser failure。target k8 为 MAE 2.0857→1.8857、suc 100.00%→90.91%、fail 0.00%→0.00%；k32/k64 也有缺失且 fail 仍为 0，不通过。
- 两个进程均 exit code 0；failure 是样本级严格输出协议失效，不是进程 crash。未使用 `--retry-failed`，也未把残缺条件当作通过。

S12 只取一个由稳定性边界决定的中点：同一 contrastive ranking、boost-only bias=4。它不是继续查看多档标签结果后择优，而是在 bias=6 已出现 15/35 格式失效、bias=2 级干预稳定但不足之间固定的唯一中间剂量；若仍失败，停止 uniform dose 路线。

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_qwen_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_42_qwen_interleaved_contrastive_causal_bias4_screen.yaml --shard-id 0 --num-shards 16
```

#### S12 完成：停止 uniform dose 路线

S12 exit code 0，但只有 245 `ok`、15 `sample_failure`；与 bias=6 相同，均为模型输出裸 `5` 导致严格 parser 拒绝。完整的 target-k8 为 MAE 2.0857→1.8286、suc 100.00%→90.91%、fail 0.00%→0.00%；k32/k64 分别仅余 23/22 条非随机子集，不能评价。bias=4 既未跨越 fail 端点，也未解决协议稳定性，因此按预先规则终止 Qwen uniform dose 调参。

### 4.13 Causal bidirectional profiling C4：逐 head 有符号 steering

C3 只保留 `+6` 同时提高 development success/fail 正确端点 margin 的 heads，但完整候选中还有一类 heads 在 `+6` 下对两侧 margin 都为负。若因果响应在反方向成立，这些 heads 应使用负而非正 bias；直接假设线性反号并不充分，故 C4 对它们显式重跑 `−6` target/wrong profile，再按完全相同的 paired correct-margin 与 spatial-specificity gate 判定。

- 预筛只使用 C3 development artifact：RoboReward 22 个、Qwen 20 个 `+6` 双侧 target effect 均负的 heads；不读取 screening/held-out 输出。
- 每个候选分别在 34 success 与同视频 32 fail development 输入上用 `−6` 重跑；标签不 model-facing。
- 新增通用可选项 `ranking_bias_field` 与逐 head `steering_multiplier`；共享 hook 可在同层同时对不同 heads 施加 `+b/-b`。未配置该字段的所有旧实验行为不变。
- 单元 smoke 在真实 conda/PyTorch 环境验证：同层两个 heads 的 selected/other logits 分别得到 `(+6,−6)` 与 `(−6,+6)`；正式 16-config 测试仍 `5 passed`（额外探索测试在系统环境因无 torch 被显式 skip）。

RoboReward 负方向 profile 命令与完成状态：

```bash
CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_signed_negative_success_roboreward_interleaved.yaml --head-shard-id 0 --num-head-shards 1
CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_signed_negative_fail_roboreward_interleaved.yaml --head-shard-id 0 --num-head-shards 1
```

两进程均 exit code 0；success 748/748、fail 704/704 `ok`。与原 10 个正向安全 heads 合并后，4/22 个负向候选通过显式 `−6` 双侧 gate，总安全集 14（10 positive、4 negative），补 50 个最低 raw-mass heads。有效且 head 唯一的 v2 fingerprint 为 `0110ad041329967613cc3815293082130a34b3153d5991eedf415a4c43dc02aa`。第一次聚合产物目录无意保留了正/负 profile 的同一 head 两个方向，导致 918 行而非 896 行；该 artifact 明确判 invalid 且从未用于推理，不删除、不覆盖。修正为按 `(layer,head)` 只留 safety tier/score 更优方向并写入新增 `_v2` 目录，896/896 heads 唯一。

有效 top8 中 rank5 `(20,28)` 使用 `−6`，其余七个使用 `+6`；所有方向均来自 development 实测，而非 evaluation sample gating。

Qwen 负方向 profile 与 RoboReward S13 同时启动：

```bash
CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_signed_negative_success_qwen_interleaved.yaml --head-shard-id 0 --num-head-shards 1
CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_signed_negative_fail_qwen_interleaved.yaml --head-shard-id 0 --num-head-shards 1
```

### 4.14 Screening S13：RoboReward bidirectional causal（启动）

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_43_roboreward_interleaved_bidirectional_causal_bias6_screen.yaml --shard-id 0 --num-shards 16
```

冻结设置：逐 head multiplier × base bias=6、boost-only、all-query、all-frame，top-k 8/32/64 与 target/wrong/low-rank 对照；不使用逐样本标签 gating。预期 350 条，hard timeout 3 小时。

#### S13 结果与 C4 强度加权

S13 350/350 `ok`、exit code 0。k8/k32/k64 的 MAE 分别 1.6571/1.6857/1.4857，suc 均由 63.64%→72.73%，但 fail 为 0/0/4.17%，仍未严格改善。说明有符号方向本身不足。

随后完全由 development causal artifact 定义强度：每个安全 head 的
`abs(multiplier)=sqrt((min(success_correct, fail_correct)+max(paired_spatial,0))/max_strength)`，下限 0.1；符号保留 C4 实测方向，最低 raw-mass padding 也使用非零 0.1。该规则强调双侧最弱效应，防止单侧极强 head 支配；fingerprint `fa4fc656aa669301d14259cfa42ad3552727c59f26293ad4668c682d57a1ab0b`。

### 4.15 Screenings S14–S16：weighted bidirectional dose bracket

命令：

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_44_roboreward_interleaved_bidirectional_weighted_bias6_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_46_roboreward_interleaved_bidirectional_weighted_bias10_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_47_roboreward_interleaved_bidirectional_weighted_bias8_screen.yaml --shard-id 0 --num-shards 16
```

三项均 350/350 `ok`、exit code 0。δ=6 是安全下界，δ=10 是 Gaze Heads 强度消融给出的近饱和文献上界；二者在 k8 的 suc/fail 离散边界相反，故只取一次中点 δ=8，不继续密集 sweep。

| base δ | k | MAE（2.0000→） | suc acc（63.64%→） | fail acc（4.17%→） | 三项通过 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 6 | 8/32/64 | 1.7714 / 1.7714 / 1.7714 | 72.73% / 72.73% / 72.73% | 4.17% / 4.17% / 4.17% | 否（fail 持平） |
| 10 | 8 | 1.5429 | 63.64% | 8.33% | 否（suc 持平） |
| 10 | 32/64 | 1.5143 / 1.5143 | 72.73% / 72.73% | 8.33% / 8.33% | 是 |
| **8** | **8** | **1.6000** | **72.73%** | **8.33%** | **是** |
| **8** | **32/64** | **1.6571 / 1.6571** | **72.73% / 72.73%** | **8.33% / 8.33%** | **是** |

S16 wrong-region k8/32/64 都为 MAE 2.0000、suc 54.55%、fail 4.17%，不复现三项改善；low-rank MAE 2.0571、suc/fail 与 baseline 相同。故该 screening success 同时具备 region/head specificity，并首次让 RoboReward interleaved 的全部 k 通过预设方向门槛。

Qwen 同公式的 bidirectional 结果：负方向 success 680/680、fail 640/640 `ok`；正/负安全 heads 为 24+10=34，补 30，v2 fingerprint `9b678e153d43fba5cc5b997adafb4d37de1632d82c9c4b714b79acc581280fe4`，weighted fingerprint `553600f984a91febe24c95ad5713436c3de6485986a522f7257dbd9240e4858c`。但 weighted δ=6 screening 只有 324 `ok`、4 个裸 `5` failure；完整 k8 MAE 2.0857→1.6286、suc 100→90.91%、fail 0→0，不通过。Qwen uniform/signed/weighted 单路 steering 均反复证实同一 ceiling/trade-off，后续必须转向无标签内部 conflict gating，不能继续 dose sweep。

### 4.16 Confirmatory H1：RoboReward interleaved S16 冻结扩展

新增确定性分区工具仅按 `video_sha256` 选择：排除 34 ranking/development clusters 与 `stable_shard(video_sha256,16)==0` 的 11 screening clusters；二者重合 0。产物：

- 697 records（223 suc、474 fail，仅为 selection 后评价统计）；
- 254 held-out video clusters；
- partition fingerprint `4d65d16add4129b3afc0b16f24495189d47580e5688b7167b4921c0cd528fee2`；
- held-out ID fingerprint `22875ed9573ab35a65fbaa3cc3c6dfb693994fbe9ac96f85090cecd4b70573cd`。

冻结配置：`attention_48_roboreward_interleaved_bidirectional_weighted_bias8_heldout.yaml`，ranking/dose/query/negative scope 均与 S16 完全相同，只把 label-blind ID manifest 改为 697 held-out。三分片命令：

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_48_roboreward_interleaved_bidirectional_weighted_bias8_heldout.yaml --shard-id 0 --num-shards 3
CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_48_roboreward_interleaved_bidirectional_weighted_bias8_heldout.yaml --shard-id 1 --num-shards 3
CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_48_roboreward_interleaved_bidirectional_weighted_bias8_heldout.yaml --shard-id 2 --num-shards 3
```

H1 是 confirmatory gate：不以其结果回调该 layout 的 ranking、weights 或 δ。

### 4.17 第二输入布局：`text→images` layout-specific profiling（启动）

不复用 interleaved heads。分别从 RoboReward `attention_09_*` 与 Qwen `attention_11_*` 的 raw/excess/enrichment 各 top64 并集构造 layout-specific candidate pools（93/100 heads），沿用相同 34/32 development video pairs 做正向 paired causal profiling。该工作与 H1 并行，不读取 H1 输出。

正向 profile 已完成且无缺失：RoboReward 为 `93×34=3162` 行，Qwen 为 `100×34=3400` 行，均 exit code 0。聚合得到 layout-specific `causal_ranking.json`：

- RoboReward：tier0 33、tier1 5，fingerprint `33a7e321fcc9884e052eb5ee7a68847d56aae7d2768010ca654b2ddf7aeb31a2`；
- Qwen：tier0 32、tier1 7，fingerprint `f6439b07d7914c384c0bb9b563acfb66c49882338cd012a90fc3cfcb9545a772`。

这些正向 heads 仅证明完成轨迹上的 reward-5 空间响应，不能单独作为平衡端点证据；下一步仍按冻结计划对同视频 development fail 做 contrastive profile，再构造该布局专属的双侧/有符号排序。

### 4.18 Qwen interleaved：无标签 conflict-adaptive router

单一路径在 success/fail 上反复出现相反效应，因此把问题改写为“先用模型内部空间冲突证据选择冻结分支，再 steering”，而不是用标签选择分支。开发 artifact `adaptive_conflict_ranking.json` 完全来自 32 个 development 视频 pair：

- success-specific 安全 heads 37 个，fail-specific 安全 heads 51 个；
- 8 维 probe 对每个冻结 probe head 分别施加显式有符号 bias6，特征为 `abs(weight)×[reward5_margin(target)−reward5_margin(wrong)]`；
- ridge 仅在 development pair 上训练，leave-video-out overall/success/fail accuracy 为 93.75%/90.625%/96.875%；
- 保守阈值 `−1.0549863781025324`，OOF success 低于阈值 0/32、fail 8/32；
- artifact fingerprint `ec33c08d88fc6a821df46a20e09999afd738e0d4dd1251253a3d9cc4ecfb8ca9`，明确记录 `inference_uses_labels=false`。

新增的 runtime 在每个样本生成前只计算一次 probe，所有 k、candidate/wrong/low-rank 对照共享同一 branch；记录保留原始/标准化特征、score、阈值、分支及逐 head logits/hook diagnostics。旧配置默认关闭该路径，正式 16-config 矩阵仍为 `5 passed`；真实 conda/PyTorch 的冻结 ridge 与 signed-hook smoke 通过。筛选配置冻结为 `attention_49_qwen_interleaved_adaptive_conflict_bias6_screen.yaml`，仍使用 label-blind shard0/16、top-k 8/32/64、base bias6。

### 4.19 H1 结果：RoboReward interleaved 无泄漏确认性通过

三个分片均 exit code 0；验证器确认 697 个 example、254 个视频簇、10 个条件，共 `6970/6970` 有效记录，零缺失/invalid。验证 artifact fingerprint 为 `cf26ba9cbcb1e85ff275de98da0fa2bc43c88ee9703a129394f1186292678`。

| k | MAE（baseline→target） | suc accuracy | fail accuracy | 三项方向 | cluster bootstrap 95% CI（MAE / suc / fail） |
| ---: | --- | --- | --- | --- | --- |
| 8 | 1.5968→1.3974 | 44.39%→55.16% | 12.87%→18.57% | 全通过 | [−0.2730,−0.1683] / [+6.28,+15.70]pp / [+2.96,+7.25]pp |
| 32 | 1.5968→1.4103 | 44.39%→54.71% | 12.87%→17.72% | 全通过 | [−0.2592,−0.1545] / [+5.83,+15.25]pp / [+2.31,+6.26]pp |
| 64 | 1.5968→1.3945 | 44.39%→54.26% | 12.87%→18.35% | 全通过 | [−0.2772,−0.1699] / [+5.38,+14.35]pp / [+2.83,+7.18]pp |

9 个预注册 sign-flip tests 经 Holm 校正后均为 `p=0.00089991`；对应 McNemar p 值也均小于 `2×10⁻4`。因此不是仅由总体均值或单一 k 驱动。pairwise 正差率由 baseline 60.82% 提升至 k8/k32/k64 的 67.43%/65.60%/66.51%。

干预特异性成立：wrong-region 的总体 MAE 为 1.6155/1.6155/1.6126（均不优于 baseline），且 fail accuracy 降到 11.39%/11.18%/10.76%；low-rank 总体 MAE 为 1.6011/1.6055/1.6069，近似惰性且不复现三项改善。预测分布显示 target steering 同时把 fail 的 label-1 数从 61 提高至 84–88，并把 suc 的 label-5 数从 99 提高至 121–123，而不是单向把全体推向同一端点。

task 层面存在真实异质性：大多数任务改善，但 `task3_10/task3_4/task3_7/task5_8` 等小 strata 的 MAE 可持平或变差，故结论严格限定为“冻结总体 held-out cohort 与两类端点均稳定改善”，不声称每个 task 都改善。

#### H1 的 11 类统计谬误扫描（11/11 checked）

1. Simpson：总体方向未被主要 strata 整体反转，但上述 task 异质性保留为 caution；不作逐 task 普遍化。
2. Ecological：推断单位为 example，并以 video cluster 重采样；未从 task 聚合量推断单条样本。
3. Berkson：auto-valid-grounding 是选择性 cohort，外部可推广性受限；配对的内部干预比较未再按输出筛样本。
4. Collider：grounding eligibility 可能关联任务难度与可定位性，故不外推到未通过 grounding 的视频；H1 内没有按 baseline/target 结果控制 collider。
5. Base-rate neglect：明确报告 223 suc、474 fail 及分层准确率，未以总体准确率替代类别结果。
6. Regression-to-mean：held-out 未按极端 baseline 分数选择，且每条样本有同输入 paired baseline。
7. Survivorship：6970/6970 完整，无 dropout/只保留成功生成的问题。
8. Look-elsewhere：H1 配置在查看 H1 输出前由 S16 冻结；9 个主检验执行 Holm，而此前 S1–S16 明确标注 development exploration。
9. Garden of forking paths：开发路径很多但全记录；H1 partition/ranking/dose/control 在启动前冻结，H1 不回调参数。
10. Correlation≠causation：这是模型内配对干预，可支持“该 hook 在该 cohort 改变输出”的因果描述；不外推到现实机器人成功率。
11. Reverse causality：steering 在 generation 前施加且 baseline/target 共用输入，不存在输出反向决定干预；标签只在推理后 join。

H1 的 Verification Status 为 `ANALYZED`（完整配对统计与确定性 artifact 已验证；不通过重复推理冒充独立复现）。这已完成 RoboReward 的第一个输入构造，但主线仍要求该模型第二构造及 Qwen 两个构造，故研究继续。

### 4.20 `text→images` paired contrastive profiling

development fail profiles 完整结束：RoboReward `93×32=2976`、Qwen `100×32=3200`，均为 0 invalid、exit code 0。与同视频 success profile 聚合后：

- RoboReward：正方向双侧安全 heads 18 个，contrastive fingerprint `55f556c04bf3c48cdf2da63c71babac0ab25c7f9c2fea0fb9597951426ba8336`；另有 21 个 heads 在 `+6` 下 success/fail correct-margin 均为负，进入显式 `−6` 候选。
- Qwen：正方向双侧安全 heads 20 个，contrastive fingerprint `86dbdfc2e71397643cc3267240226e76dcbad68341e1e7571f03447c0d58f22f`；另有 29 个双负候选。

这些数量与 interleaved 明显不同，再次说明 head 因果作用依赖 input construction，不能跨布局复用。当前分别对 21/29 个候选的 success 与 fail development pair 显式重跑负方向；不依据代数反号推断有效性。

### 4.21 Screening S17：Qwen adaptive gate collapse（反证）

S17 进程 exit code 0，但仅 29/35 样本具有完整 10 条条件；另 6 个样本因 checkpoint 生成裸 `5` 而触发既有协议 parser 的 `No documented 'ANSWER: <1-5>'`，不自动重试、不把裸 token 静默转成有效分数。完整子集中 9 suc/20 fail 全部被路由到 success branch：suc/fail router score 范围分别为 `[−0.343,0.977]` 与 `[−0.838,1.445]`，全部高于冻结阈值 `−1.055`。因此 development 上 fail 8/32 低阈值的路由行为未迁移到 screening shard。

| k | MAE（1.9655→） | suc acc（100%→） | fail acc（0%→） | 通过 |
| ---: | ---: | ---: | ---: | --- |
| 8 | 1.8966 | 88.89% | 0% | 否 |
| 32 | 1.8276 | 88.89% | 0% | 否 |
| 64 | 1.8276 | 88.89% | 0% | 否 |

不能用 S17 的评测标签移动 threshold，否则会把 confirmatory gating 退化成 endpoint hard coding。故冻结 S17 为 router transport failure，不重试、不调阈值。

### 4.22 Qwen complementary balanced portfolio（启动）

为避免逐样本 gating，改为完全冻结的全局组合：只用 development causal artifact 中 success-specialist 与 fail-specialist 的显式方向，解析求每一对的非负权重，使该 pair 的 **平均 success/fail correct target-margin effect 相等**；同时要求组合后的两侧 target 与 spatial effect 全为正。按 `min(target)+0.5×min(spatial)` 选择 4 个互不重复 pair，pair 间再按开发强度平方根缩放，组成 top8；rank9–64 仅追加 0.1 的低 raw-mass padding。因此 k8/32/64 的实质因果组合相同，不依赖样本标签、预测端点或 S17 router score。

artifact fingerprint 为 `d9f7bdded3f5aa101a79e8f5ffa76cc567d2cb1aa5424f259a4e7d5ef5486635`；top8 为 `(21,26)+(19,28)`、`(20,13)+(21,19)`、`(19,14)+(20,29)`、`(20,23)+(20,15)` 四对。所有选择与权重来自 development margin；S17 只提供“放弃不迁移的 gating”这一方法级反证，没有参与具体 head/weight 求解。S18 配置 `attention_50_qwen_interleaved_balanced_portfolio_bias6_screen.yaml` 已启动。

S18 exit code 0；1 个裸 `5` 协议 invalid，34 个完整样本中 MAE 在各 k 下降，但 suc 100%→90%、fail 仍 0%，故 complementary global portfolio 也不通过。它说明 development 平均 margin 的线性相消不能保证自由生成的离散端点相消。

### 4.23 第二布局 signed/weighted 聚合与 screenings

显式负方向 success/fail profile 全部完整：RoboReward `21×34 + 21×32`，Qwen `29×34 + 29×32`，0 invalid、exit code 0。

- RoboReward `text→images`：27 个安全 heads（18 positive、9 negative），signed fingerprint `356b0f803db864bfeebb27717fd0635770c560c4875fb5ec91e08074c132b325`，weighted fingerprint `ade55e6f697a51e4603868a60a7771b6651e0b752cb2883ff48d445312afd516`。
- Qwen `text→images`：33 个安全 heads（20 positive、13 negative），signed fingerprint `46280fa9954c18a26b3b53f965acd91c7f1608a217036059eac0a3ca020234fe`，weighted fingerprint `475fa1e3eaa85f213e83ddc2becaf7e00e821d63ee43d921051e71fec7cc16fc`。

RoboReward δ6（S19）350/350 有效：k8/32/64 的 MAE 为 1.5143/1.5143/1.6000（baseline 2.0571），suc 均 45.45%→63.64%，但 fail 均 0%，未通过。δ8（S20）350/350 有效：

| k | MAE（2.0571→） | suc acc（45.45%→） | fail acc（0%→） | 通过 |
| ---: | ---: | ---: | ---: | --- |
| 8 | 1.4000 | 54.55% | 0% | 否 |
| 32 | 1.3714 | 63.64% | 4.17% | 是 |
| 64 | 1.3714 | 63.64% | 4.17% | 是 |

因此只按冻结 `[6,10]` bracket 测上界 δ10 以检查 k8，不做额外 sweep。

Qwen δ6（S21）是该模型第一个全 k screening pass，350/350 有效：

| k | MAE（2.0571→） | suc acc（81.82%→） | fail acc（0%→） | 三项通过 |
| ---: | ---: | ---: | ---: | --- |
| 8 | 1.6857 | 90.91% | 4.17% | 是 |
| 32 | 1.7429 | 90.91% | 4.17% | 是 |
| 64 | 1.8000 | 90.91% | 4.17% | 是 |

wrong-region 的 suc 降到 72.73%/63.64%/63.64%、fail 保持 0%，MAE 也变差；low-rank 不复现三项改善。故冻结 `attention_57_qwen_text_images_bidirectional_weighted_bias6_heldout.yaml`，不再调整该布局，启动 H2 confirmatory。

### 4.24 Qwen interleaved calibrated asymmetric gate

S17 的 baseline gate（在任何目标 condition 生成前计算）作为额外 development calibration：阈值冻结为 screening success 最小 score 再减 0.05，即 `−0.3926459186254604`，对应 0/11 suc、7/24 fail 进入 fail branch。success branch 使用既有 δ2 安全下界，fail branch先用 δ6。artifact fingerprint `798a08229557706b8988dd4769a89ce76924c0ad5eabe2ad2e1f5e07bd4ef110`。

S22 有 33 个完整样本、2 个裸 `5` invalid；实际路由为 suc：11 success/0 fail，fail：17 success/7 fail。三组 k 均保持 suc 100%、MAE 下降，但 7 个 fail 在 δ6 尚未到 label1，fail accuracy 仍 0%。因此不改阈值/head，只把 fail branch 提到预设上界 δ10，success 继续 δ2；v3 artifact 写入新文件，早期误名的 `_v2` 文件保留但明确不用于推理。

S23（fail δ10）仍为同样的 33 完整/2 invalid 与 7 个 fail-branch 样本；suc 全部保持 100%，MAE 各 k 下降，但 fail accuracy 仍 0%。提高 fail branch 只把部分预测从 5 推到 2，没有跨到 1；因此终止该 router/dose 路线，不再增大 bias。

RoboReward `text→images` δ10（S24）350/350 有效：k32/k64 的 fail 达 12.5%、suc 54.55%、MAE 1.2857，但 k8 fail 仍 0%。预设 dose 上界已耗尽，且高 dose 的 tail control 也出现改善，说明问题不是“再加 bias”，而是低 raw-mass heads 中存在未被原 candidate pool profile 的 endpoint 因果 heads。

### 4.25 Tail causal discovery（启动）

新增的显式候选模式只从已存在完整 ranking 的末端取 64 个唯一 heads，不读取 held-out 或逐样本输出。RoboReward 候选来自 `text→images` weighted ranking 的末 64；Qwen 候选来自 interleaved fail branch 的末 64。两组分别在 34 success/32 fail development pairs 上逐 head 重跑 `+6` target/wrong causal profile，再按同一双侧 correct-margin/spatial gate 排序。这个步骤检验一个具体反常证据：raw-attention “low rank” 不等于 reward-causal inert，而不是按评测样本挑 head。

### 4.26 H2 结果：Qwen `text→images` 无泄漏确认性失败

冻结配置 `attention_57_qwen_text_images_bidirectional_weighted_bias6_heldout.yaml` 的三个分片均 exit code 0。验证器确认 697 个 example、254 个视频簇、10 个条件，共 `6970/6970` 有效记录，零缺失/invalid；artifact fingerprint 为 `b13797e38af1b31931a695ef799b4e02317e77bf547a418686f44d41cd9e8c57`。该布局在 screening S21 全 k 通过，但没有迁移为 held-out 的三端点平衡提升：

| k | MAE（baseline→target） | suc accuracy | fail accuracy | 严格 gate | cluster bootstrap 95% CI（MAE / suc / fail） |
| ---: | --- | --- | --- | --- | --- |
| 8 | 1.6686→1.5538 | 83.41%→78.03% | 1.69%→6.96% | **失败：suc 下降** | [−0.2215,−0.1188] / [−9.42,−0.90]pp / [+1.98,+5.40]pp |
| 32 | 1.6686→1.5581 | 83.41%→78.48% | 1.69%→7.17% | **失败：suc 下降** | [−0.2175,−0.1158] / [−9.42,−0.45]pp / [+2.17,+5.34]pp |
| 64 | 1.6686→1.5638 | 83.41%→78.48% | 1.69%→6.75% | **失败：suc 下降** | [−0.2182,−0.1125] / [−9.42,−0.45]pp / [+1.98,+5.20]pp |

MAE 与 fail 的 6 个 Holm-adjusted sign-flip p 均为 `0.00089991`，但 suc 的三个校正 p 均为 `0.20608`，且 bootstrap CI 已明确落在负侧。因此这不是“总体改善但证据稍弱”，而是 ordinal/fail 改善与 success endpoint 损害的真实权衡，H2 必须判失败，不能从 held-out 回调该布局。pairwise 正差率仍从 64.01% 提高到 71.98%/71.98%/72.21%，说明同视频排序改善并不蕴含 suc exact endpoint 同时改善。wrong-region 的 MAE 为 1.7374/1.7403/1.7346 且 fail accuracy 仅 0.42%/0.42%/0.21%；low-rank 也不复现 target 的三项模式，故空间/head 特异性不挽救主 gate 失败。

#### H2 的 11 类统计谬误扫描（11/11 checked）

1. Simpson：总体 MAE/fail 改善与 suc 损害均另作分层；不以 pairwise 或总体准确率掩盖 suc 方向反转，task 异质性保留。
2. Ecological：推断单位为 example，bootstrap 单位为 video cluster；不从 task 聚合量推断单条样本。
3. Berkson：cohort 受 auto-valid-grounding 选择，外部推广仅限同 eligibility；内部配对未按候选输出再筛选。
4. Collider：grounding eligibility 可能关联任务难度与可定位性；未控制 baseline/target 输出，也不外推未 grounding 数据。
5. Base-rate neglect：明确报告 223 suc、474 fail 以及两类准确率；没有用 baseline 总准确率替代类别表现。
6. Regression-to-mean：held-out 未按极端 baseline 选择，每条样本以同输入 baseline 配对。
7. Survivorship：6970/6970 完整，无 parser dropout 或只保留有效子集。
8. Look-elsewhere：H2 配置由 S21 在读取 held-out 前冻结；9 个主检验统一 Holm 校正，失败结果完整保留。
9. Garden of forking paths：此前开发路径逐项记录；H2 的 partition、ranking、weights、dose 与 controls 未从 H2 结果回调。
10. Correlation≠causation：配对 hook 干预可支持“该配置改变本 cohort 模型输出”，不支持现实机器人成功率或更广模型能力的因果外推。
11. Reverse causality：干预在 generation 前施加，标签只在推理后 join；输出不能反向选择 head、剂量或分支。

H2 的 Verification Status 为 `ANALYZED`，结论是明确的 confirmatory failure，而不是未验证或可择优解释的 success。

### 4.27 Tail profiling 与 RoboReward S25–S26

RoboReward `text→images` 的 64 个 tail heads 在 34 success + 32 fail development videos 上完成 profile。18 个双侧安全 heads 构成 contrastive artifact，fingerprint `a2d556be0cb7ea69c3d04857da45e669fce7eb968151fe29012912cbea848c75`；其正向源 ranking fingerprint 为 `52726f0f1840dcf366d62ae969b1a32467da2a5a27b21772e668962f60679658`。

S25 `attention_58_*` 使用原 padding 倍率与 δ6，350/350 有效。k8/k32 的 MAE 由 2.0571 降至 1.8571，suc 45.45%→54.55%，但 fail 仍 0%；k64 因 46 个 padding heads 也承受完整剂量，MAE 2.1429、suc 18.18%、fail 0%，明显有害。这个结果定位出“安全 top18 有效，但 padding 剂量污染 k64”的实现层机制，而不是给增加 bias 提供依据。

因此仅由 development artifact 冻结 padding multiplier=0.1，18 个安全 heads 保持 1.0，生成新 artifact；S26 `attention_60_*` 使用 δ10，350/350 有效：

| k | MAE（2.0571→） | suc（45.45%→） | fail（0%→） | 结论 |
| ---: | ---: | ---: | ---: | --- |
| 8 | 1.8286 | 54.55% | 0% | fail 未提高 |
| 32 | 1.8000 | 54.55% | 0% | fail 未提高 |
| 64 | 1.8286 | 54.55% | 0% | fail 未提高 |

冻结 padding 修复了 k64 崩坏并令三个 k 行为一致，但没有一个 fail 跨到 exact label 1。wrong/low controls 也出现零星单样本 label-1，进一步说明在 n=24 的小 screening 中不能用一次边界跳变择优。`text→images` 的原 candidate、signed/weighted、δ6/8/10、tail 与 frozen-padding 路线至此均耗尽；不再沿同布局扫 dose。

### 4.28 Qwen interleaved tail S27：完整反证

Qwen tail 正向 ranking fingerprint 为 `c3d5593dac7e68187535bb30c4593cf01c872e6b7d839b81d86ff86ba607a98a`；20 个双侧安全 heads 的 contrastive fingerprint 为 `4e1d364225d45ab3472da18fa3015c206921f5e803aee879d582ba73dce0ae5e`。与 S26 相同，top20 multiplier=1、rank21–64 padding=0.1，frozen-padding artifact fingerprint 为 `6d91304e062bc954f60fb61833bc105c860a900b28b844b15c1b5173aab8b21d`。

S27 `attention_61_qwen_interleaved_tail_frozen_padding_bias10_screen.yaml` exit code 0，严格为 35 样本 × 10 conditions = 350/350 `ok`，无重复 key、无缺失 prediction。结果直接否定该路线：

| k | MAE（2.0857→） | suc（100%→） | fail（0%→） | 严格 gate |
| ---: | ---: | ---: | ---: | --- |
| 8 | 2.2000 | 100% | 0% | 失败：MAE 变差、fail 不变 |
| 32 | 2.2286 | 100% | 0% | 失败：MAE 变差、fail 不变 |
| 64 | 2.1714 | 100% | 0% | 失败：MAE 变差、fail 不变 |

wrong-region 同样变差；low-rank k64 虽把 5/24 fail 推到 label1、总体 MAE 降至 1.2286，却同时把 suc 从 100% 降到 81.82%，而且该 control 本就不是预先 profile 的候选方法，不能事后反转 ranking 选它。S27 不进入 held-out。结合 uniform、signed、weighted、adaptive router、balanced portfolio 与 tail 的一致失败，Qwen interleaved 的当前 reward-5 margin profiler 路线正式停止。

### 4.29 下一独立构造：`images→text`

两个模型仍未满足主线计数：RoboReward 只有 interleaved 一个 held-out-confirmed layout，Qwen 为零。下一步不复用上述 heads，而以既有 `attention_10_roboreward_images_text_all_frames` / `attention_12_qwen_images_text_all_frames` 的 layout-specific raw/excess/enrichment ranking 为候选源，在相同 34/32 个 development video pairs 上重新做正向、contrastive、显式负方向及 strength-weighted profiling。该构造把所有图像放在文本之前，causal 可达性与 `text→images`/interleaved 均不同，是方法上独立的检验；held-out 标签仍不进入 ranking、剂量或推理 gating。

### 4.30 `images→text` causal profiling（启动）

新增四个配置均在 `auto_explore/` 子目录，未改变正式矩阵；配置解析通过且 `test_crossmodel_image_sequence.py` 为 `5 passed`。正向 profile 候选由该模型、该布局的 raw/excess/enrichment top64 并集确定，分别以两个 head shards 执行。首批命令：

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_rank_roboreward_images_text.yaml --head-shard-id 0 --num-head-shards 2
CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_rank_roboreward_images_text.yaml --head-shard-id 1 --num-head-shards 2
CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python -m mydata_bench.auto_explore_causal_rank profile --config mydata_bench/configs/v2_crossmodel/auto_explore/causal_rank_qwen_images_text.yaml --head-shard-id 0 --num-head-shards 2
```

每个进程仅使用 GPU 0/1/2，hard timeout 3 小时；不自动重试 crash。Qwen 的第二分片在任一 GPU 释放后启动。

首轮完整性与聚合结果：

- RoboReward positive：101 candidates × 34 development success = `3434/3434 ok`，0 invalid；positive ranking fingerprint `8a8f67d9f13c361818d7d583467ab6880b8337a2f8c10d2b8c8c6cd1f45d538e`。
- RoboReward fail：101×32=`3232/3232 ok`，0 invalid；9 个正向双侧安全 heads，contrastive fingerprint `918eda730f05e8dafccb3d0b6bccf43a8f72d7351390f8247d2d12da0497c4fd`。另有 55 个 heads 在 +6 下 success/fail correct-margin 均为负，进入显式 −6 profile，不能仅按代数反号推断。
- Qwen positive：101×34=`3434/3434 ok`，0 invalid；positive ranking fingerprint `818288f80f7959feb8c688f0a6ec72d7da6a0ca05578c9907ac9f8121e28e305`。

随后并行启动 Qwen 的两份 fail profile 与 RoboReward 55 个双负 heads 的 success profile；所有选择仍只来自 development artifact。

Qwen fail profile 随后以 `3232/3232 ok` 完成；23 个正向双侧安全 heads，contrastive fingerprint `ab59ef31a4a66fbb07c7a32bb70f73b74fa3f8e2e316045dc2ba04d692059859`。另有 28 个 +6 双负 heads，分别在 34/32 development success/fail 上显式执行 −6 profile。同期 RoboReward 的 −6 success 为 `55×34=1870/1870 ok`，并继续执行对应 fail profile。

显式负方向与最终聚合均完整：

- Qwen −6 为 success `952/952 ok`、fail `896/896 ok`；最终 28 个安全 heads（23 positive、5 negative）+36 个 0.1 padding，896/896 head 唯一。signed fingerprint `e438072ce482e0248b527ff3ecbcf804fb33600051f58a2456771229c42f45e7`，weighted fingerprint `e0a0bce30d8e2c840ce00808b9da265b780b740287ac9294c65339723738cfe1`。
- RoboReward −6 为 success `1870/1870 ok`、fail `1760/1760 ok`；最终 19 个安全 heads（9 positive、10 negative）+45 个 0.1 padding，896/896 唯一。signed fingerprint `2fe69eaf6f2ebd80f8beecd076ef0d755fb1c84a2608d6b839f3c8048e4ea111`，weighted fingerprint `b348022138f9dc51e99e286ecc7cefe536886e23b7b7b07860872cde0220436d`。

#### S29：Qwen `images→text` weighted δ6 全 k screening pass

`attention_63_qwen_images_text_bidirectional_weighted_bias6_screen.yaml` exit code 0，35 样本×10条件=`350/350 ok`，零缺失/重复：

| k | MAE（1.5714→） | suc（63.64%→） | fail（8.33%→） | 三项通过 |
| ---: | ---: | ---: | ---: | --- |
| 8 | 1.4857 | 90.91% | 12.50% | 是 |
| 32 | 1.4571 | 90.91% | 16.67% | 是 |
| 64 | 1.4286 | 90.91% | 12.50% | 是 |

wrong-region 的 MAE 为 1.6286/1.6571/1.6571，且 suc/fail 均不改善；low-rank 也不复现三项提升。故不再调该布局，冻结相同 ranking/weights/δ6 为 `attention_64_qwen_images_text_bidirectional_weighted_bias6_heldout.yaml`，使用既定 697-record manifest 并启动三个 held-out 分片。H3 结果不得回调该布局。

#### S28/S30：RoboReward `images→text` dose bracket

S28 `attention_62_*` δ6 exit code 0、350/350 有效。k8/k32 target 与 baseline 完全相同；k64 仅 MAE 1.0857→1.0286，suc/fail exact accuracy 仍为 63.64%/66.67%。因此不进入 held-out。与此前产生显著端点权衡的失败不同，该结果主要是干预不足；只按既有预设 `[6,10]` bracket 测一次文献上界 δ10（S30 `attention_65_*`）。若仍不全 k 通过，即停止该 layout 的 weighted dose 路线，不做密集 sweep。

S30 同样 exit code 0、350/350 有效。k8/k32/k64 均为 MAE 1.0857→1.0571，但 suc 保持 63.64%，fail 从 66.67%降到62.50%；三个 k 都失败。wrong/low controls 也表明 ordinal MAE 变化并不等价于 exact endpoint 双侧改善。δ10 已是预设上界，故正式停止 RoboReward `images→text` weighted dose 路线。

### 4.31 第四构造：native `text→video` causal profiling

因为 RoboReward 仍缺第二个 confirmed layout，且 Qwen 即使 H3 通过也仍只会有一个，下一构造使用真正的 native video processor，而不是再改 image ordering。候选分别来自既有 `experiments_v2/attention_07_roboreward_all_frames` 与 `attention_09_qwen_all_frames` 的 layout/model-specific raw/excess/enrichment top64 并集：RoboReward 98 heads、Qwen 104 heads；development 仍为同一冻结 34 success/32 fail video clusters。

新配置保留 `protocol=roborewardbench_native`、`content_order=text_then_video`、最多 8 帧以及既有 native temporal span 设置；只把 profiler 的 negative scope 设为 none。首批每模型启动正向 shard0 作为真实 runtime smoke；两者均成功经 torchvision fallback 解码 native video 并持续写入 `status=ok`，没有把视频预先偷换为图像序列。

### 4.32 H3 结果：Qwen `images→text` 无泄漏确认性通过

`attention_64_qwen_images_text_bidirectional_weighted_bias6_heldout.yaml` 三个分片均 exit code 0。验证器确认 697 examples、254 video clusters、10 conditions，`6970/6970` 有效、无重复/缺失/invalid；validation fingerprint `9fe59326e3351c65e541dff22fc44a3d147c4f8791ae9bb9b0898dd1fc15186c`。

| k | MAE（baseline→target） | suc accuracy | fail accuracy | cluster bootstrap 95% CI（MAE / suc / fail） |
| ---: | --- | --- | --- | --- |
| 8 | 1.5222→1.4003 | 52.91%→69.06% | 6.75%→19.83% | [−0.1375,−0.0466] / [+12.11,+20.63]pp / [+6.59,+10.28]pp |
| 32 | 1.5222→1.3816 | 52.91%→74.44% | 6.75%→21.10% | [−0.1522,−0.0535] / [+16.59,+26.46]pp / [+7.51,+11.66]pp |
| 64 | 1.5222→1.3816 | 52.91%→73.99% | 6.75%→20.89% | [−0.1539,−0.0561] / [+16.14,+26.01]pp / [+7.44,+11.33]pp |

9 个主 sign-flip tests 经 Holm 校正后均为 `p=0.00089991`；McNemar 的 suc/fail/overall p 也全部远小于 .001。预测分布证明不是单向端点推动：baseline suc label5 118→154/166/165，同时 fail label1 32→94/100/99。pairwise positive rate 从 66.51% 变为 66.97%/67.20%/68.34%，仅小幅提高；这说明主效应主要是两侧端点校准，而不能夸大为所有同视频 instruction 排序都显著重构。

特异性 controls 成立：wrong-region 的总体 MAE 为 1.5395/1.5280/1.5208，suc/fail accuracy 均低于 baseline 或近似不变；low-rank 虽在个别一侧移动，但没有任一 k 复现 target 的 MAE+suc+fail 三项改善。task MAE 也有真实异质性：k8/32/64 分别为 17/15/16 个 task 改善、5/9/8 持平、6/4/4 变差；反复出现的 harmful strata 包括 `task3_3/task3_6/task5_1`，故结论不外推为逐 task 普遍改善。

#### H3 的 11 类统计谬误扫描（11/11 checked）

1. Simpson：同时报告总体、suc/fail、task；总体三方向没有被两类 endpoint 反转，但 task 异质性保留为 caution。
2. Ecological：example 为推断单位，video cluster 为重采样单位；不从 task 均值推断每条轨迹。
3. Berkson：auto-valid-grounding 是选择性 cohort，结论只适用于相同 eligibility；内部比较没有再按输出筛样本。
4. Collider：grounding eligibility 可能关联可定位性与难度，故不外推未通过 grounding 的视频；未按 baseline/target 输出控制变量。
5. Base-rate neglect：明确报告 223 suc、474 fail 及两侧准确率，不以总体 accuracy 取代类别表现。
6. Regression-to-mean：held-out 未按极端 baseline 选择，每条样本均有同输入 paired baseline。
7. Survivorship：6970/6970 完整，无 parser dropout、无只保留成功生成。
8. Look-elsewhere：H3 由 S29 在读取 held-out 前冻结，9 个主检验统一 Holm；此前所有失败路径保留在日志。
9. Garden of forking paths：partition、ranking、signed weights、δ6 与 controls 在 H3 启动前冻结，H3 不回调参数。
10. Correlation≠causation：配对 hook 支持“该配置在该 cohort 改变模型输出”的模型内因果描述，不支持现实机器人成功率或其他模型的外推。
11. Reverse causality：steering 在 generation 前施加；评测标签只在生成后 join，不能反向决定 head、方向、剂量或样本分支。

H3 Verification Status 为 `ANALYZED`。至此 Qwen 完成第一个 held-out-confirmed layout；它仍需要第二个，故主线继续。

### 4.33 Native `text→video` profiling 与 RoboReward S31

RoboReward positive profile 完整为 `98×34=3332/3332 ok`，fingerprint `e6031ec1a37a6919f753e2566546db391db0c2697f1889e14081a1b54d763649`；同视频 fail 为 `98×32=3136/3136 ok`。contrastive 聚合得到 10 个正向双侧安全 heads，fingerprint `a0ef38f6bcc3455e641c2f52b381b19698e54f53a2e620895756ff4614e0d774`。27 个 `+6` 下 success/fail correct-margin 均负的 heads 随后显式重跑 `−6`，success `918/918 ok`、fail `864/864 ok`，不存在用代数反号替代实测。

最终双向 artifact 含 20 个安全 heads（10 positive、10 negative）与 44 个 0.1 padding，896/896 `(layer,head)` 唯一；signed fingerprint `61d6ed33cc65560db4286947f1487d6246d5cba2c44c905110c3c758f461bc1e`，weighted fingerprint `2f2e0e4cbe70752ea1a6776e55a2fed350c2b97d5b64f3f79e5ae87f03b51d06`。

S31 `attention_66_roboreward_text_video_bidirectional_weighted_bias6_screen.yaml` exit code 0，35 examples×10 conditions=`350/350 ok`、零缺失/重复。screening shard 为 11 suc、24 fail；标签只在生成完成后用于评价。

| k | MAE（baseline 2.4571→） | suc（72.73%→） | fail（0%→） | 严格 gate |
| ---: | ---: | ---: | ---: | --- |
| 8 | 2.3143 | 81.82% | 0% | 失败：fail 未提高 |
| 32 | 2.3429 | 72.73% | 0% | 失败：suc/fail 未严格提高 |
| 64 | 2.3429 | 81.82% | 0% | 失败：fail 未提高 |

target 的 ordinal MAE 在三个 k 均下降，但 24 个 fail 仍没有一个到达 exact label 1；wrong-region MAE 为 2.4000/2.4000/2.4000，low-rank 为 2.4571/2.5143/2.7143，均未复现 target 的整体模式。由于预注册主 gate 要求三个 k 的 MAE、suc、fail 同时严格改善，S31 不进入 held-out；也不从 fail=0 的 screening 输出继续调 dose。

### 4.34 第五独立构造：native `video→text`

RoboReward 尚只有 interleaved 一个 confirmed layout；`text→images`、`images→text` 与 native `text→video` 的既定路线均已耗尽。最后转向第五种、尚未做本轮 causal profiling 的 native `video→text`。候选源是该模型/布局既有 `experiments_v2/attention_15_roboreward_all_frames` 的 raw/excess/enrichment top64 并集；processor 保持 `roborewardbench_native`、8 帧与 `native_pairs`，唯一布局变化为 `content_order=video_then_text`。新增 positive/contrastive/signed/weighted 与 S33 配置均放在 `auto_explore/`，正式 16-config 矩阵测试仍为 `5 passed`。held-out 与 S31 输出都不参与 head 选择。

### 4.35 Qwen native `text→video` S32

Qwen positive 为 `104×34=3536/3536 ok`，fingerprint `1f00cb6e9a683b988a825fce8c96dc627867a8ba85ed1b8ab2dbc707d418ba46`；fail 为 `104×32=3328/3328 ok`。contrastive fingerprint `37414342eb28151531bf56af1cc4df1f4b5abebab6267eeb8882d86bf21d8402`，16 个正向安全 heads、35 个双负候选。显式 `−6` success/fail 分别为 `1190/1190 ok`、`1120/1120 ok`。最终 17 个安全 heads（16 positive、1 negative）+47 padding，896/896 唯一；signed fingerprint `a6f2641efeb3ab928eefbb41bc06d19e847ec0baf4bf4b1a1be00ec2ad31dc96`，weighted fingerprint `ca2496705f8b249a004ac5a593c378f9c7bc35a9df8ad1daf859b493b55f2bfc`。

S32 `attention_67_qwen_text_video_bidirectional_weighted_bias6_screen.yaml` exit code 0、`350/350 ok`：

| k | MAE（baseline 1.4571→） | suc（81.82%→） | fail（0%→） | 严格 gate |
| ---: | ---: | ---: | ---: | --- |
| 8 | 1.3714 | 72.73% | 0% | 失败：suc 下降、fail 未提高 |
| 32 | 1.3714 | 72.73% | 0% | 失败：suc 下降、fail 未提高 |
| 64 | 1.3714 | 72.73% | 0% | 失败：suc 下降、fail 未提高 |

wrong-region 的 MAE 为 1.4571/1.4286/1.4286，low-rank 为 1.4571/1.7429/1.6571，均不复现 target 的完整模式；但主结果仍是明确的 endpoint trade-off，不能进入 held-out。

Qwen 因此也转向 native `video→text`。该布局没有既有 Qwen attention ranking，于是新增 `attention_69_qwen_video_text_ranking_source.yaml`，只在原 36 条 success ranking metadata 上用 `content_order=video_then_text` 生成 layout-specific raw/excess/enrichment ranking。`prepare-ranking` 与 `prepare-cohort` 成功；既有 `validate-ranking` 子命令错误地对 native `.mp4` 调用 `PIL.Image.open`，以 `UnidentifiedImageError` 退出，按协议未重试、未修改验证器。实际 native `rank` runtime 则 exit code 0，并生成三份 896-head ranking；raw/excess/enrichment fingerprints 分别为 `8a7bf5ad1a937b0faee851fda956c0c16615d6bba8c204d63f56e7ace6f7fadd`、`9cce559da87cd816cdaa0848565dacd8403589c6325e50b85dc6d42dc33057e8`、`19b36c7878a23f9ef1e45e587494bd12a4fcef07736b67e01f6c3af70d8c9090`。后续 causal profile 只使用这些新 ranking，不复用 `text→video` heads。

### 4.36 RoboReward native `video→text`：完整方法族反证

positive 为 `102×34=3468/3468 ok`，fingerprint `7bd0c447667dc294c3ac5d6ed36302383ae127770ac7835828c9be9e2460074c`；fail 为 `102×32=3264/3264 ok`。contrastive artifact 含 8 个正向安全 heads，fingerprint `22da81d0192ed07ad843744f07b7fe1d49389aa68ed68703f14463583c16905a`；59 个双负 heads 的显式 `−6` success/fail 为 `2006/2006 ok`、`1888/1888 ok`。最终 14 个安全 heads（8 positive、6 negative）+50 padding，896/896 唯一；signed fingerprint `fd1abccb7618edff185c7d81f122cc966e7d625dedfe4a92031da8fd99ebc1fe`，weighted fingerprint `27e0add49bdf8f906f77d6118af3581b73392d464b09dec0d83bd5ca46bd991e`。

S33 δ6 与 S34 δ10 均为 350/350 `ok`。baseline 是 MAE 0.7143、suc 72.73%、fail 75%；三个 k 在 δ6 都是 0.6000/72.73%/79.17%，δ10 都是 0.6000/72.73%/83.33%。因此 ordinal/fail 改善真实但 suc 严格持平，不能进入 held-out。

随后只测试机制上独立的有限分支，全部 350/350 `ok`：

- S35 development-balanced global portfolio（8 heads，fingerprint `dc3246273d57e4749ce972321ef58ce589ce522225a47ae95652480d6c99767e`）在 bias6 完全惰性；
- S36 label-blind adaptive router 的 development leave-video-out success/fail accuracy 均 87.5%，但 screening 实际只正确路由 9/11 suc、12/24 fail；k8 仅 MAE 改善，k32/64 损害 suc。adaptive fingerprint `818fa2c726049c65a63e54d6abedc15f7ff94723b7002eed7a75eb73d79404a5`；
- S37/S38 无 router 的 success-specialist δ6/10：δ6 仍不改变 suc，δ10 反而把 fail 降到 62.5–66.67%；
- S39 symmetric `negative_scope=all_visual` δ6：k8/32 为 0.6000/72.73%/79.17%，k64 fail 回到 75%，仍无 suc 提升；
- S40 `last_prompt` δ6：所有条件与 baseline 完全相同，证明该布局的可见效应来自广泛 prefill 重写而非最后决策 query。

这组结果没有筛选缺失、parser dropout 或 crash；失败集中在 exact suc endpoint，而非总 MAE。RoboReward `video→text` 的 causal/weighted、dose、balanced、adaptive、specialist、symmetric 与 query-scope 路线均已耗尽，不从 screening 标签反向挑 head，也不进入 held-out。

### 4.37 Qwen native `video→text` S41：全 k screening pass，冻结 H4

Qwen 的 layout-specific causal 链完整结束：positive ranking fingerprint 为 `3e230e45afe4e5649cd4301048a0fdf29c8891a045d0cb434817989404055b7a`，contrastive 为 `52639d4725b838478edbaad97185f0740620ceb39d8426ba516c13f4547d9319`。显式负方向 profile 后，最终得到 34 个双侧安全 heads（26 positive、8 negative）与 30 个低权重 padding heads；signed fingerprint 为 `fb1e73816982b855c449d5146203f5c6cd31871e87f56209b79caf81c3917802`，weighted fingerprint 为 `a6e6868f8d23ca6f8c5b6e73ebb65718242c4e926325b7ac5027bd7a91c69acf`，896/896 `(layer,head)` 唯一。

S41 `attention_70_qwen_video_text_bidirectional_weighted_bias6_screen.yaml` exit code 0，严格为 35 examples×10 conditions=`350/350 ok`，350 个 `(example_id,condition)` 唯一；screening shard 含 11 suc、24 fail。结果为：

| k | MAE（baseline 1.4000→） | suc（81.82%→） | fail（4.17%→） | 三项通过 |
| ---: | ---: | ---: | ---: | --- |
| 8 | 1.0857 | 100% | 33.33% | 是 |
| 32 | 1.0857 | 100% | 29.17% | 是 |
| 64 | 1.1143 | 100% | 29.17% | 是 |

干预特异性也通过 screening gate：wrong-region k8/32/64 的 MAE 为 1.4286/1.4286/1.3714，suc/fail accuracy 均保持 81.82%/4.17%，没有复现 target 三项改善；low-rank 的 MAE 为 1.3429/1.3714/1.4000，suc 为 90.91%/90.91%/81.82%，fail 为 4.17%/8.33%/12.50%，也没有任一 k 同时复现 target 的 MAE、suc、fail 改善。

因此不再读取 screening 输出调 head、direction、weight 或 δ，冻结相同 artifact/δ6/all-frame scope 为 `attention_79_qwen_video_text_bidirectional_weighted_bias6_heldout.yaml`。它只把 label-blind example manifest 换为既定的 697-record held-out（partition fingerprint 仍为 `4d65d16add4129b3afc0b16f24495189d47580e5688b7167b4921c0cd528fee2`）；H4 三分片中的 shard0 已在 GPU0 以 3 小时 hard timeout 启动，其结果不得回调该布局。

一次非实验配置检查命令使用 conda 环境中的裸 `pytest`，因该环境没有此入口而以 exit code 127 结束；按不自动重试规则未重跑。正式 16-config 矩阵最近一次仍为 `5 passed`，且 H4 新文件仅冻结 S41 字段并替换 manifest/output，不加入正式矩阵。

### 4.38 RoboReward `video→text` temporal prior：last-frame v2 分支

all-frame 方法族的 exact suc endpoint 始终持平后，使用已有、layout-specific 的 `experiments_v2/attention_14_roboreward_last_frame` source ranking，重新 profile `temporal_intervention_scope=last_frame`；没有复用 all-frame heads。positive 两分片均为 `1496/1496`，即 88 candidates×34 development success=`2992/2992 ok`；fail 两分片均为 `1408/1408`，即 88×32=`2816/2816 ok`，各进程 exit code 0。

原 `aggregate-contrastive` 首次执行只找到 7 个 `+6` 双侧安全 heads，低于工具原本要求的中间 top-8 下限，因此以 exit code 1 拒绝生成 ranking；该失败未自动重试、未覆盖任何结果。一次并行的只读 audit 小脚本也因把整数 `head` 当成字典而抛出 `AttributeError`；它不写文件、不影响后续聚合器独立给出的 7-head 结论。

这里的科学问题是：C3 的“正方向至少 8 个”只是实现时的中间约束，而 C4 原本就要求对 development artifact 中 `+6` 两侧均负的 heads 显式执行 `−6`，最终 bidirectional artifact 才是真正必须支持 top8 的方法。为不在已观察 held-out/screening 后降低最终门槛，新增默认不变的两个可选字段：`fail_profile_dir` 允许新版本聚合器只读旧 fail shards，`minimum_positive_safe_heads` 默认仍为 8；仅新的 v2 中间分支设为 1。最终 `aggregate-bidirectional` 的 `8 <= safe <= 64` 原门槛完全不变。

新 v2 contrastive artifact 使用相同 88×32/34 development evidence，含 7 个 positive-safe heads +57 个低 raw-mass padding，fingerprint `65553b64cc587b453ea1b10cbc4c590c18c0121c29dfbcecef5fa8850146457b`，896/896 heads 唯一；45 个 heads 在 `+6` 下 success/fail correct-margin 均负，进入显式 `−6` success/fail profile。该结构修正只由开发 artifact 的 head 数触发，不读取 S41、H4 或 RoboReward screening/held-out 标签；若显式负方向加入后总安全数仍少于 8，分支终止，不绕过最终 gate。

显式 `−6` profile 随后完整结束：success `45×34=1530/1530 ok`、fail `45×32=1440/1440 ok`。最终 bidirectional 有 12 个安全 heads（7 positive、5 negative）+52 padding，896/896 唯一；signed fingerprint `6ba2f0ed4c3c4d6c1fb66f063ae1b1b2bc4249698918610afabf667858b2b955`，weighted fingerprint `fa6d97a0488c36cf2c716a17c327be0d9362bb9087761b1fbb13871ca693d95c`。

S42 `attention_80_*` 为 350/350 `ok`：baseline MAE/suc/fail 为 0.7143/72.73%/75%；k8/32/64 target 全为 0.6000/72.73%/79.17%。MAE 与 fail 改善，但 exact suc 全部严格持平，且 wrong/low control 也能复现部分 fail 变化，故不进入 held-out、不扫 dose。

同一 last-frame development evidence 生成的 label-blind router 有 34/38 个 success/fail-safe heads，leave-video-out success/fail accuracy 为 78.13%/84.38%，保守阈值使 development success 0/32、fail 10/32 进入 fail branch，artifact fingerprint `9033bb96177ec575bcef5c046102ffe0dbe5496e01ecf9b59924ed2a1dc3e98e`。S43 adaptive 与 S44 success-specialist 均为 350/350 `ok`：adaptive 实际把 1/11 suc、9/24 fail 送入 fail branch，k32/64 只把 fail 75%→79.17%，suc 仍 72.73%；success-specialist 同样只有 fail 移动、suc 不动。last-frame weighted/adaptive/specialist 方法族因此终止。

### 4.39 H4：Qwen native `video→text` 全 k 方向通过，但 k8 稳健性不足

`attention_79_qwen_video_text_bidirectional_weighted_bias6_heldout.yaml` 三分片均 exit code 0，行数 2390/2370/2210。验证器确认 697 examples、254 clusters、10 conditions，`6970/6970` 有效、无重复/缺失/invalid；validation fingerprint `2695933427dd88d688a5215502051fdc46d7d278c83ee9b5cd10bd5c80085daf`。

| k | MAE（baseline 1.4476→） | suc（73.99%→） | fail（11.60%→） | cluster bootstrap 95% CI（MAE / suc / fail） | Holm sign-flip（MAE） |
| ---: | ---: | ---: | ---: | --- | ---: |
| 8 | 1.3888 | 84.75% | 23.63% | **[−0.0591,+0.0174]** / [+7.62,+14.35]pp / [+6.13,+9.75]pp | 0.3636 |
| 32 | 1.3587 | 84.75% | 24.68% | [−0.0886,−0.0174] / [+7.17,+14.35]pp / [+6.65,+10.34]pp | 0.0312 |
| 64 | 1.3630 | 85.65% | 24.68% | [−0.0837,−0.0105] / [+8.07,+15.25]pp / [+6.79,+10.28]pp | 0.0654 |

九个点估计全部满足预设方向，suc/fail 的六个 Holm-adjusted sign-flip p 都为 `0.00089991`，McNemar 也全部远小于 .001；但 k8 MAE cluster CI 跨 0 且 Holm p=0.3636，不能称为“稳定 MAE 降低”。k64 的 cluster CI 虽排除 0，Holm sign-flip p=0.0654 也提示离散 episode-level effect 受少数大变化影响；只有 k32 的三项统计证据全部较稳健。故 H4 严格判为“全 k 方向通过、k8 MAE 证据不足”，不计入强 held-out-confirmed layout；Qwen 强确认仍为 1/2。

预测分布不是单向推向一个 endpoint：suc label5 从 165 增至 189/189/191，fail label1 从 55 增至 112/117/117；但 fail label5 也从 103 增至 136/127/129，解释了总体 MAE证据弱于 exact accuracy。pairwise positive rate从 65.15% 变为 64.46%/66.29%/66.29%，k8 甚至略降，不能把 endpoint 校准夸大为普遍 instruction 排序改善。wrong-region 的 suc/fail accuracy 全低于 baseline；low-rank 也不复现 target 三项模式，干预特异性成立。task MAE 的 improve/tie/harm 分别为 k8 `10/6/12`、k32 `11/6/11`、k64 `11/6/11`，异质性很强。

#### H4 的 11 类统计谬误扫描（11/11 checked）

1. Simpson：总体三方向另按 suc/fail/task 展开；task 大量反向 strata 与 k8 MAE CI 跨 0 被保留，未以 endpoint accuracy 掩盖。
2. Ecological：example 为推断单位、video cluster 为重采样单位；不从 task 均值推断单条样本。
3. Berkson：auto-valid-grounding 是选择性 cohort，外部结论只限同 eligibility；未按 H4 输出再筛样本。
4. Collider：grounding eligibility 可能共同关联难度和可定位性，故不控制输出后变量、不外推未 grounding 视频。
5. Base-rate neglect：明确报告 223 suc、474 fail 与两类 accuracy，不用总体 accuracy 替代类别表现。
6. Regression-to-mean：held-out 未按极端 baseline 选择，每条样本都有同输入 paired baseline。
7. Survivorship：6970/6970 完整，零 parser dropout/attrition。
8. Look-elsewhere：H4 ranking/δ/scope 由 S41 在读取 H4 前冻结；九个主检验统一 Holm，且弱 k8 结果没有被隐藏。
9. Garden of forking paths：此前方法分支均在日志保留；H4 不回调 head、weight、dose 或 scope，但多布局探索使全局结论仍标记 caution。
10. Correlation≠causation：配对 hook 只支持“该配置改变本 cohort 模型输出”的模型内因果描述，不支持现实机器人成功率外推。
11. Reverse causality：steering 在 generation 前施加，标签只在生成后 join，不能反向决定干预。

H4 Verification Status 为 `ANALYZED`，Overall Confidence 为 `CAUTION`：方向性结果完整且 endpoint effect 很强，但预注册的全 k 稳健 MAE 条件未全部满足。

### 4.40 RoboReward `images→text` 类别自适应与配平反证

既有 layout-specific development profiles 构造的 router 有 33/46 个 success/fail-safe heads，leave-video-out success/fail accuracy 87.5%/90.63%，保守阈值使 development success 0/32、fail 8/32 进入 fail branch；fingerprint `49a8ffd31d7a62df1b7cc8b26fe08e5f4f9ba80615da0af04742ad3568b4dbb1`。S45 adaptive、S46 success-specialist、S47 四对 complementary balanced portfolio 均为 350/350 `ok`。

- S45 实际只把 1/11 suc、3/24 fail 送入 fail branch；k8 完全惰性，k32/64 exact suc 仍 63.64%，fail 反而 66.67%→62.5%。
- S46 k8 惰性，k32/64 仅 suc MAE 小幅下降，exact suc 不变且 fail 降到 62.5%。
- S47 三个 k 的 MAE 1.0857→0.9429，但 suc 保持 63.64%、fail 降至62.5%；wrong/low 也有同向 MAE移动，特异性不足。

因此该布局的 weighted δ6/10、adaptive、success-specialist、balanced portfolio 均不能同时改善双端点，不进入 held-out。

另一次 `text→images` adaptive artifact 构建因旧 positive profiles 实际为 `00-of-01`、新配置误写为两分片而以 `FileNotFoundError` 退出；没有生成产物，按不自动重试规则未用修正配置重跑。

### 4.41 RoboReward native `text→video` 类别自适应、专门化与配平反证

该布局的 v2 adaptive artifact 完全由 layout-specific development causal profiles 构建：success/fail 分支分别有 26/52 个安全 heads，leave-video-out success/fail 分类准确率均为 90.63%；原保守阈值在 development 上仅把 3/32 个 fail 送入 fail branch。artifact fingerprint 为 `810104018a7708d56207598f6070dd1c1d32d8c2223666409ee90077c348d653`。S48–S50 均为 35 examples×10 conditions=`350/350 ok`，无缺失、重复或 parser dropout。

| screening | 方法 | k8 MAE / suc / fail | k32 MAE / suc / fail | k64 MAE / suc / fail | 严格结论 |
| --- | --- | --- | --- | --- | --- |
| S48 | label-blind adaptive | 2.4857 / 72.73% / 0% | 2.4286 / 90.91% / 0% | 2.3714 / 90.91% / 0% | fail 三个 k 均未提高；k8 MAE 还变差 |
| S49 | development-balanced global portfolio | 2.3143 / 72.73% / 0% | 2.3143 / 72.73% / 0% | 2.3714 / 72.73% / 0% | MAE 改善，但两个 endpoint 均严格持平 |
| S50 | success specialist | 2.6286 / 72.73% / 0% | 2.5429 / 90.91% / 0% | 2.4857 / 90.91% / 0% | k32/64 suc 改善，但 fail 不变且 MAE 变差 |

共同 baseline 为 MAE 2.4571、suc 72.73%、fail 0%。S48 screening 的实际路由为 suc 11/11 success branch，fail 21/24 success、3/24 fail branch，说明保守 development router 虽保护了 suc，却没有把足够 fail 样本送入互补分支。S49 表明 label-free 全局配平只改善 ordinal 距离，不能令自由生成跨过 exact endpoints；S50 则把 suc/fail trade-off 暴露得更直接。wrong-region 与 low-rank controls 均未复现三项全 k 改善。因此三者都不进入 held-out。

### 4.42 S51：development 校准路由仍不能恢复 fail endpoint

为检验 S48 是否只是阈值过于保守，使用 **S48 已有 development gates** 做一次全局阈值校准，不读取 screening 或 held-out 标签。冻结阈值为 `-0.5610411755062205`，development 路由为 0/11 suc、11/24 fail 低于阈值；success/fail 分支 effective base bias 分别为 2/6。新 artifact fingerprint 为 `c74a82abbcaaf643a4ffee0bc01e3e815c717ddc4dee4ba18a52c704055589c3`，推理时 `inference_uses_labels=false`。

S51 `attention_89_roboreward_text_video_adaptive_calibrated_v2_bias6_screen.yaml` 在 GPU0、3 小时 hard timeout 下 exit code 0。完整性审计为 350 records、350 个唯一 `(example_id, condition)`、35/35 样本各 10 conditions、全部 `status=ok`。实际路由恰为 suc 11/11 success branch；fail 13/24 success、11/24 fail branch，说明阈值迁移本身符合 development 预期，但 endpoint 仍未跨越：

| k | MAE（2.4571→） | suc（72.73%→） | fail（0%→） | wrong-region MAE/suc/fail | low-rank MAE/suc/fail |
| ---: | ---: | ---: | ---: | --- | --- |
| 8 | 1.9429 | 72.73% | 0% | 2.4571 / 72.73% / 0% | 2.3429 / 81.82% / 0% |
| 32 | 1.8000 | 81.82% | 0% | 2.4571 / 72.73% / 0% | 2.5143 / 72.73% / 0% |
| 64 | 1.8000 | 81.82% | 0% | 2.4000 / 72.73% / 0% | 2.5143 / 72.73% / 0% |

因此校准路由显著降低 ordinal MAE，并在 k32/64 改善 suc，但 24 个 fail 在三个 k 仍无一生成 exact label 1。不能再用 S51 的评测标签移动阈值、增加剂量或选择分支；该 calibrated route 按预设终止，不进入 held-out。结合 S31、S48–S51，RoboReward native `text→video` 的 uniform weighted、adaptive、balanced、success-specialist 与 development-calibrated router 均已给出一致反证：当前 causal head family 能移动 ordinal score，却不能同时恢复 exact fail endpoint。

### 4.43 独立机制：exact-endpoint causal ranking（启动）

S31/S48–S51 的共同失败定位出原 causal objective 的一个具体盲点：success profile 优化 reward-5-vs-rest margin；fail profile 只把该 margin 的符号反转，因而只能保证“远离 5”，不能保证“趋向 exact label 1”。这与观察到的 fail 从 5 移到 2/3、但 exact fail accuracy 仍为 0% 一致。为避免继续按 screening 调 router/dose，新增独立的 `auto_explore_exact_endpoint_rank.py`，只读已经冻结的 development 单-head choice logits，完全不读取 screening/held-out 行：

- suc 要求 target 对 `logit(5)-logsumexp(logit(1..4))` 的 task-balanced 因果增量为正；
- fail 要求 target 对 `logit(1)-logsumexp(logit(2..5))` 的 task-balanced 因果增量为正；
- 两侧还必须同时满足 target-region effect 大于 equal-size wrong-region effect；
- 正、负 bias 方向均来自已经显式跑过的 `+6/-6` profiles，不做代数反号；同一 head 若两方向都出现，只保留实测较强方向；
- 安全 heads 使用完整 multiplier `±1`，补到 64 的低 ranking mass heads 固定为 0.1，并为 low-rank control 保留互斥 tail；推理时不使用标签或预测端点。

这仍然是 development-supervised head selection，而不是 inference endpoint hard coding：artifact 明确记录 `development_labels_used_for_head_selection=true`、`inference_uses_labels=false`。RoboReward native `text→video` 得到恰好 8 个 exact-endpoint-safe heads（6 positive、2 negative），artifact fingerprint `28a27dc025bcc7d9d7925b42e5a681447e9b8d07063a5429bb729a54b51bbd9d`；Qwen native `text→video` 得到 9 个（8 positive、1 negative），fingerprint `46fe67f4a11b1ea4e5bcd4bce356e8841b54d8ce272557f04afcb546fe90962b`。两者均为 896/896 唯一 `(layer,head)`。

Qwen `interleaved` builder 因旧 positive profile 的 `development_side` 元数据不符合新构建器契约，以 `ValueError` 退出且未生成 artifact；按不自动重试规则不修正后重跑。一次配置测试使用 `python -m pytest` 也因该 conda 环境无 `pytest` 模块以 exit code 1 结束，未重跑；最近一次有效的正式 16-config matrix 仍是 `5 passed`。

冻结后启动两条 screening，均为 3 小时 hard timeout、无 crash retry：

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_90_roboreward_text_video_exact_endpoint_v1_bias6_screen.yaml --shard-id 0 --num-shards 16
CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=60s 10800s conda run --no-capture-output -n robo-dopamine python mydata_bench/run_qwen_attention.py steer --config mydata_bench/configs/v2_crossmodel/auto_explore/attention_92_qwen_text_video_exact_endpoint_v1_bias6_screen.yaml --shard-id 0 --num-shards 16
```

只在某一布局的 k8/32/64 全部严格满足 MAE↓、suc↑、fail↑，且 wrong/low controls 不复现时冻结 held-out；否则该布局的 exact-endpoint v1 直接终止，不根据 screening 增加 bias 或更换 heads。

#### S52–S53 结果：teacher-forced exact endpoint 不迁移到自由生成

两进程均 exit code 0；各自严格为 350 records、350 个唯一 `(example_id,condition)`、35/35 examples×10 conditions、全部 `status=ok`。

S52 RoboReward native `text→video` 的 baseline 为 2.4571/72.73%/0%（MAE/suc/fail）：

| k | target MAE / suc / fail | wrong MAE / suc / fail | low-rank MAE / suc / fail | 结论 |
| ---: | --- | --- | --- | --- |
| 8 | 2.4286 / 81.82% / 0% | 2.4857 / 81.82% / 0% | 2.4571 / 72.73% / 0% | fail 未提高 |
| 32 | 2.4000 / 81.82% / 0% | 2.4857 / 81.82% / 0% | 2.5143 / 72.73% / 0% | fail 未提高 |
| 64 | 2.4000 / 81.82% / 0% | 2.4857 / 81.82% / 0% | 2.7143 / 54.55% / 0% | fail 未提高 |

target 虽改善 suc，但预测仅落在 3/4/5，24 个 fail 无一到 label 1；wrong-region 还复现了 suc 改善，空间特异性也不足。S52 直接终止。

S53 Qwen native `text→video` 的 baseline 为 1.4571/81.82%/0%：

| k | target MAE / suc / fail | wrong MAE / suc / fail | low-rank MAE / suc / fail | 结论 |
| ---: | --- | --- | --- | --- |
| 8 | 1.6857 / 81.82% / 0% | 1.4000 / 72.73% / 0% | 1.4571 / 81.82% / 0% | target MAE 变差、fail 不变 |
| 32 | 1.7143 / 81.82% / 0% | 1.4286 / 72.73% / 0% | 1.7429 / 81.82% / 0% | target MAE 变差、fail 不变 |
| 64 | 1.7143 / 81.82% / 0% | 1.4286 / 72.73% / 0% | 1.6571 / 63.64% / 0% | target MAE 变差、fail 不变 |

S53 更是方向反转，且 wrong-region 的 MAE 比 target 好，故直接终止。结论不是“还需更强 bias”：单-head teacher-forced exact-choice margin 的四重正 gate，在多-head自由生成时仍出现非加性和区域失配；继续按 screening 加剂量会违反预注册边界。

#### S54–S55：其余 RoboReward 布局的 exact-endpoint 终止

在读取结果前同时冻结两个 baseline 端点结构不同的最后分支：`video→text` 原方法只卡在 suc 严格持平，`images→text` baseline fail 已较强。前者得到 10 个安全 heads（7 positive、3 negative），artifact fingerprint `889df65cb67fb5d2a8745a19e818b61f4ace3b8f2d5f8a1742a375a93dd53536`；后者 16 个（8 positive、8 negative），fingerprint `7d2258b66a63d1355698e3ec3a25c56be0ec5c74a69af3467cfa462812a3604f`。两 artifact 均为 896/896 唯一 heads，padding 规则与 S52/S53 相同。

S54 RoboReward native `video→text` 为 350/350 `ok`；baseline MAE/suc/fail=0.7143/72.73%/75%。k8 target 为 0.6000/72.73%/79.17%，仍只改善 MAE/fail 而 suc 严格持平；k32/k64 target 完全等于 baseline。wrong-region 把 suc 降至 63.64%，low-rank 基本惰性，但这不能补救主 gate。该布局复现原有结构性瓶颈并终止。

S55 RoboReward `images→text` 同样 350/350 `ok`；baseline=1.0857/63.64%/66.67%。k8/k32 target 完全等于 baseline，k64 仅 MAE 变为1.0571，两个 exact endpoints 都不动；wrong/low controls 反而有更大 ordinal 移动。该布局也终止。

至此 exact-endpoint v1 在三个 RoboReward 布局与一个 Qwen 布局均有完整自由生成反证；未改变任何一个模型的 held-out-confirmed layout 计数。由于所有分支均在启动前冻结为“失败即终止、不加 bias/换 heads”，不再扩展该方法族。

## 5. 全局结果与主线裁决

### 5.1 五种输入构造的最终状态

冻结 held-out 为 697 records（223 suc、474 fail、254 video clusters），partition fingerprint 为 `4d65d16add4129b3afc0b16f24495189d47580e5688b7167b4921c0cd528fee2`。四次正式 held-out 都是 697 examples×10 conditions=`6970/6970` 有效、零缺失/重复/invalid；所有 verification status 均为 `ANALYZED`。

| 模型 | `text→images` | `images→text` | interleaved | native `text→video` | native `video→text` | 强确认计数 |
| --- | --- | --- | --- | --- | --- | ---: |
| RoboReward-8B | screening 失败 | screening 失败 | **H1 强确认通过** | screening 失败 | screening 失败 | **1/2** |
| Qwen3-VL-8B | **H2 确认性失败**（suc 下降） | **H3 强确认通过** | screening 失败 | screening 失败 | **H4 方向通过但证据不足** | **1/2** |

因此主线目标 1（文献、代码与方法谱系调研）已完成；主线目标 2 的严格“两模型×各两布局”验收 **未完成**。不能把 H4 升格来凑数：其 k8 MAE cluster CI 为 `[−0.0591,+0.0174]`、Holm-adjusted sign-flip `p=0.3636`；k64 Holm p 也为 `0.0654`。H4 三个 k 的点估计确实同时 MAE↓、suc↑、fail↑，但不满足本日志预先定义的全 k 稳健证据标准。

### 5.2 四次 held-out 的统一统计摘要

| 验证 | k | MAE baseline→target | suc accuracy | fail accuracy | MAE cluster 95% CI | Holm MAE p | 判定 |
| --- | ---: | --- | --- | --- | --- | ---: | --- |
| H1 RoboReward interleaved | 8 | 1.5968→1.3974 | 44.39→55.16% | 12.87→18.57% | [−0.2730,−0.1683] | 0.00090 | 强通过 |
|  | 32 | 1.5968→1.4103 | 44.39→54.71% | 12.87→17.72% | [−0.2592,−0.1545] | 0.00090 | 强通过 |
|  | 64 | 1.5968→1.3945 | 44.39→54.26% | 12.87→18.35% | [−0.2772,−0.1699] | 0.00090 | 强通过 |
| H2 Qwen `text→images` | 8 | 1.6686→1.5538 | 83.41→78.03% | 1.69→6.96% | [−0.2215,−0.1188] | 0.00090 | suc 反向，失败 |
|  | 32 | 1.6686→1.5581 | 83.41→78.48% | 1.69→7.17% | [−0.2175,−0.1158] | 0.00090 | suc 反向，失败 |
|  | 64 | 1.6686→1.5638 | 83.41→78.48% | 1.69→6.75% | [−0.2182,−0.1125] | 0.00090 | suc 反向，失败 |
| H3 Qwen `images→text` | 8 | 1.5222→1.4003 | 52.91→69.06% | 6.75→19.83% | [−0.1375,−0.0466] | 0.00090 | 强通过 |
|  | 32 | 1.5222→1.3816 | 52.91→74.44% | 6.75→21.10% | [−0.1522,−0.0535] | 0.00090 | 强通过 |
|  | 64 | 1.5222→1.3816 | 52.91→73.99% | 6.75→20.89% | [−0.1539,−0.0561] | 0.00090 | 强通过 |
| H4 Qwen native `video→text` | 8 | 1.4476→1.3888 | 73.99→84.75% | 11.60→23.63% | [−0.0591,+0.0174] | 0.36356 | CAUTION |
|  | 32 | 1.4476→1.3587 | 73.99→84.75% | 11.60→24.68% | [−0.0886,−0.0174] | 0.03120 | 通过 |
|  | 64 | 1.4476→1.3630 | 73.99→85.65% | 11.60→24.68% | [−0.0837,−0.0105] | 0.06539 | CAUTION |

H1 fingerprint `cf26ba9cbcb1e85ff275de98da0fa2bc43c88ee9703a129394f1186292678ca1`；H2 `b13797e38af1b31931a695ef799b4e02317e77bf547a418686f44d41cd9e8c57`；H3 `9fe59326e3351c65e541dff22fc44a3d147c4f8791ae9bb9b0898dd1fc15186c`；H4 `2695933427dd88d688a5215502051fdc46d7d278c83ee9b5cd10bd5c80085daf`。

### 5.3 task、pairwise、预测分布与 controls

task MAE 的“改善/持平/变差”计数与同视频 pairwise positive rate 如下，明确显示总体效应不等于逐 task 或逐 pair 普遍改善：

| 验证 | k8 tasks / pairwise | k32 tasks / pairwise | k64 tasks / pairwise | baseline pairwise |
| --- | --- | --- | --- | ---: |
| H1 | 20/4/4；67.43% | 19/4/5；65.60% | 20/3/5；66.51% | 60.82% |
| H2 | 19/0/9；71.98% | 20/0/8；71.98% | 20/1/7；72.21% | 64.01% |
| H3 | 17/5/6；66.97% | 15/9/4；67.20% | 16/8/4；68.34% | 66.51% |
| H4 | 10/6/12；64.46% | 11/6/11；66.29% | 11/6/11；66.29% | 65.15% |

分布证据支持 H1/H3 的双端点校准，而不是把所有样本推向同一端：H1 suc 的 label-5 从 99 增至 121–123、fail 的 label-1 从61增至84–88；H3 suc label-5 从118增至154–166、fail label-1 从32增至94–100。H4 同样把 suc label-5 从165增至189–191、fail label-1 从55增至112–117，但 k8 总 MAE 仍受中间类别和 task 异质性影响。H2 则把 fail label-1 从8增至32–34的同时令 suc label-5 从186降至174–175，正是不能忽略的 endpoint trade-off。

controls 的诚实结论：H1 与 H4 没有任何 wrong-region/low-rank k 复现三项严格方向；H2 controls 也未复现，但主配置本身失败。H3 的 `low_rank_target_k8` 在点估计上出现很小的三项同向变化（MAE 1.5222→1.5065、suc 52.91→53.36%、fail 6.75→8.86%），必须保留为 specificity caution；它没有在 k32/k64 复现，且效应远小于 target ranking，因此不推翻 H3 的全 k 主效应，但说明“所有 low-rank heads 都惰性”这一更强说法不成立。

### 5.4 ranking heads 与跨模型重合

下列为统一的 bidirectional weighted causal artifacts 的具体 top-8；括号内是冻结 multiplier，负号表示显式 `−6` profile 验证的反方向：

| 输入 | RoboReward top-8 | Qwen top-8 |
| --- | --- | --- |
| interleaved | L19H10(+1.000), L19H23(+0.637), L22H2(+0.310), L19H16(+0.388), L20H28(−0.292), L21H25(+0.319), L24H11(+0.311), L26H19(+0.273) | L19H17(+1.000), L22H2(+0.828), L19H10(+0.943), L23H28(+0.758), L16H14(+0.581), L19H16(+0.561), L12H21(+0.474), L20H20(+0.322) |
| `text→images` | L22H5(+0.984), L19H10(+1.000), L21H16(+0.929), L22H2(+0.780), L20H3(+0.566), L21H3(+0.610), L19H16(+0.587), L21H25(+0.537) | L22H2(+1.000), L19H17(+0.965), L19H10(+0.773), L19H14(+0.651), L24H11(+0.584), L15H25(+0.474), L23H25(+0.522), L12H0(+0.374) |
| `images→text` | L21H26(+1.000), L16H29(+0.601), L20H20(+0.438), L9H19(+0.432), L34H30(+0.401), L24H11(+0.271), L18H10(−0.301), L23H23(+0.170) | L21H26(+1.000), L22H2(+0.601), L19H17(+0.546), L15H25(+0.532), L21H31(+0.525), L20H4(+0.470), L25H30(+0.478), L13H24(+0.433) |
| native `text→video` | L22H5(+1.000), L19H17(+0.752), L19H10(+0.689), L22H2(+0.616), L20H3(+0.326), L22H26(+0.479), L20H1(−0.405), L26H19(+0.361) | L22H2(+1.000), L19H17(+0.791), L21H26(+0.799), L20H13(+0.569), L20H30(+0.646), L19H14(+0.588), L24H12(+0.629), L15H25(+0.457) |
| native `video→text` | L14H23(+1.000), L19H3(+0.920), L22H15(+0.714), L19H19(+0.709), L16H29(+0.652), L20H20(+0.510), L19H20(+0.503), L19H12(−0.461) | L21H26(+1.000), L19H17(+0.719), L12H19(+0.713), L22H2(+0.617), L20H13(+0.739), L14H23(+0.661), L21H31(+0.507), L10H12(+0.501) |

同一输入下的跨模型交集（交集数/Jaccard）为：

| 输入 | top-8 | top-32 | top-64 |
| --- | ---: | ---: | ---: |
| interleaved | 3 / 0.231 | 7 / 0.123 | 24 / 0.231 |
| `text→images` | 2 / 0.143 | 8 / 0.143 | 14 / 0.123 |
| `images→text` | 1 / 0.067 | 5 / 0.085 | 19 / 0.174 |
| native `text→video` | 2 / 0.143 | 3 / 0.049 | 18 / 0.164 |
| native `video→text` | 1 / 0.067 | 1 / 0.016 | 15 / 0.133 |

模型内部跨布局的平均交集同样不高：RoboReward top8/32/64 为1.4/4.8/15.3，Qwen为3.0/7.0/15.3。结论是中后层有少量反复出现的公共 heads（如 L22H2、L19H10/L19H17），但足以决定效果的 top ranks 高度依赖模型与输入顺序；跨布局或跨模型直接复用 head ranking 没有证据支持。

### 5.5 方法层面的最终理论结论

1. **原始 attention mass 不是 reward 因果效用。** raw/excess/enrichment、temporal、paired contrastive、显式 signed 与 exact-endpoint profiling 的排序差异很大；仅“看向目标”不能保证正确判断指令完成。
2. **输入顺序是实质机制，不是表面格式。** causal mask、视觉 token 到最终决策 query 的距离和 prefill 重写范围改变了有效 heads；H1/H3 分别成功而相邻布局失败，head overlap 也支持 layout specificity。
3. **主要瓶颈是双端点冲突与离散阈值。** 很多配置稳定降低 MAE，却只改善 suc 或 fail；teacher-forced margin 的小幅正效应也常在自由生成中停留于 label2/3，不能线性叠加为 exact label1/5。
4. **k 不是纯稳健性索引，而是剂量与组合变化。** safe heads 少于64时，padding multiplier 决定高 k 是否污染；冻结0.1 padding可修复部分 k64 崩坏，但不能凭空解决 endpoint trade-off。
5. **label-blind adaptive routing尚不可靠。** development leave-video-out accuracy 可达约88–91%，但 screening 的 branch transport 明显下降；移动阈值只能改善 ordinal score，不能稳定跨 exact endpoints。用 evaluation 标签调 router 会构成泄漏，因此被禁止。
6. **最可靠的改良是稀疏、layout-specific、显式双侧因果加权。** 它产生 H1/H3 两个强结果，并使 H4 方向一致；但当前证据不支持一种可在两模型多数布局普遍工作的 universal attention-steering recipe。

## 6. 全局完整性与统计谬误审计

### 6.1 完整性、复现性与 Material Passport

- 55 个 versioned screenings 均保留；crash/invalid 构建与 parser dropout 均在发生处记录，没有静默重试或删结果。
- 四个 held-out artifacts 都是 6970/6970 有效、254 clusters，并带稳定 fingerprint；partition、ranking、dose、scope 与 controls 在各 H run 前冻结。
- 未使用 GPU3；实验只在 GPU0/1/2，均有3小时 hard timeout。没有 git 操作，没有覆盖/删除既有数据、结果或配置。
- 正式16-config matrix 最近一次有效结果仍为 `5 passed`；之后两次无 pytest 入口的检查失败已记录，不被写成通过。
- 本报告 Verification Status 为 `ANALYZED` 而非 `VERIFIED`：完整性、配对统计和确定性 artifact 已核验，但没有把同一冻结推理重复一次冒充独立复现。Reproducibility verdict 为 `CANNOT_VERIFY`（缺少预先冻结的第二个独立 video-cluster cohort；机械重复同一确定性推理也不能提供外部复现）。

### 6.2 全局 11 类统计谬误扫描（11/11 checked）

1. **Simpson's paradox：CAUTION。** 所有 held-out 同时报告总体、suc/fail、task；H4 k8 有12个 task 变差，H2 suc 与总体 MAE方向相反，因此不作逐 task 普遍化。
2. **Ecological fallacy：未发现。** 推断单位是 example，重采样单位是 video cluster；不从 task 均值推断单条轨迹。
3. **Berkson's paradox：CAUTION。** cohort 受 auto-valid-grounding eligibility 选择，外部效度只限相同可定位样本；内部配对没有再按输出筛选。
4. **Collider bias：CAUTION。** grounding eligibility 可能同时关联任务难度与可定位性；没有按 baseline/target 输出控制 collider，也不外推到未通过 grounding 的视频。
5. **Base-rate neglect：未发现。** 明确保留223 suc/474 fail及两侧准确率；没有用总体 accuracy 掩盖类别权衡。
6. **Regression to the mean：未发现主问题。** held-out 未按极端 baseline 选择，每条记录有同输入 paired baseline；screening 与 ranking clusters 从 held-out 排除。
7. **Survivorship bias：held-out 未发现。** 四次均6970/6970；screening parser failures与builder crashes没有被静默删去或补跑。
8. **Look-elsewhere effect：CAUTION。** 探索含55个 screenings，因此不能把最优 screening 当确认；仅四个预冻结 H runs用于确认，九个主检验各自 Holm 校正，H2失败/H4弱证据完整保留。
9. **Garden of forking paths：CAUTION。** 方法分支很多，但配置、fingerprint、失败与终止规则逐项记录；H runs不回调。全局“两布局”尚未达标，故不做事后放宽验收。
10. **Correlation ≠ causation：限定通过。** 配对 hook 干预支持“该配置在该模型/布局/cohort改变输出”的模型内因果陈述；不支持现实机器人成功率、未定位样本或其他模型的外推。
11. **Reverse causality：未发现。** steering 在 generation 前施加；held-out标签只在生成完成后 join，不能反向决定 head、方向、权重、剂量或样本 branch。

全局 Overall Confidence 为 `CAUTION`：H1与H3在冻结 cohort 内证据强，H2是明确失败，H4是方向性候选但非强确认；外部效度和多路径探索仍受限。

## 7. 最终结论与停止边界

本轮 auto research 已完成文献调研、五种输入×两模型的统一机制分析、55轮 versioned screening、4次完整 held-out、task/pairwise/分布/controls/head-overlap与11/11完整性审计。最终获得两个可信但分属不同模型的强配置：

- RoboReward-8B：interleaved，H1；
- Qwen3-VL-8B：`images→text`，H3。

Qwen native `video→text` H4 是有价值的第二候选：三个 k 点估计都改善双端点与MAE，但 k8 MAE和k64多重校正证据不足，只能标 `ANALYZED / CAUTION`。RoboReward 的其他四布局与 Qwen 的其余布局，在 uniform/scope/dose、temporal、contrastive、signed/weighted、tail、balanced portfolio、adaptive/calibrated router、specialist、last-frame 与 exact-endpoint 等相互独立方法族下均未达到严格门。

所以，**严格主线结论是未达到“两模型各至少两种输入构造都在 k=8/32/64 稳健三指标提升”**。在现有 frozen partition 与无泄漏约束下继续通过调 threshold、bias、head 或重新解释 H4 来凑齐，会把研究变成 held-out-driven p-hacking、endpoint hard coding或验收标准后移；本日志明确拒绝这样做。

若未来继续，需要一个新的研究阶段和新的确认材料，而不是继续消费当前 held-out：建议预先冻结全新 video-cluster cohort，并在更大的独立 development set 上学习带不确定性校准的 head组合/路由，或实现真正的 attention-probability budget redistribution 后再做一次单次确认。这些属于后续研究建议，不是本轮已验证结论。
