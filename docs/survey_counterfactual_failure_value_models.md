# 调研报告:具身长程任务中的"目标物体错误"失败与 Value/Reward Model 的语义分辨能力

> 调研日期:2026-08-05
> 调研方式:基于公开文献的系统性网络检索(arXiv / ICLR / CVPR / CoRL / NeurIPS / ICCV / ACL 等),围绕五个主题展开:(1) 机器人操作失败检测与推理;(2) 长程任务失败分析与运行时监控;(3) 目标物体错误 / 反事实(counterfactual)/ 语义错位(semantic misalignment)失败;(4) 通用 value / reward / critic model;(5) training-free 的模型内部干预(attention/activation steering)方法。
> 本报告不参考本机任何既有调研或 proposal 文档,为独立调研结果。

---

## 1. 研究想法回顾

**核心假设**:在具身智能长程任务中,"目标物体错误"(instruction 指定物体 A,机械臂却操作了物体 B)是一类致命但隐蔽的失败——运动学上完全正常、无掉落等物理异常,仅语义上与指令不符。现有用于判断任务进度/成败的 value model / critic model(如 Robo-Dopamine GRM、GVL、VLAC 等)对这类"反事实错误"的分辨能力可能不足,会把"顺利地做错了任务"打出高进度分("成功幻觉"),进而导致长程任务级联崩溃。

**已开展的工作**:
1. 采集了大量 instruction-object 不匹配的真机遥操数据,构建数据集;
2. 提出一种 training-free 方法提升 value model 对指令语义的理解与跟随能力,已在 Robo-Dopamine GRM 上初步验证有效。

## 2. 总体结论(TL;DR)

**该方向可行,且有多条独立证据链支撑,处于研究窗口期。**

1. **问题真实存在且已被独立证实**:LIBERO-Plus(CVPR 2026)通过系统扰动实验发现 VLA 模型普遍"指令失明"(instruction blindness)——换掉指令中的目标物体后成功率跌至接近 0,模型实际退化为忽略语言的 Vision-Action 模型;"When Vision Overrides Language"(LIBERO-CF)将这种现象正式命名为 **counterfactual failures** 并构建了 policy 侧的第一个反事实基准。这说明"目标物体错误"不是长尾偶发,而是当前 VLA 的系统性失败模式,**检测它的监督器/critic 是刚需**。
2. **critic 侧的语义盲区已被初步指出、但远未解决**:I-FailSense 明确指出在 AHA 的七类失败中,六类是控制错误,只有 "Wrong Target Object" 属于语义错位,且是 **critical yet underexplored**;Guardian 的混淆矩阵显示 "wrong object state or placement" 是最容易被误判为 success 的类别;RoboReward 的评测(22 个 VLM、2831 条真机 episode)结论是"当前通用 VLM 尚不是可靠的 reward model",并且其训练集必须靠 **counterfactual relabeling**(同一视频换指令构造负样本)才能补上语义负例——这从侧面证明:**没有显式的反事实监督,reward model 学不会指令-物体对齐**。你的真机遥操 instruction-object 不匹配数据集正好命中这一供给缺口。
3. **training-free 干预路线有坚实的方法学先例、且在 reward/value model 上是空白**:PASTA(ICLR 2024)证明只需对少数 attention head 做注意力重加权即可显著提升 LLM 的指令跟随;"Your LVLM Only Needs A Few Attention Heads for Visual Grounding"(CVPR 2025)与 SparseMM(ICCV 2025)证明冻结 VLM 内部存在稀疏的 **localization heads / visual heads**(约 <5%,甚至 3 个头即可完成 grounding);机制可解释性工作进一步证明这些头对定位有因果作用。但检索范围内,**尚未发现任何针对机器人 reward/value model 的 attention-head 级 training-free 干预工作**(最接近的 VLS 是对 policy 的扩散采样过程做 steering,CAG/RSS 是对 policy 的 logits 做双分支对比)。你的"GRM attention head 识别 + steering"方法在这个交叉点上具有明确的新颖性。
4. **需要警惕的竞品与风险**:I-FailSense(2025.09)在做语义错位检测但走的是"训练 FS 分类头"路线且数据来自仿真/已有数据集改造;RoboReward、Robometer、VLAC 都已把语义负样本纳入训练数据(training-based);"The Hard Positive Truth"(ECCV 2024)警示纯 hard-negative 优化会导致过敏感(把正确改写也判为错误),你的方法与评测需要同时报告 hard-positive(同义改写指令)上的表现,证明提升不是以"见谁都判错"为代价。

## 3. 分主题综述

### 3.1 机器人操作失败检测与推理(critic 侧)

这一支从 REFLECT(CoRL 2023,LLM 基于多传感器分层摘要做失败解释,提出 RoboFail 数据集)开始,经 AHA(ICLR 2025,把失败检测形式化为 free-form reasoning,FailGen 程序化扰动 RLBench 成功轨迹生成 49K 失败样本,七类失败中含 Wrong Target Object)发展到 2025-2026 年的一批工作:RoboFAC(9,440 条错误轨迹 + 78K QA,分 task planning / motion planning / execution control 三层)、Guardian(RLBench-Fail / BridgeDataV2-Fail / UR5-Fail 三个新基准,多视角推理 VLM)、I-FailSense(专攻 semantic misalignment,在 VLM 中间层挂轻量分类头做集成仲裁)。

**关键观察**:
- 失败数据几乎全部来自**仿真程序化扰动**或对已有数据集的改造;真机失败数据规模极小(UR5-Fail 仅约 570 条执行失败样本)。真机遥操采集的语义失败数据是稀缺资源。
- "wrong object manipulated" 已成为该领域公认的失败类别(REFLECT/AHA/Guardian/RoboFAC 的 taxonomy 都包含),但**专门研究它的工作只有 I-FailSense 一篇**,且承认这类错误"can extend to broader forms of semantic misalignment"。
- Guardian 论文明确报告:执行类失败里 "wrong object state or placement is often mistaken for success" ——直接印证了你"看起来正常、实际错了"的动机。

### 3.2 长程任务失败与运行时监控

Sentinel(CoRL 2024)将生成式 policy 的失败拆成两类:erratic(动作时间不一致,用 STAC 统计量低成本检测)与 task progression failure(动作一致但不解决任务,必须用 VLM 做 video QA 检测)——你关注的"目标物体错误"恰属于后者:**运动统计完全正常,只有语义通道能发现**。SAFE(NeurIPS 2025)从 VLA 内部特征回归失败概率,配合 conformal prediction 给出报警阈值,但它检测的是"任务在失败"这一泛化信号,论文未针对语义错位类失败做分辨。DoReMi(2023)用 VLM 做 plan-execution 约束检测并触发重规划;RACER(ICRA 2025)用 VLM supervisor 提供富语言纠错指令。skill chaining 系列(Adversarial Skill Chaining、SCaR 等)从 RL 角度形式化了子任务误差沿链条累积导致长程崩溃的机制,为"子任务级失败必须尽早发现"提供理论动机。

**关键观察**:长程 pipeline 里的 verifier/monitor 位置已经就绪(Sentinel/DoReMi/RACER 都留了接口),缺的是一个**对语义错位敏感的进度/成败信号源**——这正是增强后的 GRM 可以扮演的角色。

### 3.3 目标物体错误 / 反事实失败(policy 侧成因)

- LIBERO-Plus(CVPR 2026):七维扰动系统评测 10 个 SOTA VLA,发现模型对语言扰动"最不敏感"的真相是**完全忽略语言**;把指令目标物体换掉后成功率近 0(Finding 7/8)。
- When Vision Overrides Language(LIBERO-CF + CAG):正式定义 counterfactual failures——视觉捷径压倒语言意图,模型默认执行场景中训练最充分的任务/最常见物体;提出 training-free 的双分支推理 CAG(语言条件 policy 与无语言 VA 分支做对比)缓解。
- Flatness Preserves Instruction Following、RSS(ACL 2026):分别从损失景观锐度和"视觉先验主导梯度"解释 instruction blindness 的成因,RSS 用"减去无条件分支"提取纯语义信号——与 CAG 同属"对比出语言的因果贡献"思想。

**关键观察**:policy 侧已确认会大量产生"目标物体错误"型 rollout(这保证了你的问题设定有持续的样本来源与现实意义);policy 侧的缓解方法(CAG/RSS)使用的**反事实对比/双分支思想可以迁移到 critic 侧**——例如比较 value(video, l_true) 与 value(video, l_counterfactual) 的 margin 来度量 critic 的语义敏感度,这在文献中尚无人系统做过。

### 3.4 通用 value / reward / critic model

- SuccessVQA(2023,DeepMind):最早把成功检测形式化为 VQA,发现 VLM 比 bespoke reward model 更抗分布外变化。
- GVL(ICLR 2025):冻结 VLM 通过 shuffled-frame 自回归预测任务完成百分比;shuffle 的动机本身就是为了打破时序捷径——证明 VLM 做进度估计时**倾向输出与内容无关的单调值**。
- Robo-Dopamine GRM(CVPR 2026):本仓库,任务描述 + 初始/目标/前后状态多视角图像 → 相对进度增量,多视角进度融合(incremental/forward/backward 三模式)。
- VLAC(InternVL 系,2025):pairwise critic,输出带符号的 progress delta;其训练显式构造了大量 **negative and semantically mismatched samples** 以"拒绝无关 prompt、识别倒退与停滞"——再次证明语义负样本对 critic 是必需品。
- RoboReward / RoboRewardBench(2026):22 个 VLM 的系统评测,结论"no model excels in all tasks / 通用 VLM 尚不可靠";训练集用 counterfactual relabeling + temporal clipping 造负例;8B 模型即超 GPT-5/Gemini。
- Robometer(2026):进度监督 + 轨迹间偏好监督双目标,RBM-1M(百万级、含大量失败/次优轨迹);评测中直接将 Robo-Dopamine 作为 baseline 之一。
- ReWiND(CoRL 2025):video rewinding 造合成失败 + **对 misaligned video-language pair 强制输出 0 进度**——同样是显式语义负监督。
- 其他:LRM(在线 dense reward)、RARM(置信门控进度 reward,指出 VLM reward "prone to false positives on visually plausible but physically incorrect states")、T2-VLM(training-free 时序一致 reward,Bayesian 子目标跟踪)、ROVER(递归分解降低长视频推理幻觉)、时空幻觉分析(GPT-4o 在两帧顺序反转下判断崩溃,证明"sequence bias"先验)。

**关键观察**:
- 整个 reward model 社区在 2025-2026 年集中转向"如何补语义负样本"(counterfactual relabeling / mismatched pairs / rewinding),但全部走 **training-based** 路线,且负样本几乎全部是**合成的**(换指令文本、倒放视频),鲜有真机遥操采集的自然反事实轨迹。
- 尚无公开基准**专门**度量 reward/value model 对"指令-物体不匹配"的分辨能力。RoboRewardBench 度量整体 MAE;Robometer 的 confusion matrix 评测最接近,但仍是 success/fail 粗粒度。**"critic 侧的 LIBERO-CF"仍然空缺。**

### 3.5 Training-free 内部干预方法(方法学基础)

- PASTA(ICLR 2024):一次性 profiling 找出少数 attention head,推理时对用户强调的 prompt 片段做注意力重加权,LLaMA-7B 平均提升 22%,零训练零延迟。
- Your LVLM Only Needs A Few Attention Heads for Visual Grounding(CVPR 2025):冻结 LVLM 中存在 localization heads(用 attention sum + 空间熵两个准则筛选),**仅 3 个头**的 text-to-image attention map 即可达到与微调方法可比的 grounding 性能。
- SparseMM(ICCV 2025):<5% 的 head 承担视觉理解(visual heads),提供 training-free 的头级视觉相关性量化框架。
- Mechanisms of Object Localization in VLMs(2026):因果中介分析证明极少数 head 因果地承担定位,且"先识别物体、再空间定位"是串行机制。
- 相关但需区分的:VLS(2026,training-free 但 steering 的是 diffusion policy 的去噪采样,不是 critic 内部);CAG/RSS(training-free 双分支 logits 对比,作用于 policy)。

**关键观察**:头级稀疏性 + 注意力可操控性这两个前提在通用 VLM 上已被反复证实,而 GRM 本身就是 VLM(RoboBrain-2.0/Qwen-VL 系),你的方法(在 GRM 中定位 instruction/object grounding heads 并 steering)是这条方法学在 reward modeling 领域的**首次落地**;仓库中已有的 `rank_grm_heads_by_comics.py`、`steer_grm_heads.py`、`scan_localization_heads_best.py` 等脚本与该文献路线完全同构,可直接引用这批工作作为机制依据。

## 4. 相关工作总结表

分类图例:**A** = 失败检测/推理(critic 侧);**B** = 长程任务监控/恢复;**C** = 反事实/指令跟随失败(policy 侧);**D** = value/reward/critic model;**E** = training-free 干预/机制可解释性;**F** = 语义组合性(远缘)。
"是否支撑"含义:该工作的结论是否为本研究想法(问题重要性、方法可行性或数据集价值)提供支撑。

| # | 文章标题 | 来源 | 一句话总结 | 分类 | 是否支撑 |
|---|---------|------|-----------|------|---------|
| 1 | REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction | CoRL 2023 (arXiv:2306.15724) | LLM 基于多传感器分层摘要做失败定位/解释/纠正,提出 RoboFail 数据集 | A/B | 支撑(奠定失败分析范式与 taxonomy) |
| 2 | AHA: A VLM for Detecting and Reasoning Over Failures in Robotic Manipulation | ICLR 2025 (arXiv:2410.00371) | FailGen 程序化扰动成功轨迹造 49K 失败数据,微调 VLM 做 free-form 失败推理;七类失败含 Wrong Target Object | A | 支撑(证明失败数据稀缺、程序化负样本有效;但真机语义失败数据仍空缺) |
| 3 | Guardian: Detecting Robotic Planning and Execution Errors with VLMs | arXiv:2512.01946 | 自动失败合成 + 多视角推理 VLM,建 RLBench-Fail/BridgeDataV2-Fail/UR5-Fail 三基准 | A | 强支撑(混淆矩阵显示 wrong object 类最易被误判为 success) |
| 4 | I-FailSense: Towards General Robotic Failure Detection with VLMs | arXiv:2509.16072 | 专攻 semantic misalignment 失败,VLM 中间层挂 FS 分类头集成仲裁 | A | 强支撑(明确"语义错位是 critical yet underexplored";亦是最直接竞品,需差异化:其为训练法+仿真数据) |
| 5 | RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction | arXiv:2505.12224 | 9,440 条错误轨迹 + 78K QA 的失败分析/纠正数据集与 7B critic 模型 | A/B | 支撑(证明外挂 critic 可提升真机 VLA 成功率 29.1%) |
| 6 | SAFE: Multitask Failure Detection for Vision-Language-Action Models | NeurIPS 2025 (arXiv:2506.09937) | 用 VLA 内部特征回归任务失败概率,conformal prediction 定阈值,跨任务泛化 | A/B | 支撑(证明内部表征含成败信息;但未区分语义错位类失败,留下空间) |
| 7 | Unpacking Failure Modes of Generative Policies (Sentinel) | CoRL 2024 (arXiv:2410.04640) | 失败分为 erratic(统计检测)与 task progression(VLM 检测)两通道并行监控 | B | 支撑(证明"运动正常但任务不对"的失败只有语义通道能捕捉) |
| 8 | DoReMi: Detecting and Recovering from Plan-Execution Misalignment | arXiv:2307.00329 | LLM 生成约束 + VLM 持续检测违例,触发即时重规划 | B | 支撑(长程闭环中 verifier 的系统位置) |
| 9 | RACER: Rich Language-Guided Failure Recovery Policies | ICRA 2025 (arXiv:2409.14674) | VLM supervisor 输出富语言纠错指令引导 actor 恢复 | B | 部分支撑(证明失败恢复价值;其 VLM 仍会幻觉误判失败) |
| 10 | Adversarial Skill Chaining via Terminal State Regularization | CoRL 2021 (arXiv:2111.07999) | 子任务终态偏移沿技能链级联放大导致长程失败的形式化与正则化 | B | 支撑(长程误差累积的理论动机) |
| 11 | SCaR: Refining Skill Chaining via Dual Regularization | NeurIPS 2024 | 双正则化增强子任务内/间依赖,缓解链式误差累积 | B | 支撑(同上,背景) |
| 12 | LIBERO-Plus: In-depth Robustness Analysis of VLA Models | CVPR 2026 (arXiv:2510.13626) | 七维扰动评测 10 个 VLA:模型普遍忽略语言,换目标物体后成功率近 0 | C | 强支撑(系统证实"指令失明"与目标物体错误的普遍性) |
| 13 | When Vision Overrides Language: Counterfactual Failures in VLAs (LIBERO-CF + CAG) | arXiv:2602.17659 | 定义 counterfactual failures,建 policy 侧首个反事实基准,training-free 双分支 CAG 缓解 | C | 强支撑(概念与基准范式可迁移到 critic 侧;critic 侧对应物仍空缺) |
| 14 | Flatness Preserves Instruction Following in VLA Models | arXiv:2606.23641 | 从损失景观锐度解释 instruction blindness,SAM 优化缓解 | C | 支撑(成因分析) |
| 15 | Stable Language Guidance for VLA Models (RSS) | ACL 2026 | 减去无语言条件分支的"视觉本能先验"以提取纯语义信号 | C | 支撑(反事实对比思想同源) |
| 16 | Vision-Language Models as Success Detectors (SuccessVQA) | CoLLAs 2023 (arXiv:2303.07280) | 把成功检测形式化为 VQA,VLM 比 bespoke reward model 更抗 OOD | D | 支撑(奠基;其语言变体泛化实验是先例) |
| 17 | Vision Language Models are In-Context Value Learners (GVL) | ICLR 2025 (arXiv:2411.04549) | 冻结 VLM 经 shuffled-frame 自回归预测任务进度,300+ 真实任务零/少样本 | D | 支撑(证明 VLM 进度估计有时序捷径偏置,需干预) |
| 18 | Robo-Dopamine: General Process Reward Modeling (GRM) | CVPR 2026 (arXiv:2512.23703) | 多视角前后状态 + 指令 → 进度增量的通用过程奖励模型,多视角进度融合 | D | 支撑(本研究的基座平台) |
| 19 | VLAC: A Vision-Language-Action-Critic Model for Real-World RL | arXiv:2509.15937 | pairwise progress critic 统一 actor/critic;显式构造大量语义不匹配负样本训练 | D | 强支撑(证明 critic 必须补语义负监督;其为 training-based 路线) |
| 20 | RoboReward: General-Purpose Vision-Language Reward Models for Robotics | arXiv:2601.00675 | 22 个 VLM 的 reward 评测(RoboRewardBench)+ counterfactual relabeling 造负样本训练 4B/8B 模型 | D | 强支撑("通用 VLM 尚不可靠"+ 反事实重标注证明语义负例必要;但其负例为合成、评测非语义专项) |
| 21 | Robometer: Scaling Robotic Reward Models via Trajectory Comparisons | arXiv:2603.02115 | 进度监督 + 轨迹偏好监督双目标,RBM-1M 百万轨迹(含大量失败) | D | 支撑(把 Robo-Dopamine 列为 baseline;确立 reward model 评测生态) |
| 22 | ReWiND: Language-Guided Rewards Teach Robot Policies without New Demos | CoRL 2025 (arXiv:2505.10911) | video rewinding 合成失败 + 对 misaligned video-language 对强制 0 进度 | D | 支撑(语义错位负监督的又一独立证据) |
| 23 | RARM: Confidence-Gated Progress Reward Modeling | arXiv:2606.22027 | 置信门控进度 reward;指出 VLM reward 对"视觉合理但实际错误"状态易假阳性 | D | 支撑(直接指出成功幻觉问题) |
| 24 | Training-Free Generation of Temporally Consistent Rewards from VLMs (T2-VLM) | ICCV 2025 | VLM 子目标分解 + Bayesian 跟踪,training-free 时序一致 reward | D | 部分支撑(training-free reward 先例,但干预在 pipeline 层非模型内部) |
| 25 | Large Reward Models (LRM) | arXiv:2603.16065 | Qwen3-VL 微调为在线逐帧 dense reward 生成器,24 数据源 | D | 背景(生态;training-based) |
| 26 | ROVER: Recursive Reasoning Over Videos | 2026 (RAI Institute) | 递归子任务分解降低长视频进度估计中的幻觉,复杂度线性化 | D | 部分支撑(长程进度估计幻觉的缓解思路) |
| 27 | A Progressive Training Strategy to Counteract Spatio-Temporal Hallucinations | arXiv:2604.10506 | 揭示 VLM 判断两帧任务进度时依赖输入顺序先验("multi-image reasoning hallucination") | D | 支撑(GRM 类两帧比较范式的已知偏置,评测需控制) |
| 28 | Tell Your Model Where to Attend (PASTA) | ICLR 2024 (arXiv:2311.02262) | 一次性 profiling 选头 + 推理时注意力重加权,零训练提升 LLM 指令跟随 22% | E | 强支撑(head-level training-free steering 的方法学奠基) |
| 29 | Your LVLM Only Needs A Few Attention Heads for Visual Grounding | CVPR 2025 (arXiv:2503.06287) | 冻结 LVLM 中 3 个 localization heads 的注意力图即可完成竞争级 visual grounding | E | 强支撑(GRM 内定位 grounding heads 的直接依据) |
| 30 | SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs | ICCV 2025 | <5% 的 attention head 承担视觉理解,training-free 头级视觉相关性量化 | E | 支撑(头稀疏性普适证据) |
| 31 | Mechanisms of Object Localization in Vision-Language Models | arXiv:2605.19792 | 因果中介分析:极少数 head 因果承担定位,识别→定位串行机制 | E | 支撑(head 干预的因果合法性) |
| 32 | VLS: Steering Pretrained Robot Policies via VLMs | arXiv:2602.03973 | training-free 用 VLM 生成可微奖励引导冻结扩散 policy 的去噪采样 | E | 部分支撑(training-free steering 思想;但作用于 policy 采样,非 critic 内部——差异化点) |
| 33 | When and Why VLMs Behave like Bags-of-Words (ARO / NegCLIP) | ICLR 2023 (arXiv:2210.01936) | CLIP 类模型组合语义近似词袋;组合感知 hard negatives 微调显著改善 | F | 支撑(语义盲区的根因远缘证据) |
| 34 | The Hard Positive Truth about Vision-Language Compositionality | ECCV 2024 (arXiv:2409.17958) | 纯 hard-negative 微调使模型"过敏感",需 hard positives 平衡 | F | 部分支撑(重要警示:评测与方法必须含同义改写正例) |
| 35 | UR5-Fail / RLBench-Fail 数据集(Guardian 配套) | HuggingFace (paulpacaud) | 失败类别显式含 "wrong object manipulated";真机部分仅数百样本 | A | 支撑(taxonomy 先例 + 真机语义失败数据的稀缺性证据) |

## 5. 可行性判断

### 5.1 支撑面(为什么可行)

1. **问题重要性有独立多方证据**:policy 侧(LIBERO-Plus、LIBERO-CF)证明反事实失败普遍;critic 侧(Guardian、RARM、RoboReward)证明现有评估器对这类失败易出假阳性;系统侧(Sentinel、skill chaining 系列)证明长程任务中此类失败若不被捕捉将级联放大。三条线索在"需要一个语义敏感的进度/成败评估器"上汇聚。
2. **数据集贡献有明确缺口**:现有失败数据 95% 以上来自仿真程序化扰动或合成重标注(FailGen、Guardian、RoboReward counterfactual relabeling、ReWiND rewinding);真机遥操自然采集的 instruction-object 不匹配轨迹(带真实的机械臂动力学、光照、遮挡、多视角)在公开资源中几乎不存在(最接近的 UR5-Fail 仅数百条且非专门针对语义错位)。
3. **方法新颖性初步成立**:training-free 干预已有成熟方法学(PASTA/localization heads/SparseMM),但均止步于通用 VLM 的 QA/grounding 任务;reward/value model 的语义增强现有方案全部 training-based(I-FailSense、VLAC、RoboReward、Robometer、ReWiND)。"training-free 提升 reward model 指令跟随"这一交叉点检索范围内没有先例。
4. **基座与评测生态成熟**:GRM 开源可复现;GVL/VLAC/RoboReward/Robometer/GPT-5/Gemini 可作为横向对比;RoboRewardBench/Robometer 的评测协议(MAE、VOC、Kendall τ、success-fail margin)可直接复用并扩展语义维度。

### 5.2 风险面(需要正面处理)

1. **与 I-FailSense 的区分**:必须强调 (a) 真机遥操自然失败数据 vs 其仿真改造数据;(b) training-free vs 其 LoRA+分类头;(c) 进度(process reward)粒度 vs 其 episode 级二分类;(d) 长程子任务场景 vs 其单任务。
2. **过敏感风险**(Hard Positive Truth 的教训):放大指令敏感度可能把同义改写、指代变体也判为不匹配。数据集与评测必须包含 hard positive(同一行为、改写指令)对照组;方法上 steering 强度需可调并报告 trade-off 曲线。
3. **时序/顺序偏置混淆**:GRM 是两帧比较范式,已知存在 sequence bias(arXiv:2604.10506)。评测语义分辨力时需固定时序因素(同一视频、只换指令),否则语义效应与时序效应混淆。
4. **training-free 的天花板**:若 head steering 的提升幅度有限,应准备"training-free 方法 + 少量数据微调"的混合消融,证明两者互补(steering 可作为免训练即插即用方案,数据集可作为进一步微调的燃料——两个贡献互为备份)。

## 6. 研究空白与机会(gap analysis)

| 空白 | 现状 | 机会 |
|------|------|------|
| G1: critic 侧反事实基准 | LIBERO-CF 只测 policy;RoboRewardBench 只测整体 MAE | 建立首个专测 value/reward model "指令-物体错位"分辨力的真机基准与指标(如 counterfactual value margin) |
| G2: 真机自然语义失败数据 | 失败数据几乎全为仿真扰动/合成重标注 | 真机遥操 instruction-obj 不匹配数据集(你已有) |
| G3: reward model 的 training-free 语义增强 | 全部 training-based | attention head 识别 + steering(你已初验) |
| G4: 语义敏感 verifier 在长程闭环的价值量化 | Sentinel/DoReMi 用通用 VLM 做 QA,弱语义分辨 | 用增强后的 GRM 做子任务 verifier,端到端量化长程成功率收益 |

## 7. 结论

想法的三个组成部分——(1) 目标物体错误是长程任务的致命隐蔽失败、(2) 现有 value model 分辨力不足、(3) training-free 语义增强可行——分别得到 [强、强、中强] 的文献支撑,且组合后的完整故事(真机反事实数据集 + critic 侧基准 + training-free 方法 + 长程闭环验证)在检索范围内无直接重合的先行工作。主要竞争压力来自 I-FailSense(语义错位检测)与 RoboReward/Robometer(reward 评测生态),建议以"真机数据 + training-free + 过程级(process-level)+ 长程闭环"四个差异化维度构筑贡献。

基于本报告,三个具体 proposal 见:
- `proposal_1_counterfactual_grm_bench.md` — 基准+数据集+系统评测(风险最低,发表面最宽)
- `proposal_2_grounding_head_steering.md` — 机制分析+training-free 方法(方法贡献最锐利)
- `proposal_3_semantic_verifier_longhorizon.md` — 长程闭环系统(应用价值最高)

## 8. 主要参考文献(按分类)

**失败检测/推理**:REFLECT (arXiv:2306.15724); AHA (arXiv:2410.00371); Guardian (arXiv:2512.01946); I-FailSense (arXiv:2509.16072); RoboFAC (arXiv:2505.12224); SAFE (arXiv:2506.09937)
**长程监控/恢复**:Sentinel (arXiv:2410.04640); DoReMi (arXiv:2307.00329); RACER (arXiv:2409.14674); Adversarial Skill Chaining (arXiv:2111.07999); SCaR (NeurIPS 2024)
**反事实/指令跟随失败**:LIBERO-Plus (arXiv:2510.13626); When Vision Overrides Language / LIBERO-CF+CAG (arXiv:2602.17659); Flatness Preserves Instruction Following (arXiv:2606.23641); RSS (ACL 2026)
**value/reward model**:SuccessVQA (arXiv:2303.07280); GVL (arXiv:2411.04549); Robo-Dopamine GRM (arXiv:2512.23703); VLAC (arXiv:2509.15937); RoboReward (arXiv:2601.00675); Robometer (arXiv:2603.02115); ReWiND (arXiv:2505.10911); RARM (arXiv:2606.22027); T2-VLM (ICCV 2025); LRM (arXiv:2603.16065); ROVER (2026); 时空幻觉分析 (arXiv:2604.10506)
**training-free 干预/机制**:PASTA (arXiv:2311.02262); Localization Heads (arXiv:2503.06287); SparseMM (ICCV 2025); Mechanisms of Object Localization (arXiv:2605.19792); VLS (arXiv:2602.03973)
**组合语义**:ARO/NegCLIP (arXiv:2210.01936); The Hard Positive Truth (arXiv:2409.17958)

*本报告由 AI 辅助调研工具生成,所有文献均经网络检索核实存在;结论以引用文献摘要与正文片段为据,建议在撰写正式论文前对关键数值做原文复核。*
