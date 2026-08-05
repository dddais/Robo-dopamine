# Proposal 2: Grounding-Head Steering — 面向过程奖励模型的 Training-Free 指令语义增强

> 类型:机制分析 + training-free 方法(主方法论文)
> 目标会议:ICLR / NeurIPS / CVPR(方法 track)
> 依托资产:已初验的 GRM attention head 识别与 steering 方法(仓库内 `rank_grm_heads_by_comics.py`、`steer_grm_heads.py`、`scan_localization_heads_best.py` 等脚本);Proposal 1 的数据集(或其子集)作为评测平台

---

## 1. 一句话故事

**过程奖励模型内部存在一小撮"指令-物体 grounding heads";它们在预训练中已学会绑定指令与目标物体,却在进度回归输出中被时序/运动先验淹没——我们以零训练的注意力干预把它们"重新接回"输出,使 reward model 学会分辨"做对了"与"顺利地做错了"。**

## 2. 动机

### 2.1 问题侧

Value/reward model 对"目标物体错误"型失败分辨力弱(证据:Guardian 混淆矩阵、RARM 的假阳性观察、RoboReward"通用 VLM 不可靠"结论、以及我们在 GRM 上的初步实验)。现有修复路线全部 training-based:

- I-FailSense(arXiv:2509.16072):LoRA 后训练 + 中间层分类头;
- VLAC(arXiv:2509.15937):训练时构造大量语义不匹配负样本;
- RoboReward(arXiv:2601.00675):counterfactual relabeling 扩充训练集;
- ReWiND(arXiv:2505.10911):对 misaligned 对强制 0 进度。

training-based 路线的代价:需要大规模负样本工程、每个新基座重训、可能损害已校准的进度回归能力(灾难性遗忘),且无法即插即用地服务社区已有的各类 reward model checkpoint。

### 2.2 机制侧(方法为什么应该有效)

三条已确立的机制证据表明"能力在模型里,只是没被用上":

1. **稀疏 grounding heads 存在**:冻结 LVLM 中仅 3 个 attention head 的 text-to-image 注意力图即可完成竞争级 visual grounding(CVPR 2025, arXiv:2503.06287);<5% 的 head 承担视觉理解(SparseMM, ICCV 2025);因果中介分析证实极少数 head 因果地承担"识别→定位"(arXiv:2605.19792)。GRM 基座(RoboBrain-2.0 / Qwen-VL 系)继承这些结构。
2. **注意力可低成本操控**:PASTA(ICLR 2024)证明只对选定 head 重加权注意力即可零训练提升指令跟随 22%;
3. **失败的成因是"先验压倒语义"而非"语义缺失"**:policy 侧的 CAG(arXiv:2602.17659)与 RSS(ACL 2026)证明减去无语言分支即可恢复语言的因果贡献——同样的"视觉/时序先验主导"病理在 critic 侧表现为:GRM 的进度输出主要由运动流畅度与时序单调性驱动(GVL 已证明 VLM 进度估计有强时序捷径,arXiv:2411.04549),指令 token 的贡献被稀释。

**核心假设**:GRM 在进度回归微调后,其 grounding heads 仍保留指令-物体绑定能力(可由注意力图验证),但这些 head 对最终 value token 的贡献权重不足;放大它们(或以它们的对齐度门控输出)即可恢复语义敏感性,无需任何梯度更新。

## 3. 方法(三个递进模块,均 training-free)

### 3.1 模块 A:Grounding Head 识别(一次性 profiling)

在少量校准数据(几十条匹配/不匹配对)上,对每个 (layer, head) 计算:

- **对齐分**:指令中目标物体 token → 图像 patch 的注意力质量(结合 GT bbox 用 attention-in-box 比例,类似仓库现有 `rank_heads_by_bbox.py` 的做法;无 bbox 时用空间熵 + attention sum 双准则,同 arXiv:2503.06287);
- **判别分**:该 head 的注意力模式在匹配 vs 反事实对上的可分性(如注意力图的 JS 散度)。

两分数联合排序,选出 top-k grounding heads。产出副产品:GRM 的 head-level 机制图谱(本身就是分析贡献)。

### 3.2 模块 B:注意力 Steering(推理时干预)

对选定 heads,在前向中将指令物体 token 相关的注意力按系数 α 重加权(PASTA 式乘性缩放 + 重归一化),把"模型看到了什么"重新对齐到"指令要求什么"。α 可调,给出敏感度-特异度 trade-off 曲线。

### 3.3 模块 C:反事实一致性校验(推理时集成,可选)

双前向:V(video, l_true) 与 V(video, l_null/l_perturbed)。定义 semantic margin 作为语义置信度,用于 (a) 门控/校正 value 输出,(b) 直接作为 wrong-object failure 检测分数。这是 CAG 双分支思想在 critic 侧的首次实例化,与模块 B 正交可叠加。

### 3.4 方法性质

- 零参数更新、零训练数据(校准集只用于选 head,量级几十条);
- 即插即用:同一流程可应用于 GRM 之外的开源 reward model(VLAC、RoboReward、Robometer),支撑"通用性"主张;
- 计算开销:模块 B 免费,模块 C 增加一次前向(可只在关键帧触发)。

## 4. 实验设计

### 4.1 主实验:语义分辨力

在 CF-GRM-Bench(Proposal 1)或其先行子集上,报告 CVM / Hallucination Rate / Semantic AUROC / Paraphrase Robustness:

- Baselines:原始 GRM、prompt 工程(CoT 描述物体)、GPT-5/Gemini zero-shot、I-FailSense(training-based 参照)、随机 head steering(证明选头非平凡)、全 head 均匀放大(证明稀疏性必要);
- Ours:A+B、A+B+C 消融。

### 4.2 通用性实验

同一方法应用到 VLAC / RoboReward / Robometer 开源 checkpoint,验证 grounding head 稀疏性与 steering 有效性跨基座成立(哪怕效果有梯度,也是重要发现)。

### 4.3 无损性实验

Robo-Dopamine-Bench 与 RoboRewardBench 原协议:证明 steering 后正例进度估计(VOC/MAE)不退化——回应"过敏感"风险(The Hard Positive Truth 的教训)。

### 4.4 下游价值实验(至少其一)

- **失败检测**:以 semantic margin 做 wrong-object 检测器,与 SAFE/Sentinel-VLM 通道对比 AUROC 与检测延迟;
- **RL/数据过滤**:在 Dopamine-RL 或数据筛选管线中替换原始 GRM,验证语义增强的 reward 是否提升下游任务成功率或数据质量(对齐 GVL/Robometer 的下游应用协议)。

### 4.5 机制验证实验(论文深度)

- head 消融的因果检验:knock out top grounding heads → 语义分辨力应显著下降而运动进度估计基本保持;
- 注意力可视化:steering 前后指令物体 token 的注意力热图对比(仓库已有 `visualize_stage3_head_attention.py` 可复用);
- 层级分布分析:grounding heads 的 layer 分布与文献(LLaVA early-mid vs InternVL mid-late, arXiv:2605.19792)对照。

## 5. 贡献点

1. **C1(发现)**:首次揭示过程奖励模型的"语义盲区"机制——grounding 能力在内部注意力中存在但未传导至 value 输出,并给出 head-level 因果证据;
2. **C2(方法)**:首个针对 reward/value model 的 training-free 语义增强框架(head 识别 + 注意力 steering + 反事实一致性校验),零训练、即插即用、跨基座通用;
3. **C3(实证)**:在真机反事实数据上系统验证方法有效且无损(进度估计能力保持、hard positive 不误伤),并展示失败检测/下游 RL 的应用价值;
4. **C4(资源)**:开源 head 图谱、steering 工具库与评测代码。

## 6. 风险与备选

- **风险 1:steering 提升幅度不够大** → 模块 C(margin 校验)独立于 B 仍可作为检测器主打;或转向"steering 定位问题 + 少量数据高效微调(只调 grounding heads 相关参数)"的混合方案,故事改为"机制引导的参数高效修复";
- **风险 2:grounding heads 在 GRM 微调中已退化** → 这本身是重要负结果:说明进度回归微调破坏了基座 grounding(可与原始 RoboBrain 基座对照定量),故事转为"reward 微调的语义遗忘"分析 + 数据侧修复(接 Proposal 1 的微调 upper bound);
- **风险 3:审稿人质疑只在 GRM 上成立** → 4.2 的跨基座实验是必答题,至少覆盖 2 个额外开源 reward model;
- **风险 4:与 PASTA/localization heads 的增量质疑** → 差异:(a) 任务从 QA/grounding 变为进度回归,head 选择准则需引入判别分(3.1);(b) 反事实一致性校验模块无先例;(c) 应用域(robot reward modeling)与下游验证(RL/失败检测)全新。

## 7. 时间线(约 14 周)

| 阶段 | 内容 |
|------|------|
| W1-2 | head 识别准则定稿与图谱构建(GRM 4B/8B) |
| W3-5 | steering 实现与超参搜索;CF 子集主实验 |
| W6-7 | 反事实一致性模块;消融矩阵 |
| W8-9 | 跨基座通用性(VLAC/RoboReward/Robometer) |
| W10-11 | 无损性 + 下游实验 |
| W12-14 | 机制章节、可视化、撰写 |
