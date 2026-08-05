# Proposal 1: CF-GRM-Bench — 面向 Value/Reward Model 的真机反事实失败基准与系统评测

> 类型:数据集 + 基准 + 系统实证分析(附一个轻量方法作为 upper-bound 演示)
> 目标会议:CoRL / ICRA / RSS(数据集与基准 track),或 NeurIPS Datasets & Benchmarks
> 依托资产:已采集的真机遥操 instruction-object 不匹配数据集;Robo-Dopamine GRM 及其评测代码;已初验的 training-free 方法

---

## 1. 一句话故事

**机器人 reward/value model 会给"顺利地做错任务"打高分——我们构建第一个真机反事实失败基准,系统量化这种"成功幻觉"(semantic success hallucination),并证明它可以被检测与缓解。**

## 2. 动机与问题陈述

长程操作任务中,子任务失败不可避免。物理型失败(掉落、碰撞、抓空)有明显的视觉/动力学特征,现有失败检测器可以较好捕捉;但**目标物体错误**——指令要求操作物体 A,机械臂却以完全正常、流畅的动作操作了物体 B——不产生任何物理异常,只在语义层面与指令冲突。这类失败:

1. **普遍**:LIBERO-Plus(CVPR 2026)与 LIBERO-CF(arXiv:2602.17659)证明 SOTA VLA 存在系统性"指令失明",在反事实指令下成功率趋零且执行的是"看起来很成功"的错误任务;
2. **隐蔽**:Guardian(arXiv:2512.01946)的混淆矩阵显示 wrong-object 类失败最常被 VLM 判为 success;Sentinel(CoRL 2024)证明这类"动作一致但任务不对"的失败无法用动作统计量检测;
3. **致命**:skill chaining 文献证明子任务级错误沿长程链条级联放大;若 value model 在错误物体上持续报告进度上升,下游的 replan/recovery/RL 全部失效。

然而,**没有任何公开基准专门度量 reward/value model 对这类失败的分辨能力**。RoboRewardBench(arXiv:2601.00675)度量整体 reward MAE;Robometer(arXiv:2603.02115)度量 success/fail 混淆;两者的负样本均为合成(同视频换指令文本、视频倒放),缺乏真实机器人"自然地做错"的轨迹分布。

## 3. 研究问题

- **RQ1**:当前的通用与专用 value/reward model(GRM、GVL、VLAC、RoboReward、Robometer、GPT-5/Gemini 级通用 VLM)在真机"目标物体错误"轨迹上的分辨能力如何?"成功幻觉"率有多高?
- **RQ2**:分辨失败的模式是什么——是感知问题(没看到物体差异)、grounding 问题(看到了但没绑定到指令)、还是先验问题(时序单调偏置压倒语义证据)?
- **RQ3**:哪些低成本手段(prompt 工程、反事实 margin 探针、training-free steering、少量数据微调)能以什么代价缓解?

## 4. 提出的基准:CF-GRM-Bench

### 4.1 数据集(核心资产,已部分完成)

真机遥操采集,每组数据为一个 **counterfactual triplet**:

- **匹配正例** (video, l_true):指令与实际操作物体一致的成功轨迹;
- **反事实负例** (video, l_cf):同一场景/同类动作,操作了非指令物体的完整流畅轨迹(遥操作员刻意执行,保证运动学自然——这是与程序化扰动数据的本质区别);
- **hard positive** (video, l_para):同一视频配同义改写指令(换称谓、换句式、加修饰),用于检测过敏感(借鉴 The Hard Positive Truth, ECCV 2024 的教训)。

维度设计(每维度可控消融):物体类别距离(同类不同色 / 不同类)、场景干扰物数量、单臂/双臂、子任务位置(长程任务的第 k 步)、视角(高位/腕部)。规模目标:≥50 任务 × ≥10 场景,数千条轨迹,全部人工标注失败类型与关键帧。

### 4.2 评测协议与指标

固定视频、只换指令,消除时序混淆(应对 GRM 类两帧比较范式已知的 sequence bias, arXiv:2604.10506):

1. **Counterfactual Value Margin (CVM)**:`V(video, l_true) − V(video, l_cf)`,逐帧与终局两个版本;理想模型 margin 显著为正;
2. **Hallucination Rate (HR)**:反事实负例中被判为"进度 > 阈值 / 成功"的比例;
3. **Semantic AUROC**:以 margin 为分数区分匹配/不匹配对;
4. **Paraphrase Robustness (PR)**:hard positive 上的 value 保持率(防过敏感);
5. **进度曲线质量**:VOC(与 GVL/Robometer 协议对齐)在正例上的保持——语义增强不得损害原有进度估计能力。

### 4.3 被评测模型

GRM-2.0(4B/8B)、GVL(Gemini 系 in-context)、VLAC、RoboReward 4B/8B、Robometer、通用 VLM(GPT-5、Gemini、Qwen3-VL)。所有模型用统一 prompt 模板与帧采样协议。

### 4.4 诊断分析(RQ2,提升论文深度)

- **注意力诊断**:借鉴 localization heads(arXiv:2503.06287)方法,可视化各模型在指令物体 token 与图像 patch 间的注意力分布,区分"没看见"vs"没绑定";
- **消融探针**:遮挡指令物体 / 遮挡实际操作物体 / 空指令三组对照,量化语言对 value 输出的因果贡献(借鉴 LIBERO-Plus 的 blank-instruction 实验设计,迁移到 critic 侧);
- **错误分层**:按物体类别距离统计 HR,验证"越相似越难分辨"的假设。

### 4.5 缓解方案演示(RQ3,提供 baseline 与 upper bound)

1. Prompt 侧:显式要求模型先描述被操作物体再打分(CoT 校验);
2. 推理侧:反事实 margin 探针——同时前向 (video, l_true) 与 (video, l_shuffled-obj),用 margin 校正输出(critic 侧的 CAG 对应物);
3. **Training-free steering**(你已初验的方法,在本 proposal 中作为缓解方案之一而非主角);
4. 数据侧:用本数据集少量微调 GRM,给出 training-based upper bound。

## 5. 贡献点

1. **C1(数据集)**:首个真机遥操采集的 instruction-object 反事实失败数据集,含匹配/反事实/hard-positive 三元组结构,填补"自然语义失败数据"空白(现有全部为仿真程序扰动或合成重标注);
2. **C2(基准与指标)**:首个专测 value/reward model 语义分辨力的评测协议,提出 CVM/HR/PR 等指标,固定视频只换指令的设计消除时序混淆——相当于"critic 侧的 LIBERO-CF";
3. **C3(系统实证)**:对 8+ 个主流 reward model 的首次语义维度横评,量化"成功幻觉"并通过注意力诊断/因果探针给出机理解释;
4. **C4(缓解方案谱系)**:从 prompt 到 training-free steering 到微调的完整缓解 baseline,为后续研究提供起点。

## 6. 实验计划

| 阶段 | 内容 | 产出 |
|------|------|------|
| P1 (4周) | 数据集补齐与标注规范定稿;hard-positive 子集构建 | 数据集 v1 + datasheet |
| P2 (3周) | 评测 harness(复用 Robo-Dopamine eval 代码,扩展 API 模型接口) | 8+ 模型的 CVM/HR/AUROC/PR 全量结果 |
| P3 (3周) | 诊断分析:注意力可视化、遮挡/空指令因果探针、类别距离分层 | 机理章节 |
| P4 (4周) | 四类缓解方案实现与消融 | 缓解章节 + 榜单 |
| P5 (2周) | 撰写、开源(数据 + 代码 + leaderboard) | 投稿 |

## 7. 风险与备选

- **风险 1**:某些模型 HR 不高(问题不存在)→ 预实验已在 GRM 上观察到分辨力不足(你的初验);且 Guardian/RARM 的文献证据表明假阳性普遍。若个别新模型表现好,则转化为"哪些设计选择带来语义鲁棒性"的归因分析,同样有价值;
- **风险 2**:数据规模不足以支撑微调 upper bound → 微调仅作参考线,主贡献是评测;可用仿真数据补充训练、真机数据只做评测;
- **风险 3**:与 RoboRewardBench 的重合质疑 → 差异化:真机自然失败 vs 合成重标注;语义专项 vs 整体 MAE;triplet 结构支持因果归因 vs 单点打分。

## 8. 与其他两个 proposal 的关系

本 proposal 是地基:数据集与评测协议被 Proposal 2(方法)用作主实验平台,被 Proposal 3(系统)用作 verifier 的离线验收标准。三者可拆可合:资源紧张时,P1+P2 合并为"基准+方法"单篇强论文。
