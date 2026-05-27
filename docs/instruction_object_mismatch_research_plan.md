# 面向机器人过程奖励模型的 Instruction-Object Mismatch 诊断与缓解研究计划

## 摘要

本研究关注语言条件机器人过程奖励模型中的一种潜在失效模式：当指令要求操作对象 A，而视频实际正确操作了对象 B 时，模型仍可能因为观察到“某个物体完成了目标关系”而给出较高进度分数。本文档将该现象称为 **Instruction-Object Mismatch Reward Leakage**，简称 **IOM leakage**。

基于 Robo-Dopamine GRM 的初步分析中，我们观察到一个具有代表性的案例：

```text
Instruction: 把瓶子放到盘子上
Observed video: 把胡萝卜放到盘子上
Model behavior: 进度分数随胡萝卜完成放置而升高
Gradient attribution: 热区主要覆盖胡萝卜及放置关系相关区域
```

该观察提示模型可能学到了“任意可操作物体被放到盘子上即代表进度”的捷径，而没有充分绑定语言指定对象。需要强调的是，attention 或 gradient heatmap 单独不能证明因果关系；本研究将用 counterfactual evaluation 和 occlusion intervention 对该假设进行验证。

本研究的核心目标是：

1. 建立可复现的 instruction-object mismatch 评测协议与诊断指标。
2. 验证 GRM 是否存在对错误对象的奖励泄漏，以及它在任务、对象和视角上的分布。
3. 通过语义困难负样本、配对排序目标和可选的对象 grounding 约束，提高奖励模型对指令对象一致性的敏感度。
4. 评估改进是否在降低 mismatch false reward 的同时，保留正常任务上的进度估计性能和 RL 可用性。

## 1. 问题背景

### 1.1 研究对象

Robo-Dopamine 的 GRM 接收任务指令和多视角视觉状态，比较 `BEFORE` 与 `AFTER` 状态并预测进度变化。现有输入包含：

```text
task instruction
reference start / reference end
before: cam_high / cam_left_wrist / cam_right_wrist
after:  cam_high / cam_left_wrist / cam_right_wrist
```

这种过程奖励模型的价值在于：它可以为精细机器人操作提供比成功/失败标签更密集的反馈，并用于强化学习中的 reward shaping。

### 1.2 失效模式定义

设指令由动作、目标对象和目标位置组成：

```text
tau = (action, object, receptacle)
```

例如：

```text
tau_correct = (put, carrot, plate)
tau_wrong   = (put, bottle, plate)
```

对同一段真实视频 `v_carrot_to_plate`，合理的奖励关系应满足：

```text
r(v_carrot_to_plate, tau_correct) >> r(v_carrot_to_plate, tau_wrong)
```

如果：

```text
r(v_carrot_to_plate, tau_wrong) remains high
```

则模型发生了 instruction-object mismatch reward leakage。此时奖励函数偏向于识别泛化动作结构或目标容器关系，而没有验证被操作对象是否匹配语言指令。

### 1.3 为什么该问题重要

该问题不仅影响离线打分准确率，而且可能直接影响下游 RL：

- **奖励投机**：策略可能学会操作更容易移动的错误物体，而不是指令指定对象。
- **语义安全问题**：对于整理、分拣、递送或医疗操作，物体身份错误不能被视为部分成功。
- **跨对象泛化混淆**：模型若过度依赖容器、机械臂移动轨迹或物体运动，而忽略对象名称，会在新任务中系统性误判。
- **高精度 manipulation 的关键缺口**：过程奖励不仅需要识别动作进展，还必须验证任务语义约束。

### 1.4 当前初步证据及其边界

目前已有的可视化工具输出 attention rollout、gradient attribution 和 attention distribution diagnostics。初步观察包括：

- score token 的 attention/rollout 在某些样本中主要落在非图像 tokens；该现象不能推出模型没有使用视觉信息。
- 对 mismatch 案例，gradient attribution 在实际被操作的错误物体（如胡萝卜）上更明显。
- 同一 mismatch 案例中，模型进度分数仍上升。

这些观察构成研究动机，但尚不是严格证据。需要进一步执行：

- 相同视频下正确/错误指令配对打分；
- 高热区与低热区遮挡对照；
- 目标对象与错误被操作对象的 mask 级 attribution 分析；
- 不同对象、容器、动作类别上的系统评测。

## 2. 研究问题与假设

### 2.1 Research Questions

```text
RQ1: GRM 在 instruction-object mismatch 下是否系统性地产生虚假进度奖励？

RQ2: 该误差主要来自对象身份不敏感、容器关系捷径，还是动作阶段捷径？

RQ3: Counterfactual instruction relabeling 与 pairwise ranking 是否能降低 mismatch reward leakage？

RQ4: 加入对象 grounding 或显式语义一致性判断，是否能进一步提升鲁棒性而不损害正常进度估计？

RQ5: mismatch-robust GRM 是否能减少下游 RL 中操作错误对象的 reward hacking 行为？
```

### 2.2 可检验假设

```text
H1: 在固定视频、仅替换指令对象的设置中，基础 GRM 对错误对象指令仍会给出显著正向进度。

H2: 基础 GRM 的 gradient attribution 对实际移动物体的响应显著高于对指令指定但未移动物体的响应。

H3: 使用 instruction-object counterfactual negatives 进行训练可显著降低 false progress rate。

H4: 在 counterfactual negatives 上叠加 pairwise ranking loss，可在保持 matched-task 性能的前提下扩大 correct-vs-mismatch reward margin。

H5: 基于 object grounding 的辅助一致性监督或推理 gate 能进一步降低复杂干扰场景中的错误奖励。
```

## 3. 相关工作

### 3.1 语言条件机器人奖励学习

**LIV** 将语言-图像表征学习与机器人控制奖励学习结合，使模型能够针对语言或目标图像对未见视频状态给出稠密奖励。它说明语言条件视觉奖励是可行路线，但其研究重点不是细粒度的对象错配诊断。

**RoboCLIP** 使用视频-语言模型从单个视频演示或文本描述生成机器人策略学习所需奖励，体现了预训练视频语言语义作为奖励的潜力。该路线同样可能受到视觉动作结构与精确语义绑定之间差距的影响。

### 3.2 通用过程奖励模型与失败数据

**Robo-Dopamine** 提出多视角、step-aware 的 GRM 以及用于 RL 的奖励塑形方法。它为本研究提供基础模型和问题场景：高精度过程奖励必须同时满足视觉进度和语言语义一致性。

**RoboReward** 与本研究最直接相关。该工作针对机器人奖励 VLM 引入 negative examples 和 near-miss augmentation，其中包括对成功轨迹做 counterfactual relabeling。本研究将进一步聚焦于对象身份替换这一特定但高风险的语义混淆，并加入 attribution 与 intervention 验证。

**Robometer** 通过帧级进度监督与跨轨迹 preference supervision 结合，提高 reward model 从失败与次优轨迹学习的能力。其 trajectory comparison 设计启发本研究采用 matched/mismatched instruction pair 的排序约束。

### 3.3 视觉 grounding 与对象级监督

**Grounding DINO** 支持通过类别名称或 referring expression 定位开放集合对象；**SAM** 可将检测结果扩展为像素级对象 mask。二者可作为离线标注与归因评估工具，为“指令目标物体”和“实际被操作物体”提供 mask，不要求将检测器部署进最终奖励模型。

### 3.4 可解释性与证据可信度

**Attention is not Explanation** 指出 attention 权重不能直接视为预测因果解释。因此，本研究不把 attention 低或 attention 热区作为模型失败的单独证据。

**Attention Rollout** 提供了跨层聚合 Transformer attention 的信息流近似方法，适合辅助检查 score token 与 image tokens 的连接分布。

**Transformer Interpretability Beyond Attention Visualization** 强调仅可视化 attention 的局限，并通过 relevancy propagation 改善 Transformer 解释质量。受此启发，本研究将 gradient attribution 与 perturbation-based validation 作为主要解释证据。

### 3.5 现有工作的缺口

已有通用机器人奖励模型关注跨任务泛化、过程评估或失败轨迹利用，但对于以下问题仍需要系统分析：

```text
在视觉动作结果正确、但被操作对象与指令对象不一致时，
奖励模型是否会错误奖励？如何验证并针对性修复？
```

本研究的预期贡献不是提出另一种泛化 reward model，而是建立该语义失效模式的评测、证据链和缓解方案。

## 4. 可能采取的方法

### 4.1 方案总览

拟研究方法称为：

```text
Counterfactual Instruction-Object Consistent Reward Modeling (CIO-RM)
```

方法包含四个可逐步验证的组件：

```text
A. Mismatch diagnostic benchmark
B. Counterfactual instruction-object hard negatives
C. Pairwise instruction-conditioned ranking objective
D. Optional object-grounded consistency module
```

其中 A、B、C 为主线方法，D 作为增强实验，避免一开始将外部检测器误差引入核心结论。

### 4.2 Counterfactual mismatch 样本构造

对已有成功或部分成功视频，保持视觉输入不变，生成语义上错误但表面合理的新指令。

#### 样本类型

| 类型 | 原任务 | Counterfactual 指令 | 期望 |
| --- | --- | --- | --- |
| Object mismatch | put carrot on plate | put bottle on plate | 不应升分 |
| Receptacle mismatch | put carrot on plate | put carrot in bowl | 不应升分 |
| Relation mismatch | put carrot in bowl | put carrot next to bowl | 不应判完成 |
| Action-stage mismatch | lift carrot | place carrot on plate | 未完成后续步骤 |
| Attribute mismatch | put red block in bowl | put green block in bowl | 不应升分 |

#### 样本选择原则

- 错误对象应尽可能在场景中真实存在，构造困难负样本，而非简单缺失对象。
- 对象类别应覆盖形状相近和外观差异大的组合。
- 容器与空间关系替换应作为独立轴，防止方法只改善对象名称而忽略其他语义槽位。
- 视频、对象、场景布局在 train/validation/test 中按 episode 或 task combination 隔离，避免 counterfactual pair 泄漏。

#### 标签策略

对于错误对象指令，核心监督采用排序约束，而非完全依赖绝对标签：

```text
r(v, tau_correct) >= r(v, tau_wrong) + margin
```

此外，对明显不相关的完整动作，设置目标进度为零或接近零：

```text
target_progress(v, tau_wrong) = 0
```

这样既保留语义严谨性，又减少对“错误动作是否构成负进度”的主观争议。

### 4.3 训练目标

#### Baseline loss

保留 GRM 当前对进度分数或 hop 的训练目标：

```text
L_progress
```

该项保证模型原有的过程评分能力不退化。

#### Pairwise mismatch ranking loss

对同一视频和一对正确/错误指令，加入 margin ranking：

```text
L_rank = max(0, margin - r(v, tau_correct) + r(v, tau_wrong))
```

推荐将 `margin` 作为验证集调参项，首轮采用：

```text
margin = 0.3
```

其中 reward 归一化到 `[0, 1]` 或统一 progress 量纲。

#### Mismatch classification auxiliary loss

增加结构化语义判断目标：

```text
object_match: true / false
receptacle_match: true / false
relation_match: true / false
```

如果不修改模型 head，可将其设计为生成式辅助答案；若允许添加轻量预测头，则可使用 binary cross entropy：

```text
L_match
```

该目标用于明确训练“任务是否匹配”，而不只是输出一个合成进度分数。

#### 总损失

核心版本：

```text
L = L_progress + lambda_rank * L_rank
```

增强版本：

```text
L = L_progress + lambda_rank * L_rank + lambda_match * L_match
```

首轮建议：

```text
lambda_rank = 1.0
lambda_match = 0.5
```

权重最终以 validation mismatch accuracy 和 matched progress correlation 的 Pareto trade-off 决定。

### 4.4 结构化 reward decomposition

作为方法增强项，可将最终 reward 解释为：

```text
reward = semantic_match_gate * procedural_progress
```

其中：

```text
semantic_match_gate = f(object_match, receptacle_match, relation_match)
procedural_progress = current progress conditional on intended task
```

推理输出格式可以扩展为：

```json
{
  "instructed_object": "bottle",
  "manipulated_object": "carrot",
  "object_match": false,
  "relation_match": true,
  "progress": 0
}
```

研究中应将其作为独立 ablation，因为结构化输出可能提升可解释性，但也可能因为生成格式更复杂而降低 score 稳定性。

### 4.5 Object-grounded consistency

第二阶段可使用 Grounding DINO + SAM 离线得到：

```text
instruction object mask
actual manipulated object mask
target receptacle mask
```

这些 mask 可用于两种用途：

1. **评测 attribution alignment**：不影响训练，仅验证模型 attribution 是否落到正确对象。
2. **训练辅助监督**：要求正确任务下模型对 instructed object / receptacle 更敏感，mismatch 任务下不能只依赖错误被操作对象。

优先建议先做用途 1，再评估是否需要用途 2。原因是 grounding 模型自身可能在机器人遮挡、腕部相机和小物体上产生标签噪声。

### 4.6 不作为主要方法的方案

仅修改 prompt，例如加入：

```text
If the manipulated object does not match the task object, output zero progress.
```

可以作为低成本 baseline，但不应成为主贡献。它不能证明模型学会了视觉语义绑定，并可能仅改变输出偏置。

## 5. 实验设计

### 5.1 实验目标

实验需要同时回答三类问题：

```text
Diagnosis: 基础 GRM 是否存在 IOM leakage？
Mitigation: 所提方法是否减少错误奖励？
Utility: 改进后是否保留正常进度评估和下游训练价值？
```

### 5.2 数据集与评测集构造

#### Evaluation set: IOM-Bench

建立专门的 mismatch 测试集 `IOM-Bench`，每个原视频构造配对指令：

```text
(video, correct_instruction)
(video, object_mismatch_instruction)
(video, receptacle_mismatch_instruction)
(video, relation_mismatch_instruction)
```

最低可行规模：

| Split | 原始视频数量 | 每视频指令变体数 | 总 pair 数 |
| --- | ---: | ---: | ---: |
| Validation | 100 | 4 | 400 |
| Test | 300 | 4 | 1200 |

扩展版本应覆盖：

- 不少于 10 种 manipulated objects；
- 不少于 5 种 receptacles；
- 至少 3 类关系或动作阶段；
- 单视角与多视角设置；
- goal image 与 blank goal 两种设置。

#### Training augmentation set

从训练集成功/部分成功轨迹生成 counterfactual pairs，首轮采用三档比例：

```text
negative ratio = 10%, 25%, 50% of matched training instances
```

用于观察增加困难负样本是否存在正常性能退化点。

### 5.3 评价指标

#### Reward correctness

**Matched Progress Quality**

在正确指令下保留原有指标，例如 progress correlation、pairwise accuracy 或项目现有 benchmark accuracy，确保方法没有只通过压低所有 reward 来改善 mismatch。

**Mismatch False Progress Rate (FPR)**

```text
FPR@tau = proportion of mismatched samples with predicted progress >= tau
```

报告：

```text
tau in {0.2, 0.5, 0.8}
```

**Counterfactual Reward Gap (CRG)**

```text
CRG = mean[r(v, tau_correct) - r(v, tau_wrong)]
```

越大越好。分别报告 object / receptacle / relation mismatch。

**Counterfactual Pair Accuracy (CPA)**

```text
CPA = P(r(v, tau_correct) > r(v, tau_wrong))
```

#### Explanation and intervention

**Object Attribution Alignment (OAA)**

基于 instructed object mask 与错误 manipulated object mask：

```text
OAA = attribution_mass(instructed_object_mask)
      - attribution_mass(wrong_manipulated_object_mask)
```

在 mismatch 视频中，需谨慎定义目标：如果指令对象未发生运动，理想 reward 的依据可能是识别“不匹配”而非将 attribution 全部放在未动对象上。因此 OAA 作为辅助分析，不作为唯一优化目标。

**Occlusion Faithfulness Gap (OFG)**

对同一输入分别遮挡：

```text
wrong manipulated object
instruction-specified object
target receptacle
random background
```

统计 score 变化：

```text
delta_wrong_object = r(original) - r(mask_wrong_object)
delta_background   = r(original) - r(mask_background)
```

若基础模型在 mismatch case 中主要依赖错误对象，则应观察到：

```text
delta_wrong_object >> delta_background
```

训练改进后的理想现象是：mismatch reward 已接近零，且不再由错误对象的完成状态触发高分。

#### Downstream RL utility

构造含多个可操作物体的环境，指令仅指定一个对象。报告：

```text
target-object success rate
wrong-object interaction rate
episode reward
environment true success rate
```

关键指标是 `wrong-object interaction rate` 是否因改进后的 reward model 而降低。

### 5.4 对比方法与消融实验

| 模型 | 训练方式 | 目的 |
| --- | --- | --- |
| Base GRM | 原模型或原任务微调 | 建立 failure baseline |
| Prompt-only | 增加 mismatch 规则提示 | 测试仅依赖提示的收益 |
| CF-Neg | 增加 counterfactual negative 绝对监督 | 测试数据贡献 |
| CF-Rank | `L_progress + L_rank` | 主方法最小版本 |
| CF-Rank-Match | `L_progress + L_rank + L_match` | 测试显式一致性监督 |
| CF-Rank-Match-Ground | 加 object grounding 辅助 | 测试对象级监督收益 |
| Gate Inference | semantic gate 乘 progress | 测试显式安全约束 |

必要消融：

```text
object mismatch only vs all semantic mismatches
negative ratio: 10% / 25% / 50%
with vs without goal image
single view vs multi view
forward vs incremental vs backward vs fused score
with vs without occlusion-derived evidence
```

### 5.5 统计方法

- 对 `CRG`、matched progress metric、occlusion delta 报告均值与 bootstrap 95% confidence interval。
- 对 `CPA` 和 `FPR` 报告比例置信区间，并使用 paired bootstrap 比较同一视频上的模型差异。
- 所有主要比较使用固定 test pairs，避免不同 counterfactual 采样造成偏差。
- 在训练选择上仅使用 validation 集；test 集只在最终模型与确定 ablation 上评估。

### 5.6 预期结果与判定标准

主方法有效的最低标准：

```text
CPA(object mismatch) 显著高于 Base GRM
FPR@0.5(object mismatch) 明显下降
CRG(object mismatch) 明显扩大
Matched Progress Quality 不发生显著下降
```

较强结果应额外满足：

```text
receptacle/relation mismatch 上同样有效
在 unseen object combinations 上有效
下游 RL 的 wrong-object interaction rate 显著下降
```

## 6. 具体开展步骤

### Phase 0: 固定基线与复现实验

目标：把当前观察变成可重复的 baseline。

工作项：

1. 固定模型 checkpoint、prompt、frame interval、三种 eval mode 与 fused score 计算方式。
2. 固定当前 mismatch 案例：

   ```text
   bottle instruction vs carrot-to-plate video
   ```

3. 保存基础模型输出：

   ```text
   pred_vllm.json
   attribution_vis.mp4
   attention_diagnostics.json
   heatmaps/*.npz
   ```

4. 为每个视频建立正确指令与错误对象指令对。
5. 输出首版配对表格：correct score、mismatch score、reward gap。

交付物：

```text
results/iom_baseline/
docs/iom_baseline_observations.md
```

### Phase 1: 建立 IOM-Bench

目标：从个例扩展到可量化评估集。

工作项：

1. 从已有数据抽取 object/receptacle/action metadata。
2. 生成 object、receptacle、relation 三类 counterfactual instructions。
3. 人工抽查全部 validation pair 和至少 20% test pair，排除歧义指令。
4. 确保测试中的 object combination 不与训练 augmentation 完全重合。
5. 实现评测脚本，输出：

   ```text
   CPA
   CRG
   FPR@0.2 / FPR@0.5 / FPR@0.8
   per-mismatch-category metrics
   ```

交付物：

```text
data/iom_bench/{val,test}.json
eval/evaluate_iom_bench.py
results/iom_baseline_metrics.json
```

### Phase 2: 因果诊断与 attribution 验证

目标：验证错误奖励是否由错误物体视觉证据驱动。

工作项：

1. 用现有 attribution 脚本生成 matched/mismatched pairs 的 gradient maps。
2. 使用 Grounding DINO 定位 instruction object、manipulated object、receptacle。
3. 使用 SAM 细化 masks；人工校验用于论文图和关键 test subset 的 mask。
4. 实现 occlusion evaluation：

   ```text
   mask wrong manipulated object
   mask instructed object
   mask receptacle
   mask random area
   ```

5. 输出 OAA、OFG 和代表性 qualitative figures。

交付物：

```text
eval/evaluate_iom_occlusion.py
results/iom_occlusion/
figures/iom_failure_examples/
```

### Phase 3: Counterfactual negative 数据增强

目标：构造训练所需语义困难负样本。

工作项：

1. 在训练轨迹上构造 counterfactual instruction pairs。
2. 先实现 object mismatch，再扩展 receptacle 与 relation mismatch。
3. 生成 10%、25%、50% 三种 negative ratio 数据配置。
4. 对训练样本记录来源与 counterfactual 类型，保证可做消融。
5. 检查指令对象是否场景中真实存在；分别保留 absent-object 与 present-distractor 子集。

交付物：

```text
train/tools/build_iom_counterfactual_data.py
train_data/iom_cf_10/
train_data/iom_cf_25/
train_data/iom_cf_50/
```

### Phase 4: 训练主方法

目标：比较数据增强、排序损失与语义辅助目标。

训练顺序：

1. `Base GRM`：复现或沿用现有 checkpoint。
2. `Prompt-only`：只调整提示词，不更新模型或做最小微调。
3. `CF-Neg`：只加入 counterfactual absolute labels。
4. `CF-Rank`：加入 pairwise ranking，为论文主模型候选。
5. `CF-Rank-Match`：增加 object/receptacle/relation matching auxiliary target。
6. 仅在前述结果证明需要时运行 `CF-Rank-Match-Ground`。

模型选择准则：

```text
优先最大化 validation CPA 与 CRG；
约束 matched progress metric 退化不超过预先设定容忍范围；
若两个模型接近，选择结构更简单且不依赖外部 grounding 的版本。
```

交付物：

```text
train/scripts/finetune_iom_*.sh
train/checkpoints/iom_*/
results/iom_validation_model_selection.csv
```

### Phase 5: 全面评测与下游验证

目标：支撑论文主要结论。

工作项：

1. 在 IOM-Bench test 集评测全部确定的模型。
2. 在原有正常 reward benchmark 上验证保真性能。
3. 运行 attribution 与 occlusion 定量分析。
4. 选择含多个物体、可发生错误操作的 RL 任务做 downstream test。
5. 输出定量表格与关键案例视频。

主要论文表格：

```text
Table 1: Matched progress quality and IOM-Bench CPA/CRG/FPR
Table 2: Ablation on counterfactual type, ranking loss, negative ratio
Table 3: Attribution alignment and occlusion faithfulness
Table 4: Downstream RL true success and wrong-object interaction rate
```

主要论文图：

```text
Figure 1: Instruction-object mismatch failure example
Figure 2: CIO-RM training pipeline
Figure 3: Correct/mismatch score curves over video progress
Figure 4: Attribution plus occlusion intervention comparisons
```

### Phase 6: 论文写作与复现整理

建议论文结构：

```text
1. Introduction
2. Related Work
3. Instruction-Object Mismatch in Process Reward Models
4. Counterfactual Instruction-Object Consistent Reward Modeling
5. IOM-Bench and Experimental Protocol
6. Experiments
7. Limitations and Broader Implications
```

复现材料：

```text
数据生成脚本
训练配置
评测脚本
固定 prompt / model checkpoints
定量结果 JSON/CSV
attribution 与 occlusion 可视化视频
```

## 7. 时间安排建议

| 周期 | 目标 | 里程碑 |
| --- | --- | --- |
| Week 1 | 固定 baseline；构建首版 IOM-Bench | 可报告基础模型 CPA/CRG/FPR |
| Week 2 | 完成 mask/occlusion 因果诊断 | 证明或反驳 reward leakage 来源 |
| Week 3 | 生成 counterfactual train data；跑 CF-Neg | 首个缓解模型 |
| Week 4 | 加入 ranking/match objectives；做消融 | 确定主模型 |
| Week 5 | 全量测试与 downstream RL 验证 | 完整主要表格 |
| Week 6 | 写作、补实验、整理复现材料 | 论文初稿 |

## 8. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| Attribution 不具备因果可信度 | failure claim 不稳 | 以 occlusion intervention 和配对 score 作为核心证据 |
| Counterfactual instruction 产生歧义 | 标签噪声 | 人工审核 validation/test；区分场景中有/无 distractor |
| Grounding 模型漏检小物体或受遮挡影响 | mask 指标不可靠 | grounding 仅作辅助；关键 subset 人工校正 |
| 训练后模型整体压低 reward | mismatch 指标虚假改善 | 同时报 matched progress quality 和下游 true success |
| 只改善 seen objects | 泛化不足 | 按 unseen object combinations 划分 test |
| 外部 detector gate 影响部署复杂度 | 方法价值下降 | 将不依赖 detector 的 CF-Rank 作为主模型，grounding 作为增强 |

## 9. 预期论文贡献

如果实验支持假设，论文可主张以下贡献：

1. 揭示机器人视觉语言过程奖励模型中的 instruction-object mismatch reward leakage，并提供系统诊断证据。
2. 提出 IOM-Bench，以 paired counterfactual instructions 测量对象、容器和关系错配下的虚假奖励。
3. 提出 CIO-RM，通过 counterfactual semantic hard negatives 与 instruction-conditioned ranking 提高语义一致性。
4. 结合 attribution 与 occlusion intervention 建立更可信的 reward model failure analysis protocol。
5. 证明降低 mismatch leakage 可减少 RL 中的错误对象交互，而不显著损害正常过程奖励能力。

## 10. 参考文献与资源

1. Tan et al. **Robo-Dopamine: General Process Reward Modeling for High-Precision Robotic Manipulation**. arXiv:2512.23703, 2025.  
   <https://arxiv.org/abs/2512.23703>

2. Ma et al. **LIV: Language-Image Representations and Rewards for Robotic Control**. ICML, 2023.  
   <https://arxiv.org/abs/2306.00958>

3. Sontakke et al. **RoboCLIP: One Demonstration is Enough to Learn Robot Policies**. NeurIPS, 2023.  
   <https://arxiv.org/abs/2310.07899>

4. Lee et al. **RoboReward: General-Purpose Vision-Language Reward Models for Robotics**. arXiv:2601.00675, 2026.  
   <https://arxiv.org/abs/2601.00675>

5. Liang et al. **Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons**. arXiv:2603.02115, 2026.  
   <https://arxiv.org/abs/2603.02115>

6. Jain and Wallace. **Attention is not Explanation**. NAACL, 2019.  
   <https://arxiv.org/abs/1902.10186>

7. Abnar and Zuidema. **Quantifying Attention Flow in Transformers**. ACL, 2020.  
   <https://aclanthology.org/2020.acl-main.385/>

8. Chefer, Gur, and Wolf. **Transformer Interpretability Beyond Attention Visualization**. CVPR, 2021.  
   <https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html>

9. Liu et al. **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**. arXiv:2303.05499, 2023.  
   <https://arxiv.org/abs/2303.05499>

10. Kirillov et al. **Segment Anything**. arXiv:2304.02643, 2023.  
    <https://arxiv.org/abs/2304.02643>

