# RoboRewardBench 反事实指令构造分析

分析日期：2026-07-22  
分析对象：`/home/dais/workspace/data/RoboRewardBench/test/metadata.jsonl` 及对应视频  
metadata SHA-256：`3901cea9a7981d2e2d9fde4225a4b2ff0ebf2f4af69fc96972a2cfbe557db0cd`

## 1. 结论先行

反事实 relabeling **不是把一个与视频场景完全无关的随机 instruction 放到视频上**，也
不只是机械地把原指令中的 `A` 替换成 `B`。更准确的描述是：

> 保持成功 rollout 视频完全不变，在同一视频的可见物体和关系范围内，生成一组对该
> 最终状态分别应得 1、2、3、4 分的替代指令；原始成功指令为 5 分。

其中两种直觉都只说对了一部分：

- 如果“完全无关”指与机器人实际执行的动作无关，那么 **reward=1 经常如此**。论文
  明确要求 1 分指令对应“机器人没有做任何与目标相关的动作”，典型方式就是视频实际
  操作 A，而新指令要求操作同场景中可见但未被操作的 B。
- 如果“完全无关”指与视频场景无关，那么 **不是**。生成 prompt 禁止捏造不可见物体，
  要求只使用视频分析中可见的物体和关系；验证器还会拒绝无法在视频中落地的指令。
- `pick A → pick B` 是 1 分 hard negative 的一种典型形式，但不是唯一形式。它也可以
  改成同类物体的另一个实例、另一个位置，或场景内另一项完全未执行的任务。
- reward=3 和 reward=4 反而被强制要求继续引用原任务的**同一个主目标物体**。3 分通常
  修改核心空间关系或加入多个未满足要求；4 分通常保留核心物体和核心关系，只增加一个
  视频末态未满足的次要约束，例如朝向、精度、是否释放夹爪或是否保持某物体不动。
- reward=2 位于两者之间：视频对新目标只有少量相关进展。论文 prompt 没有像 3/4 分
  那样强制它保留同一主目标物体，因此实际数据中既有同物体错误终点，也有场景内其他
  物体与关系的有限相关任务。

所以，最合适的名称不是“随机负指令”，而是 **scene-grounded、instruction-conditioned、
按失败程度分级的 counterfactual command ladder**。

## 2. 证据来源与可信度

本分析使用三个一手来源：

1. RoboReward 论文正文 §3 和附录 “Data Cleaning and Augmentation Details”。论文给出
   生成模型、完整 prompt、1–5 分规则和验证流程。[^paper]
2. 官方数据集说明。其简化描述是：“保持同一 rollout 视频，但换成根据最终状态应得到
   更低分的替代任务指令”。[^dataset]
3. 发布的 RoboRewardBench test metadata 和 2,831 个视频文件。本报告直接分析其中
   669 条 OXE 反事实记录，并以视频文件 SHA-256 恢复内容完全相同的指令组。

这些来源对“作者如何构造数据”具有直接证据力，但不是相互独立的第三方复现：论文和
数据均来自同一作者团队，metadata 中的 `gpt5_mini_check` 也是流水线内部验证结果。
论文称 test split 另外经过逐条人工验证，但 release 没有提供独立人工标注日志。

## 3. 论文中实际采用的生成流程

### 3.1 先区分语法清理与反事实改写

论文先用 Qwen3-4B-Instruct-2507 对原任务做语法、拼写和大小写清理，并明确要求“不改变
含义”。这一步只是 invariant clean-up，例如将拼错的 `palce` 修正为 `place`，不是反事实
relabeling。

反事实语义改写是后续独立的多阶段流程：

1. **视频分析**：GPT-5 mini 以 1 FPS 查看 rollout，包括真实终帧；描述初始场景、可见
   物体、机器人动作和最终状态，并被要求不得捏造不可见物体。
2. **失败梯度规划**：GPT-5 mini 基于原始 5 分任务设计 1、2、3、4 四种不同失败模式，
   强制满足 `1 < 2 < 3 < 4 < 5`。
3. **逐分生成命令**：Qwen3-4B-Instruct-2507 按 1→4 的顺序，每个分数生成一条命令；
   后生成的命令会看到已经生成的低分命令，以减少重复和梯度冲突。
4. **视频验证与拒绝采样**：GPT-5 mini 再以 1 FPS 检查“指令是否能在视频中落地”和
   “给定分数是否符合 rubric”；不一致就丢弃并重新生成。
5. **test 人工复核**：论文称进入 RoboRewardBench 的 test 样本还会逐条人工确认任务、
   视频和 reward 是否一致。

这是一套生成整条 imperative command 的流程，不是对原字符串做固定槽位替换。命令
生成 prompt 还要求：

- 新指令必须比原任务更严格或不同，不能是原任务蕴含的更容易子任务；
- 只能使用视频分析中可见的物体和关系；
- 不能在文本中提及“原任务”“分数”或其他元信息；
- 以动词开头、使用 plain ASCII、少于 25 个词，并匹配原任务的措辞风格；
- 不得提出在视频初始状态就已经成立的目标。

### 3.2 各 reward 的设计含义

| reward | 论文构造约束 | 常见改写方式 | 与原始 5 分任务的关系 |
|---:|---|---|---|
| 1 | 最终状态没有任何与新目标相关的变化 | 换成同场景中未操作的物体、另一实例或完全未执行的场景内任务 | 可以很远；典型是“实际操作 A，新指令要求 B” |
| 2 | 对目标只有小而不足的变化 | 同物体但错误终点；或对另一可见物体/关系只有有限进展 | 中间层；论文没有强制保留原主物体 |
| 3 | 有明显进展，但违反一个核心要求或多个要求 | 保留同一主物体，修改核心目的地、核心空间关系或加入多个未满足条件 | 强制使用原任务同一主目标物体 |
| 4 | 核心区域和意图正确，只缺一个次要要求 | 在原任务上增加朝向、精度、夹爪状态、支撑面或轻微位置约束 | 强制使用原任务同一主目标物体，最接近原指令 |
| 5 | 所有要求都满足 | 原始 OXE 成功任务 | 未做反事实改写 |

论文给出的直观例子是：原视频完成“把 pepper 放进炉灶上的 pot”。替代指令可以是：

- `place pepper in the shelf`：仍然操作 pepper，但目标位置错误，构成 partial progress；
- `clean the pot on the stove`：视频没有执行清理动作，构成 no success。

因此“语义距离”是按视频最终状态和任务要求定义的，不是简单按两个句子的文本编辑距离
定义的。

## 4. test split 中反事实数据的组成

### 4.1 识别规则

本报告将满足以下条件的记录视为 OXE 反事实任务：

- 不属于 `robo_arena`；
- 文件名不满足时间截断模式 `*_attempt_<n>_score_<1-4>.mp4`；
- metadata reward 为 1–4。

文件名末尾的 `_1`、`_2`、`_3` 等只表示同一视频内容的多个存储副本，**不能**据此
推断 reward。例如某些原始 5 分任务位于 `_2.mp4`，而 1 分任务反而没有数字后缀。
reward 必须读取 metadata。

### 4.2 数量与文本属性

test 中共有 **669** 条 OXE 反事实任务，覆盖全部 22 个 OXE subset：

| reward | 数量 | 占 669 条比例 | 唯一指令数 | 平均词数 |
|---:|---:|---:|---:|---:|
| 1 | 228 | 34.1% | 224 | 11.36 |
| 2 | 112 | 16.7% | 109 | 15.90 |
| 3 | 212 | 31.7% | 211 | 15.74 |
| 4 | 117 | 17.5% | 117 | 16.63 |
| **总计** | **669** | **100%** | **661** | **14.43** |

release 的其他可检查属性：

- 669/669 条 `gpt5_mini_check` 均以 `ANSWER: TRUE` 结束；这说明全部通过发布流水线的
  自动验证，但不能视为独立第三方验证。
- 661 条唯一指令；重复出现造成的额外记录只有 8 条。
- 按空格计词，666 条少于 25 词，3 条恰好 25 词，没有超过 25 词。
- 659/669 条任务为纯 ASCII。另有 10 条包含弯引号、en dash、non-breaking hyphen 或
  `jalapeño` 等字符，与生成 prompt 的 plain-ASCII 约束存在轻微不一致，但不影响语义。

各 subset 的 reward 组成如下：

| subset | r=1 | r=2 | r=3 | r=4 | total |
|---|---:|---:|---:|---:|---:|
| `austin_sirius_dataset_converted_externally_to_rlds` | 9 | 3 | 7 | 4 | 23 |
| `berkeley_autolab_ur5` | 19 | 2 | 4 | 6 | 31 |
| `berkeley_fanuc_manipulation` | 6 | 9 | 9 | 3 | 27 |
| `berkeley_mvp_converted_externally_to_rlds` | 5 | 7 | 4 | 4 | 20 |
| `berkeley_rpt_converted_externally_to_rlds` | 17 | 0 | 1 | 5 | 23 |
| `bridge` | 19 | 5 | 13 | 7 | 44 |
| `cmu_play_fusion` | 9 | 9 | 8 | 5 | 31 |
| `dlr_edan_shared_control_converted_externally_to_rlds` | 4 | 2 | 4 | 1 | 11 |
| `droid` | 15 | 4 | 8 | 8 | 35 |
| `fractal20220817_data` | 16 | 6 | 14 | 10 | 46 |
| `iamlab_cmu_pickup_insert_converted_externally_to_rlds` | 10 | 15 | 19 | 5 | 49 |
| `jaco_play` | 15 | 15 | 11 | 6 | 47 |
| `kaist_nonprehensile_converted_externally_to_rlds` | 5 | 2 | 5 | 2 | 14 |
| `roboturk` | 8 | 6 | 16 | 8 | 38 |
| `stanford_hydra_dataset_converted_externally_to_rlds` | 7 | 2 | 4 | 1 | 14 |
| `taco_play` | 16 | 3 | 16 | 8 | 43 |
| `tokyo_u_lsmo_converted_externally_to_rlds` | 0 | 1 | 9 | 3 | 13 |
| `ucsd_kitchen_dataset_converted_externally_to_rlds` | 10 | 7 | 8 | 3 | 28 |
| `ucsd_pick_and_place_dataset_converted_externally_to_rlds` | 8 | 1 | 20 | 13 | 42 |
| `utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds` | 8 | 1 | 20 | 12 | 41 |
| `utokyo_xarm_bimanual_converted_externally_to_rlds` | 8 | 8 | 9 | 3 | 28 |
| `viola` | 14 | 4 | 3 | 0 | 21 |
| **总计** | **228** | **112** | **212** | **117** | **669** |

## 5. 用相同视频 SHA-256 恢复指令梯子

test metadata 没有 `original_task` 或 `source_video_id` 字段，因此不能只靠 JSONL 为全部
669 条反事实任务找回原始 5 分指令。本报告改为对全部 1,040 条非时间截断 OXE 文件
计算 SHA-256，并将字节完全相同的视频分组。

结果为：

| 项目 | 数量 |
|---|---:|
| 非时间截断 OXE 记录 | 1,040 |
| 其中反事实 reward 1–4 | 669 |
| 其中原始成功 reward 5 | 371 |
| 唯一视频 SHA-256 | 760 |
| 同时包含 reward 5 和至少一个反事实任务的视频组 | 142 |
| 可与同一字节视频的 reward 5 配对的反事实记录 | 191 |
| test 中未保留对应 reward 5 行的反事实记录 | 478 |

191/669（28.6%）的反事实记录可以仅凭 release 内的文件内容与 5 分原指令精确配对。
剩余 478 条不是“视频被改过”的证据，而是 test 的人工过滤和 subsampling 没有保证每个
底层视频的完整 1–5 梯子都同时发布。缺少原始行时，release 本身不足以逐条恢复原指令。

在 142 个可配对视频组中，104 组保留 1 条反事实记录，28 组保留 2 条，9 组保留 3 条，
只有 1 组保留完整的四条反事实记录。191 条可配对记录按 reward 分别为：r1=63、r2=30、
r3=64、r4=34。

### 5.1 一个完整的 1–5 实例

以下五个 MP4 的文件 SHA-256 完全相同：
`63a37d34778040a437800a2dca150d4877c13311ec7fa7e8218d00fdf668a59c`。
它们来自 `fractal20220817_data_originalsplit_train_index_22148`，唯一变化是 instruction 和
reward：

| reward | instruction | 构造逻辑及 validator 判断 |
|---:|---|---|
| 1 | `Move the red Coca-Cola can to the center of the metal table.` | 视频实际操作绿色薯片袋，红罐未动；典型的场景内错物体 hard negative |
| 2 | `Bring the green jalapeno chip bag nearer to the red Coca-Cola can.` | 已操作正确的绿色袋并稍微靠近红罐，但终点仍不足 |
| 3 | `Place the green jalapeno chip bag directly on top of the RxBar blueberry so the RxBar is fully covered.` | 保留绿色袋，要求新的核心关系“完全覆盖”；视频只有明显但不完整的进展 |
| 4 | `Move green jalapeno chip bag to be immediately to the right of RxBar blueberry, lying flat with logo upward` | 主物体和核心相对位置已满足，只缺“平放且 logo 朝上”的次要朝向约束 |
| 5 | `Move green jalapeno chip bag near RxBar blueberry` | 原始成功任务，视频完全满足 |

这个实例直接说明：1 分确实很像“视频做 A，指令要求 B”；但 3/4 分不是换成无关任务，
而是围绕原主物体构造核心关系错误和次要约束错误。

### 5.2 其他精确配对实例

| 原始 5 分指令 | 反事实指令 | reward | 类型 |
|---|---|---:|---|
| `Insert the blue gear onto the right peg, followed by the red gear.` | `Move the small beige wooden block assembly to the far right corner of the tabletop.` | 1 | 换成同场景中未操作的米色零件 |
| `Pick up the object and place it in the box.` | `Slide the mug so it rests against the outside wall of the box.` | 2 | 仍围绕被操作物和箱子，但动作及目标关系只得到有限进展 |
| `Put mushroom in pot, laying down.` | `Wedge the mushroom across the pot rim so it is partly inside and partly supported by the rim.` | 3 | 同一 mushroom 和 pot，修改核心 inside/support 关系 |
| `Move orange near blue chip bag` | `Move orange near blue chip bag and retract robot gripper to base` | 4 | 原核心任务已完成，仅新增夹爪回撤约束 |

## 6. 配对指令的文本相似度只支持“趋势”，不能替代语义判断

对 191 条有精确 5 分配对的反事实指令，本报告将文本转为小写字母数字 token，去除常见
功能词和通用动作词，再计算与原始指令的 token Jaccard。结果如下：

| reward | 配对数 | Jaccard 均值 | 中位数 | 零 content-token 交集 | Jaccard < 0.2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 63 | 0.125 | 0.083 | 25（39.7%） | 50（79.4%） |
| 2 | 30 | 0.247 | 0.200 | 3（10.0%） | 13（43.3%） |
| 3 | 64 | 0.352 | 0.333 | 2（3.1%） | 15（23.4%） |
| 4 | 34 | 0.312 | 0.290 | 3（8.8%） | 12（35.3%） |

1 分指令与原指令在词面上明显最远，3/4 分整体更近，这与论文 prompt 一致。但文本
相似度并不严格单调：一些原任务非常简略，例如 `Layout laundry`，反事实近失误指令会
变成详细的毛巾位置和朝向描述，词面重合很低但语义仍围绕同一任务；相反，要求打开另一
扇 cabinet door 的 1 分指令可能与原文共享很多词。不能用 Jaccard 自动替代视频条件下的
任务语义分类。

metadata 中 validator 文本也呈现出预期分层。使用透明的固定短语匹配，而不是新的模型
分类，可观察到：

- 138/228 条 r1 解释明确包含 `no goal-relevant change`；
- 112/112 条 r2 解释包含 `minimal progress` 或 `small but insufficient`；
- 199/212 条 r3 解释包含 `major requirement`、`multiple requirements` 或
  `partial completion`；
- 114/117 条 r4 解释包含 `minor requirement`、`auxiliary constraint` 或
  `near completion`。

这些数字只说明发布的验证解释遵循统一 rubric，不应被误读成独立的人工作业标签。

## 7. 对 RoboRewardBench 评估含义的影响

1. **反事实部分主要测 task-conditioned grounding。** 同一字节视频可以对应 1–5 中多个
   reward。只判断“动作是否看起来成功”、但不认真读取 instruction 的模型会在这些样本
   上失败。
2. **反事实低分不等于物理 rollout 失败。** 底层 OXE 视频本来是成功 demonstration；
   它只是对替代 instruction 失败。因此 OXE 反事实 MAE 更接近“组合式指令—末态匹配”
   能力，而不是单纯的动作失败检测。
3. **1 分包含场景内困难负例。** 它通常不是随便抽一个跨场景指令，而是选用当前画面中
   真正可见的干扰物体或未执行目标，更容易暴露模型只看显著动作而忽略目标对象的问题。
4. **3/4 分强调细粒度空间关系。** 它们需要区分 inside/on-rim/near/touching、左右、
   朝向、是否释放等条件，端点图像的遮挡和单视角会直接影响判断。
5. **样本并非统计独立。** test 内存在字节相同视频的多个 instruction 版本；若为指标
   构造置信区间，按底层视频做 cluster bootstrap 会比把每条 instruction 当独立样本更
   严谨。官方 MAE 本身仍应按每条 task-video pair 计分。

## 8. 无法从当前 release 严格回答的部分

- test metadata 没有保存每条反事实记录对应的 `original_task`、生成 plan、生成轮次或
  rejection 日志。除 191 条有同视频 5 分副本的记录外，不能从 release 单独恢复完整
  1–5 梯子。
- 不能仅凭低文本相似度断言“换物体”，也不能仅凭高文本相似度断言“只改一个属性”；
  这需要同时比较原指令和视频最终状态。
- `gpt5_mini_check` 是模型生成的验证理由。论文称 test 另经人工验证，但 release 不包含
  可审计的人工判定细节。
- 因此，本报告对**生成规则**的结论主要来自论文完整 prompt，对**发布数据表现**的结论
  来自可复核的 metadata、文件哈希和配对样例；没有对 669 条记录强行给出不可靠的人工
  细分类标签。

## 9. 最终回答

对问题“是完全无关 instruction，还是把 `pick A` 改成 `pick B`”，答案是：

- **不是跨场景随机无关 instruction**；指令必须能在当前视频的物体和关系中落地。
- **reward 1 经常类似 `pick A → pick B`**，但 B 是同一场景中可见、机器人没有按要求
  操作的物体，所以它是有意设计的场景内错目标负例。
- **reward 2 是有限相关进展**，可以保留同一物体，也可以换到另一可见物体/关系。
- **reward 3/4 主要是修改正确 instruction 的要求**：保留同一主目标物体，分别制造
  核心关系/多个要求未满足，以及仅一个次要约束未满足的 near-miss。
- 整体上它是一套依据视频末态生成的 1–5 分语义梯子，而不是一种单一的字符串改写规则。

[^paper]: Tony Lee, Andrew Wagenmaker, Karl Pertsch, Percy Liang, Sergey Levine, and Chelsea Finn. “RoboReward: General-Purpose Vision-Language Reward Models for Robotics,” arXiv:2601.00675, 2026. [论文页面](https://arxiv.org/abs/2601.00675)；相关内容见 §3 “The RoboReward Dataset and Benchmark” 及附录 “Data Cleaning and Augmentation Details”。
[^dataset]: 官方 RoboReward 数据集说明：[Hugging Face `teetone/RoboReward`](https://huggingface.co/datasets/teetone/RoboReward)。
