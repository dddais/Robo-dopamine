# 第33轮 attention 增量低秩变换：统计核查

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-12
- Verification Status: ANALYZED
- Version Label: head_delta_reft_validation_v1
- Overall Confidence: CAUTION（数据集内自适应探索）

对象为固定最终第33轮方法的9输入、61个冻结主条件，及其全部预定对照。主条件已有完整逐例审计与独立统计算术重构；本文不是重新训练的独立复现。主target统计完成于16:53，对照仍在运行，完成后在本文追加核查结果。完整方法和支持相邻k见[17:00记录](PROGRESS_20260912_1700.md)。

结果根目录为当前 Robo-Dopamine 的 `results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911/`，下文路径均相对于此目录。只使用本目标目录的证据。

## 已确认的效应与不确定性

完整846条为268 suc/578 fail；validation660为209/451；old_holdout730为234/496，包含discovery70。三人口相互重叠，不能视作三次独立重复。监督训练与功能head排名使用discovery70/28个视频组，validation在整个研究中反复被查看。

61个主条件中54个在三人口同时满足MAE下降、suc/fail准确率上升、总准确率点增益至少10个百分点，并且四项视频组95%区间方向有利。九个输入均有至少一对相邻已测k，Qwen4输入、RoboReward5输入；原五输入都已在discovery尝试，Qwen interleaved六条件失败，未进入full。这里的10是绝对百分点；没有要求或声称准确率增益CI下界超过10pp。连线不外推未测整数k，失败点不能连接成连续稳定范围。

CI按原video_sha256带全部指令重采样5000次。sign-flip p依赖视频组差值在零假设下的符号可交换性，是附加探索统计，不是随机分派产生的严格随机化检验。报告的单侧p最小分辨率为1/5001。各项原始p、Holm p及CI均在points.json/CSV逐条件提供，不能将有限重采样的p写成0。

主target checkpoint中每项Holm校正包含122条阈值记录，实际上是61条件的两套重复阈值。四项Holm p同时小于.05的条件为validation40、full50、old52。控制加入会扩大该校正范围；这仍不覆盖全部33轮研究，不能证明独立确认性显著。接受规则仍是原登记的效应量与CI方向，不能事后改用其中一种p值挑结果。

`audit/head_delta_statistical_structure_20260912_165615.json`从五类直方图、task与配对六档重构全部183条主汇总，算术一致。只有5条没有恶化task。唯一micro/macro方向反转为Qwen text_image k24 validation：micro ΔMAE −0.1757576、task等权macro +0.0176615，13 task改善、7恶化、8不变。这是权重敏感性反例，不能把它误称为所有分层都相反的严格Simpson悖论。

validation31/61、full29/61、old30/61条件没有任何中间类别2/3/4预测。逐例五个finite logits、softmax与argmax契约通过；未使用输出映射、类别offset或端点硬编码，但端点监督后出现高度端点化。没有中间真实标签或外部数据，不能声称中间完成度校准、抗记忆能力或外部分布泛化已经验证。

full有543个suc/fail配对、validation421个，并非846/660个独立配对。原discovery中心在full的严格正分差比例7/9下降；同ROI分层的9个原中心均下降。平均分差增大、4档差增多和严格排序成功率不是同一指标。`analysis/head_delta_paired_roi_20260912_165601.json`中的delta_mean_separation以progress计，乘4才是reward差；同ROI均值区间多数跨0，不能宣称任务绑定普遍解决。

## 11项统计谬误核查

| 项目 | 判定与证据边界 |
| --- | --- |
| 1. Simpson反转 | 183条task按原样本数精确重构总体；报告上述唯一micro/macro反转。未由总体平均推断每task均改善。 |
| 2. 生态谬误 | 原始评分单位为逐例，重采样单位为视频组。任务均值、模型覆盖和配对均值不能推出每条指令改善或真实机器人成功率。 |
| 3. Berkson/选择偏差 | 用户接受的grounding cohort作为固定研究总体，不重新审核；筛选人口及28训练视频组不代表所有机器人任务。 |
| 4. Collider偏差 | 主target要求完整固定人口，没有按输出类别或正确性选样本。wrong-region不可用/padding差异按标签读取前的输入契约排除；共同子集仍是条件化人口，ROI分层亦非随机比较。 |
| 5. 忽略基率 | 分别报告上述三人口suc/fail基率，并要求两类同时提高。类别平衡训练不改变评价基率，总准确率不能代替分层准确率。 |
| 6. 均值回归 | 每条件与同输入、同checkpoint原生baseline配对。discovery最低MAE选scope/原中心仍有赢家偏差；完整已通过中心邻域与弱点均保留。 |
| 7. 幸存者偏差 | 51606条主target全部有效；完整九输入/61k包含失败点，不只汇总有效k。对照全部k均登记，wrong-region缺失与严格子集排除须单独计数，不能伪称完整人口。 |
| 8. 多重寻找效应 | 整个33轮、多个输入/scope/k与方法都曾探索，validation反复被查看。保留完整结果与负结果；checkpoint级Holm不校正全研究自适应选择。 |
| 9. 分析路径自由度 | 第33轮公式、rank4、损失、训练最终步、discovery选择与完整k并集规则先登记；其提出受前32轮失败影响，明确为自适应研究。对照完成前不改scope/k或代表中心。 |
| 10. 相关与因果混淆 | 同模型内部干预的计算效应可直接测量；ROI/head语义机制不能仅由准确率推断。low_rank同时改变head身份及训练曝光；B0移除新矩阵，但不隔离额外训练预算与容量。 |
| 11. 反向因果 | attention干预先于原生输出，推理不读取标签或配对对手；监督训练确实读取discovery标签，不能包装成无监督因果发现。 |

覆盖：11/11。主要结论限定为该数据集内的少样本监督、自适应算法改进；实现契约通过不等于科学解释全部成立。未重新训练，复现判定不提升为VERIFIED。完整控制与最终校正范围将追加于后。


## 指标依赖与最终校正口径补充

在真实标签及某配置的预测都仅为1/5时，该配置MAE严格等于`4*(1-accuracy_all)`；存在中间预测的baseline通常不满足此恒等式，所以不能进一步把ΔMAE等同于−4Δaccuracy。四指标来自同一批预测，本就相关，不能把多个门槛看作独立重复证据。总体准确率还由固定suc/fail基率加权。

最终含控制checkpoint的阈值记录数与实际具有有限p的检验数须分开报告；有不可用wrong-region的完整人口配对统计可能不完整。原±6与B0位于单独的严格配对JSON，并不属于主variant的target/wrong/low checkpoint Holm家族。最终自动核查将逐指标计数，不预先假定366条记录全部进入Holm；未针对整个33轮自适应研究校正的限制仍保持。


## 18:05 九输入共享的已测k交集

从全部九输入的完整三人口支持集合求交，得到共同已测k为24、32，且24/32在各输入原冻结k列表中均相邻。又从三人口原points.json逐项核验54条记录，四描述门槛及四CI方向全部通过。来源：`audit/head_delta_shared_k_intersection_20260912_180445.json`。这是完整结果的描述性交集，不新增scope/k/模型参数选择，不更换原discovery代表中心；不证明25–31等未测整数有效。Qwen text_image k24在validation的task等权MAE反而上升这一反例仍保留，交集只针对原登记的micro及准确率门槛。


## 首个完整控制（18:10）

Qwen text_image三个k的原±6/B0/区域/head控制已逐例审核并完成三人口配对。相对原±6及B0，全部k四指标CI有利；相对wrong-region则没有三人口共同的四CI支持，k40 full还有fail准确率下降。low-rank的k40 validation/old suc区间触零。严格子集排除12/14/14条均为suc，改变了基率；完整人口原bias/B0比较与该子集必须分开。精确数值和来源见[18:10记录](PROGRESS_20260912_1810.md)。其余八输入对照仍待完成，不外推该输入的控制结论。


## 18:37 评分标签文件来源一致性

当前已接受的`labels_for_scoring_only.json`与第32/33轮两模型共四份training_manifest保存的标签源SHA逐项相同，检查前后文件指纹也一致。仅核对字节来源，没有重新标注、审核grounding、重训或评分。证据：`audit/head_delta_label_source_identity_20260912_183711.json`。这补足最终结果对固定标签源的追溯，不把来源一致性当成外部泛化或独立验证。


## 第二个完整控制（18:40）

Qwen text_video六k全部相对原±6四指标CI有利；相对B0仅24/32/80在三人口四CI通过，k40的suc有负向CI。相对wrong-region无三人口共同四CI支持，k32的fail有负向CI；相对low-rank的原中心k64则suc下降而fail/总准确率提高。排除13/16/16条均为suc，完整人口与严格子集分开。精确区间和全部支持/反例见[18:10持续记录的18:40节](PROGRESS_20260912_1810.md)。不能把对原bias的优势外推成所有消融、所有k与各类都优胜。


## 18:57 第三个完整控制

Qwen video_text全部8k相对原±6四CI有利，但原中心k32相对B0与低排名head的suc点估计下降；k48相对B0、k40/48/64/80相对低排名head有负向suc区间。wrong-region没有三人口共同四CI支持。严格子集排除13/16/16条均为suc。完整数值与来源见[持续对照记录](PROGRESS_20260912_1810.md)。当前结论仍不能外推至尚未完成的六输入。


## 19:02 第四个完整控制

RR video_text相对原±6的共同四CI支持k32/40/48/64/80；B0没有共同四CI支持，原中心的suc+1.44pp区间跨0，k80有负向suc区间。wrong-region无共同四CI支持，原中心fail有负向区间；低排名head共同支持k24/32/40/48/64/80。严格共同子集排除19/23/23条全为suc。关键弱输入跨过主baseline的10pp门槛，不等于相对B0各类均明确改善。精确CI与来源见[持续对照记录](PROGRESS_20260912_1810.md)。


## 19:21 第五个完整控制

Qwen image_text原中心k32相对原±6/B0/低排名head四CI有利，相对wrong-region的suc区间跨0。其相对wrong的共同支持只在k8/16/24；高k有负向fail区间，B0及低排名head也有suc反例。排除12/14/14条均为suc。精确数值见[持续对照记录](PROGRESS_20260912_1810.md)，没有把部分低k的ROI支持外推到整个范围。


## 19:24 第六至八组完整控制

RR text_image、interleaved、text_video全部k相对原±6四CI有利，但三者原中心相对B0的suc区间都跨0。wrong/低排名head有明确suc/fail和MAE反例，严格子集排除全部来自suc。支持集合、关键区间及来源见[持续对照记录](PROGRESS_20260912_1810.md)。当前完整控制8/9，GPU计算已结束。


## 19:27 第九个完整控制

RR image_text相对原±6的三人口共同四CI支持k24/32/40/48/64/80；相对B0/wrong只有k8，相对低排名head为k24/32/40/80。原中心相对B0的suc区间跨0；本输入无任何wrong/padding排除。所有九输入已审阅，最终checkpoint的Holm范围及一致性审核尚待CPU收尾完成。精确数值见[持续对照记录](PROGRESS_20260912_1810.md)。


## 19:34 最终完整核查：11/11，统计解释保持CAUTION

全部9输入、61个冻结k及四类控制均已逐例审核；三人口共有732个配对控制比较、2928条指标行。四类控制各51606个最新样本记录：原±6、B0和低排名head均全部有效且输入匹配；wrong-region有226条不可构造、670条输入padding不匹配，严格共同子集的896个样本×条件排除不能称作896个不同个体。除RR image_text零排除外，其余排除均为suc。具体分母与全部反例保留在最终汇总。

B0与已有第32/32b同scope/k预测的22个来源文件逐值回放通过。不存在旧文件的k只作为本轮B0对照，不称作旧实验独立复现。

### 最终Holm范围与报告一致性

| 人口 | 最终阈值记录 | 每指标有限p检验数 | 原→最终四Holm<.05配置数 | 主target非Holm字段 |
| --- | ---: | ---: | ---: | --- |
| validation | 366 | 260 | 40→0 | 122条阈值记录全部逐值一致 |
| full_cohort | 366 | 260 | 50→0 | 122条阈值记录全部逐值一致 |
| old_holdout | 366 | 260 | 52→0 | 122条阈值记录全部逐值一致 |

260来自61个target、61个低排名head、8个完整wrong-region条件的两套阈值；其余53个wrong-region完整人口统计含无效输出，没有有限配对p。阈值重复仍按原程序保留，不能将其称作260个独立条件。原±6/B0的单独配对比较不在这个checkpoint的Holm家族。

**最终没有配置满足四项Holm p<.05。** 当前单侧sign-flip的最小p为1/5001≈.00019996；家族260条记录下最小Holm为260/5001≈.0519896021。原效应、CI和raw p没有变，改变的是家族大小及校正结果。没有事后删除重复阈值、缩小家族或增加重采样次数来跨过.05，也不把.05199写成通过.05。

原接受规则为效应量门槛及视频组CI方向，不是Holm<.05；该数值覆盖规则仍通过。这个区别必须同时报告，不能将结果宣传为独立确认性、多重校正后显著。所有33轮的自适应选择仍未被校正；full/validation/old重叠和五档端点化的限制保持。

### 全部对照后的综合判定

原baseline门槛共同支持54/61。进一步与相对同head原±6的四CI有利及总点增益至少10pp求交，得到51/61，九输入各有相邻已测k支持；这只是完成结果的描述性交集，未更换参数/scope/中心。来源：`audit/head_delta_joint_baseline_original_bias_support_20260912_192842.json`。矩阵相对B0、目标区域相对wrong、功能head相对低排名head均存在明确类别权衡和不确定性，不能推断所有组成在每个k、每类上都优胜。

主target所有非Holm字段的完整逐值一致证明，前述task异质性、端点预测分布与同ROI配对反例没有因对照加入而消失。11项核查已结合全部控制补齐：选择/collider与幸存者偏差按真实分母披露；多重寻找和分析路径按全程自适应处理；不作普遍ROI、任务绑定、中间完成度或跨架构因果/泛化声明。统计状态为ANALYZED，未独立重新训练，未新增人类复核。

完整数值与来源：[机械汇总](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911/analysis/head_delta_final_control_synthesis_20260912_193036/report.md)、同目录audit.json和paired_control_statistics.csv。validation/full共8张新对照PNG已实际查看，观察和PNG/PDF来源SHA见`audit/head_delta_control_visual_review_20260912_193353.json`，不把工具查看声称为独立人工审稿。
