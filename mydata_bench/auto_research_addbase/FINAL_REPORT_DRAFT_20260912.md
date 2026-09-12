# 跨模型 attention steering 自动研究总结（草稿）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate + research synthesis
- Origin Date: 2026-09-12
- Verification Status: ANALYZED（实现含实际前向回放；未独立重新训练）
- Overall Confidence: CAUTION（数据集内自适应探索）

第33轮主效果已满足计划的数值覆盖要求；本草稿不代表结案，完整四类对照及最终图表核查尚未全部结束。最终报告需由全部完成证据生成。

本轮证据来自当前 `/mnt/public1/dais/workspace/Robo-Dopamine`；原计划引用但本目录不存在的20260908文件没有作为完成证据。下文结果根目录 `OUT` 指 `results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911/`。遵守用户更新后的GPU限制，后续计算只用物理0、1；未重新审核已接受的grounding/cohort。旧数据和结果保留，新实验、来源审计、修复日志及负结果均另存。

## 三个主线的结论

**主线1：已有新增baseline复核完成。** 阅读并核对Robometer/SOLE实现及原始结果，旧契约测试13/13通过；重新扫描197,292条预测，复核468组汇总、13,176行任务统计、79,056行预测分布、54套排名和2,988行重合度。Robometer官方readout真实重跑：738个checkpoint张量strict load，progress/success输出误差均为0，ranking前后baseline误差也为0。未发现需要推翻并重跑整个旧矩阵的错误。SOLE原格式失败及wrong-region无法构造的样本保留，不从解释文字猜标签；官方递推布局与普通输入的差异单独报告。详见[审计记录](AUDIT_20260911.md)。

**主线2：18篇一手文献完成核验与方法对比。** 摘要、正文及SHA清单存于`OUT/references/`，没有上传本地数据。已纠正PASTA/PAI链接对应，区分注意力定位、功能head选择、视觉/任务对比分支、内部表示干预和视觉细节补充。主要机制来源是[PASTA](https://arxiv.org/abs/2311.02262)、[PAI](https://arxiv.org/abs/2407.21771)、[ASCD](https://arxiv.org/abs/2506.14766)、[CAST](https://arxiv.org/abs/2605.04641)、[VCD](https://arxiv.org/abs/2311.16922)、[CAD](https://arxiv.org/abs/2305.14739)、[V*](https://arxiv.org/abs/2312.14135)与[ReFT](https://arxiv.org/abs/2404.03592)。各论文的作者结果没有转述成本机器人数据的效果；18篇完整机制表与边界见[文献记录](LITERATURE_20260911.md)。

**主线3：第33轮固定门控attention增量的rank4变换达到数据集内主效果要求。** Qwen3-VL-8B为4/5输入、RoboReward-8B为5/5输入，均包含video→text。同一算法、损失、优化及固定训练步数用于两模型，各自训练一份参数。Robometer与SOLE也经过前期候选研究，但没有证明第33轮对它们有效；不把不同方法的局部成功拼成最终覆盖。前32轮路径、失败及其引出的下一步见[研究路径索引](RESEARCH_PATH_20260912.md)。

## 最终算法与代价

原±6使目标/其它视觉key的相对赔率乘`exp(12)≈162755`，还改变视觉/文字注意力比例。局部集中不保证冻结网络正确使用这些信息。前期模态质量守恒、软区域、时间平面约束、功能排名、视觉/任务对比、完整指令阻断、高分辨率及固定输出方向均有实际试验；同一方案未覆盖两模型各三输入。第32轮1792个逐head门控及全部已登记k扩展仍为Qwen3/RR2，RR video→text达不到总准确率+10pp。

第33轮固定第32轮最终门控，只对它产生的内部attention增量学习无常数项的方向变换。对选中layer/head/query：

```text
p0 = softmax(S)
pg = softmax(S + 6*g_v*r + 4*g_t*t)
d  = (pg-p0) @ V
o  = (original_SDPA + cast(d)) + cast(d @ A_l.T @ B_l.T)
```

S保留原causal/padding mask；r为scope内目标ROI的+1、其它视觉key的−1；t为当前指令key指示。d来自当前真实Q/K/V，包含前层干预影响。两次BF16加法保持实际代码顺序。layers8–35每层A为4×128、B为128×4，共28,672个新参数/模型；跨head、query、五输入、scope和k共享。原模型、输出embedding及原1,792个gate冻结。无类别offset、常数输出项或logit对比；一个完整forward，仍读原五类logits/softmax/argmax。rank4只限制新增变换的秩，并不等于小扰动：BA最大谱范数约10，不能据此保证近恒等或避免记忆。

监督使用discovery70（25 suc/45 fail、28个视频组），五输入×两scope×k32/64训练3epoch，类别平衡原五类CE加`.01*mean(||BA||²_F/128)`，Adam lr .001、clip1、batch1累积8；CPU初始化seed20260912，A高斯/sqrt128、B为0。每模型固定4,200次前后向、525次更新，仅使用最后一步checkpoint。加上第32轮gate训练，每模型8,400次前后向、1,050次更新；这些数字**不包括此前head profiling和33轮研究搜索**。不是零样本或同训练预算比较，额外局部QK/softmax/低秩计算也增加推理成本。

真实梯度、零增量返回baseline、B0回放及完整训练审计通过。没有按标签、任务ID、文件名、配对对手或预测类别选择推理输出。推理保留五类并不证明学到了中间完成度；端点监督后的输出集中现象另见下文。完整公式、训练和源码定位见[方法说明](METHOD_HEAD_DELTA_REFT_20260912.md)。

## 评价人口与接受规则

full846含268 suc/578 fail；validation660含209/451；old_holdout730含234/496并包含discovery70。三人口重叠，不能视为三次独立重复。validation没有参加本轮梯度训练，但整个研究多次查看其结果，后续方法选择受此前结果影响，因此是数据集内自适应探索。

MAE采用1–5奖励尺度上的逐例绝对误差均值，另报本数据的task等权macro MAE。本数据没有RoboReward论文benchmark的固定子集，不能将此处micro MAE直接称作论文的等权Overall。准确率为真实suc=5、fail=1的原生精确命中率，总准确率受两类基率加权。

对各自原生baseline要求MAE下降、suc/fail准确率同时提高、总准确率点增益至少10**个百分点**，且至少有两个相邻已测k支持；附加检查四项视频组95%CI方向。不是相对提高10%，也没有要求CI下界超过10pp。CI按原视频组带全部指令重采样5,000次。两套阈值0.125/0.875与0.2/0.8在原生五档结果逐字段相同，不算独立重复。

9输入共61个冻结主条件、51,606条target预测及各846条baseline已逐例核验；54/61条件在三人口同时通过四门槛及四CI方向。各输入scope与原中心来自discovery，完整列表为该scope下所有通过中心的相邻k并集；以下不按validation重新挑中心。

### validation660：原discovery中心

MAE以1–5奖励单位计；准确率为百分数，箭头为原生baseline→方法。

| 模型 / 输入 | scope / k | MAE | 总准确率% | suc准确率% | fail准确率% |
| --- | --- | --- | --- | --- | --- |
| Qwen / image_text | all_frames / 32 | 1.5061→0.7152 | 21.82→82.12 | 53.59→81.34 | 7.10→82.48 |
| Qwen / text_image | all_frames / 32 | 1.6773→1.2545 | 27.12→68.64 | 82.78→94.74 | 1.33→56.54 |
| Qwen / text_video | last_frame / 64 | 1.4152→0.5939 | 15.00→85.15 | 43.06→62.20 | 2.00→95.79 |
| Qwen / video_text | last_frame / 32 | 1.4803→0.7212 | 27.27→81.97 | 51.67→61.72 | 15.96→91.35 |
| RoboReward / image_text | last_frame / 64 | 0.8848→0.5894 | 62.73→85.15 | 55.98→64.59 | 65.85→94.68 |
| RoboReward / interleaved | all_frames / 32 | 1.6242→0.6591 | 23.33→83.18 | 46.41→63.64 | 12.64→92.24 |
| RoboReward / text_image | all_frames / 64 | 1.5500→0.6848 | 17.27→82.88 | 44.98→50.72 | 4.43→97.78 |
| RoboReward / text_video | all_frames / 32 | 2.1106→0.7909 | 18.64→80.15 | 58.85→74.64 | 0.00→82.71 |
| RoboReward / video_text | all_frames / 32 | 0.9333→0.7576 | 65.91→81.06 | 62.20→70.81 | 67.63→85.81 |

同一中心的配对变化及视频组95%CI，准确率增益单位为百分点：

| 模型 / 输入 | ΔMAE [CI] | Δ总pp [CI] | Δsuc pp [CI] | Δfail pp [CI] |
| --- | --- | --- | --- | --- |
| Qwen / image_text | -0.791 [-0.886, -0.698] | +60.303 [+56.829, +63.759] | +27.751 [+20.500, +35.023] | +75.388 [+70.474, +80.090] |
| Qwen / text_image | -0.423 [-0.492, -0.352] | +41.515 [+37.252, +45.652] | +11.962 [+7.726, +16.418] | +55.211 [+49.074, +60.986] |
| Qwen / text_video | -0.821 [-0.920, -0.724] | +70.152 [+67.259, +72.855] | +19.139 [+12.136, +26.190] | +93.792 [+91.125, +96.154] |
| Qwen / video_text | -0.759 [-0.861, -0.660] | +54.697 [+51.067, +58.346] | +10.048 [+3.846, +16.346] | +75.388 [+70.737, +79.954] |
| RoboReward / image_text | -0.295 [-0.376, -0.218] | +22.424 [+19.068, +25.914] | +8.612 [+3.774, +13.810] | +28.825 [+24.130, +33.641] |
| RoboReward / interleaved | -0.965 [-1.083, -0.845] | +59.848 [+55.940, +63.764] | +17.225 [+9.661, +24.762] | +79.601 [+74.304, +84.669] |
| RoboReward / text_image | -0.865 [-0.972, -0.760] | +65.606 [+62.995, +68.161] | +5.742 [+0.469, +11.331] | +93.348 [+90.265, +96.272] |
| RoboReward / text_video | -1.320 [-1.451, -1.179] | +61.515 [+58.233, +64.715] | +15.789 [+9.091, +22.597] | +82.705 [+78.471, +86.609] |
| RoboReward / video_text | -0.176 [-0.271, -0.085] | +15.152 [+11.869, +18.524] | +8.612 [+3.791, +13.876] | +18.182 [+14.035, +22.609] |

### full846：同一冻结中心的复核

| 模型 / 输入 | k | MAE baseline→方法 | Δ总 / suc / fail pp |
| --- | ---: | --- | --- |
| Qwen / image_text | 32 | 1.4279→0.6478 | +60.76 / +24.25 / +77.68 |
| Qwen / text_image | 32 | 1.5969→1.1596 | +43.03 / +10.82 / +57.96 |
| Qwen / text_video | 64 | 1.3877→0.5768 | +70.21 / +19.78 / +93.60 |
| Qwen / video_text | 32 | 1.4444→0.6667 | +56.97 / +11.57 / +78.03 |
| RoboReward / image_text | 64 | 0.8203→0.5118 | +22.46 / +9.70 / +28.37 |
| RoboReward / interleaved | 32 | 1.5863→0.6087 | +61.11 / +17.16 / +81.49 |
| RoboReward / text_image | 64 | 1.4917→0.6336 | +65.96 / +5.22 / +94.12 |
| RoboReward / text_video | 32 | 2.1111→0.7447 | +61.82 / +14.93 / +83.56 |
| RoboReward / video_text | 32 | 0.8818→0.6761 | +15.60 / +8.58 / +18.86 |

### 三人口共同支持的全部相邻已测k对

| 模型 / 输入 | 相邻已测k对 |
| --- | --- |
| Qwen / image_text | 8/16, 16/24, 24/32, 32/40, 64/80 |
| Qwen / text_image | 24/32, 32/40 |
| Qwen / text_video | 24/32, 32/40, 40/48, 48/64, 64/80 |
| Qwen / video_text | 8/16, 16/24, 24/32, 64/80 |
| RoboReward / image_text | 24/32, 32/40, 40/48, 48/64, 64/80 |
| RoboReward / interleaved | 24/32, 32/40, 40/48, 48/64, 64/80 |
| RoboReward / text_image | 8/16, 16/24, 24/32, 32/40, 40/48, 48/64, 64/80 |
| RoboReward / text_video | 24/32, 32/40 |
| RoboReward / video_text | 8/16, 16/24, 24/32, 32/40, 40/48, 48/64, 64/80 |

所有九输入共同支持已测k24、32，二者在各冻结列表中相邻；该交集从完整结果描述性求得，没有替换原中心，不外推25–31或其它未测整数。`OUT/audit/head_delta_shared_k_intersection_20260912_180445.json`复核三人口54条原记录。Qwen interleaved六个discovery条件失败，未扩full；last/k8的reward全同baseline不代表logits逐值相同。

主效果反例全部保留：Qwen image_text k48损伤suc；Qwen video_text k40/48的suc区间跨0；RR image_text k8的suc区间跨0、k16总增益不足10pp；RR text_video k48/80的suc区间不足，k64孤立通过，不能跨过失败点把高k连成稳定范围。

## 四类对照与新增项的证据

全部61个冻结k的四类控制均完成逐例来源、原生输出、参数及输入匹配审核；四类控制共244个条件，每条件保留846条尝试记录。原±6、B0及低排名head均完整有效；wrong-region不可构造与padding不匹配单独保留。

**相对原方法的补充复核也通过：** 将主baseline的四门槛及CI支持集，与相对同head原±6四CI有利且总点增益≥10pp的集合求交，得到51/61个条件；全部九输入仍各有相邻已测k支持。这是完整冻结结果的描述性交集，没有更换scope、原中心或参数。RR video_text的交集从k32开始，不能把相对baseline共享的24/32误称为相对原±6也共同成立。证据为`OUT/audit/head_delta_joint_baseline_original_bias_support_20260912_192842.json`。

下表保留**每类对照**在三人口共同四CI有利的全部已测k；此表本身不另要求相对各消融达到10pp，也不排除主baseline未通过的k。相邻关系只按原冻结列表判断。

| 模型 / 输入 | 原±6 | B0固定gate | wrong-region | 低排名head |
| --- | --- | --- | --- | --- |
| qwen/image_text | 8,16,24,32,40,64,80 | 16,24,32,40,64 | 8,16,24 | 24,32 |
| qwen/text_image | 24,32,40 | 24,32,40 | 无 | 24,32 |
| qwen/text_video | 24,32,40,48,64,80 | 24,32,80 | 无 | 24,40,48 |
| qwen/video_text | 8,16,24,32,40,48,64,80 | 8,16,24,64,80 | 无 | 8,16,24 |
| roboreward/image_text | 24,32,40,48,64,80 | 8 | 8 | 24,32,40,80 |
| roboreward/interleaved | 24,32,40,48,64,80 | 无 | 无 | 24,32,40,48,64,80 |
| roboreward/text_image | 8,16,24,32,40,48,64,80 | 8,16,24,32,40,48 | 8 | 8,24,40,48 |
| roboreward/text_video | 24,32,40,48,64,80 | 24 | 无 | 32,40 |
| roboreward/video_text | 32,40,48,64,80 | 无 | 无 | 24,32,40,48,64,80 |

原discovery中心在validation相对原±6的配对变化与95%CI如下，准确率单位为百分点：

| 模型 / 输入 | k | ΔMAE [CI] | Δ总pp [CI] | Δsuc pp [CI] | Δfail pp [CI] |
| --- | ---: | --- | --- | --- | --- |
| qwen/image_text | 32 | -1.145 [-1.251, -1.039] | +67.273 [+63.440, +70.973] | +37.799 [+30.244, +45.327] | +80.931 [+76.256, +85.193] |
| qwen/text_image | 32 | -0.436 [-0.509, -0.361] | +41.212 [+36.842, +45.319] | +9.569 [+5.687, +13.810] | +55.876 [+49.773, +61.590] |
| qwen/text_video | 64 | -0.765 [-0.852, -0.679] | +71.970 [+68.938, +74.810] | +29.187 [+22.476, +36.364] | +91.796 [+88.155, +94.931] |
| qwen/video_text | 32 | -0.848 [-0.955, -0.744] | +52.576 [+49.028, +56.049] | +6.699 [+0.490, +13.132] | +73.836 [+69.444, +78.139] |
| roboreward/image_text | 64 | -0.429 [-0.524, -0.339] | +27.727 [+24.006, +31.579] | +13.876 [+7.805, +20.197] | +34.146 [+29.075, +39.374] |
| roboreward/interleaved | 32 | -0.165 [-0.248, -0.085] | +12.727 [+9.767, +15.790] | +20.096 [+14.078, +26.318] | +9.313 [+5.656, +12.910] |
| roboreward/text_image | 64 | -0.606 [-0.694, -0.521] | +57.121 [+53.604, +60.656] | +9.091 [+3.810, +14.623] | +79.379 [+74.664, +84.000] |
| roboreward/text_video | 32 | -0.580 [-0.687, -0.469] | +64.242 [+60.863, +67.442] | +24.402 [+18.268, +30.732] | +82.705 [+78.471, +86.609] |
| roboreward/video_text | 32 | -0.432 [-0.534, -0.331] | +26.515 [+22.807, +30.204] | +6.220 [+1.442, +11.111] | +35.920 [+30.973, +40.750] |

B0对照表明矩阵项在多种输入提供额外效应，但没有对所有k、所有类别的全面优势。尤其RR image_text/interleaved/video_text的原中心相对B0，MAE、总准确率和fail准确率区间有利，suc区间跨0；RR text_image/text_video原中心的suc区间亦跨0。Qwen video_text原中心的suc相对B0点估计下降；Qwen image_text k48、Qwen text_video k40、RR video_text k80等还有负向suc区间。

区域/head机制也有反例：Qwen image_text仅低k8/16/24得到相对wrong-region的共同四CI支持，高k64/80的fail反而下降；RR text_image只在k8、RR image_text只在k8有共同四CI支持，其余六输入没有。RR interleaved原中心相对wrong-region的suc+47.72pp，但fail−3.77pp [−7.21,−.66]。Qwen text_video原中心相对低排名head的suc−16.33pp [−21.47,−10.99]，RR text_image原中心则−17.77pp [−23.21,−12.44]；RR text_image k16相对低排名head的MAE还上升+.1435 [.0166,.2702]。不能由总体收益推出ROI或功能排名在全范围、每类都优胜。

原±6/B0比较使用完整人口；区域/head共同比较的分母如下。除RR image_text外，排除全部来自suc，不能隐去这种基率变化；按k的精确排除与全部效应见完整对照汇总。

| 输入组 | validation共同n | full共同n | old共同n | 排除suc（validation/full/old） |
| --- | ---: | ---: | ---: | --- |
| Qwen image_text/text_image；RR text_image/interleaved | 648 | 832 | 716 | 12/14/14 |
| Qwen video_text/text_video | 647 | 830 | 714 | 13/16/16 |
| RR video_text/text_video | 641 | 823 | 707 | 19/23/23 |
| RR image_text | 660 | 846 | 730 | 0/0/0 |

对原±6的比较保持同head、输入和原生读出，但增加了监督与训练容量；B0保留训练后的标量gate，仅移除矩阵项，不能隔离额外优化预算与参数自由度。low-rank在这里指**低排名head控制**，与rank4矩阵的“低秩”不是同一个概念；它同时改变head身份和训练曝光。

wrong-region只换空间r为等大小、不相交的区域，仍保留完整视频、当前任务t、固定gate、A/B、原SDPA及未选heads。它不是视觉移除实验。target未优于wrong既不能证明目标ROI独有贡献，也不能推出“视觉无用”。严格共同子集的排除先于标签读取，原±6/B0完整人口与区域/head子集的人口不同。

## 任务分布、五档输出、配对与head排名

逐任务MAE/准确率、suc/fail五类预测分布、逐task五类分布和配对六档差值均保存在三人口exports。183条三人口主统计由直方图、task和配对重构后算术一致；仅5条没有恶化task。唯一micro与task等权macro的方向反转是Qwen text_image k24 validation：micro ΔMAE−0.175758，macro +0.017662（13任务改善、7恶化、8不变）。总体改善不能外推为每个task改善。

validation31/61、full29/61、old30/61条件没有2/3/4预测；原五类finite logits、softmax和argmax仍通过实现契约。没有端点硬编码，但输出高度端点化，没有中间真实标签或外部测试，不能宣称中间完成度校准。若某条件预测和标签都仅1/5，则其MAE=4×(1−accuracy)；baseline可能有中间类别，因此不能同样把ΔMAE写成−4Δaccuracy。

full为543个suc/fail配对、validation为421个，多条fail共享suc，不能当成846/660个独立配对。原中心full严格正分差比例7/9下降；同ROI的9个原中心严格正分差比例全部下降。同ROI均值分差区间多跨0，均值或4档差增多不等于严格排序普遍提高，**没有证明任务绑定问题普遍解决**。ROI分层的`delta_mean_separation`以progress计，乘4才是reward差；分组任务难度不同，不能把组间差异当随机因果效应。

沿用stage8的监督功能排名，按8-head组worst-class NLL变化选择；不是raw attention mass。全部五输入×两scope的具体top8和跨模型top8/32/64重合已列于[排名简表](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911/analysis/head_gate_frozen_rank_summary_20260912_150211/summary.md)。十个同scope/输入比较的top8重合均为0；top64重合为9–29/64，索引从0开始。相同层/head编号重合不证明两个模型语义功能等价。

## 统计解释与可复核性

{{FINAL_STATISTICS}}

视频组sign-flip p依赖零假设下组差值符号可交换，最小分辨率1/5001；没有随机分派，不作为严格随机化试验。各CI未经全研究多重校正，四指标来自同一预测且互相关联。完整数据、失败范围、冻结policy、原始p和Holm p保留；不能把最后一个checkpoint的校正包装成对33轮自适应搜索的控制。没有独立重训，因此统计报告为ANALYZED，不升级成独立复现VERIFIED。

11/11统计谬误检查与控制边界见[最终统计核查](STATISTICAL_REVIEW_HEAD_DELTA_REFT_20260912.md)。当前已接受评分标签文件与第32/33轮两模型四份训练manifest的SHA一致；这是文件来源核对，没有重新标注、复审grounding或训练。

## 结果与复用入口

固定policy为`OUT/selection_learned_head_delta_reft_v1.json`；固定gate为`OUT/learned_head_gates_v1/training/{model}/final_gates.json`，固定A/B为`OUT/learned_head_delta_reft_v1/training/{model}/final_adapter.json`。方法实现为[head_delta_reft.py](head_delta_reft.py)，推理入口为[head_delta_inference.py](head_delta_inference.py)。已完成文件按example_id只读复用，不覆盖原结果；有排他锁的相同参数worker只是计算分工，不是独立实验重复。

例如在仓库根目录复用两个video→text冻结矩阵的命令如下。入口要求原完整k并集，不能只传描述性交集24/32；已有结果会复用，因此再次执行不等于独立重新推理。新数据验证需另建输入、输出和登记，不混写当前结果。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/dais/miniconda3/envs/robo-dopamine/bin/python -B \
  -m mydata_bench.auto_research_addbase.head_delta_inference \
  --model qwen --protocols video_text --population full_cohort \
  --ks 8 16 24 32 40 48 64 80 --scopes last_frame

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/dais/miniconda3/envs/robo-dopamine/bin/python -B \
  -m mydata_bench.auto_research_addbase.head_delta_inference \
  --model roboreward --protocols video_text --population full_cohort \
  --ks 8 16 24 32 40 48 64 80 --scopes all_frames
```

主target完整报告、任务/分布/配对CSV、相邻k图、行为图、训练算子图及四类对照图的最终索引如下。数据、JSON、PNG及PDF均保留；图只连接已测k，不证明其间整数有效。

{{COMPLETION_EVIDENCE}}

当前证据支持的科学结论是：少样本监督下，对固定attention steering产生的内部增量学习共享低秩方向变换，能在当前接受数据集的两模型、多种输入和相邻已测k上改善原生评分。它没有证明通用跨架构迁移、独立外部分布泛化、普遍ROI特异性、每任务改善或中间进度校准。后续独立数据/重复训练研究属于新的验证工作，不把这些未验证命题混入本计划的数据集内数值验收。
