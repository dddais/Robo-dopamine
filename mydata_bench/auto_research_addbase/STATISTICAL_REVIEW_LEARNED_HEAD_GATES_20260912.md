# 第32轮逐head门控：完整主条件的统计核查

## Material Passport

状态：ANALYZED。对象为第32轮六个冻结输入、18个完整主条件，在validation660、full846、old_holdout730上的同样本配对结果。三人口相互重叠；18是不同配置数，不是独立试验数。原公式/完整输入/固定最终1792门控的实现已另行审计，当前不是重新训练复现。第32b轮新增k仍在运行，本记录不对其局部预测评分。原第32轮控制尚未全部完成，后续追加其边界。

来源为三批主target检查点`analysis/checkpoint_head_gates_{first_image_text,second_inputs,third_inputs}_{population}_20260912_{1423,1436,1447}/`（依次对应三批），逐文件SHA和完整重构结果在`audit/head_gates_statistical_structure_20260912_145638.json`。该JSON中的`independent_conditions=18`只表示去除重复阈值后的18个不同配置，不能按字面理解为统计独立重复。

## 主要判定

原第32轮为Qwen三个输入、RoboReward两个输入获得相邻已测k支持，**未达总覆盖**。RR video_text三k虽然四指标CI方向有利，总准确率只提高约6.7–7.5pp，不足10pp。Qwen video_text k48的validation MAE CI跨零；支持范围限k64/80。RR image_text点估计总增益约10.4–11.5pp，CI下界没有超过10pp，不声称效应量的置信下界达门槛。

误差/准确率区间按video_sha256带全部指令重采样5000次。当前区间未针对完整研究的反复探索校正；单checkpoint的Holm也不覆盖全部32轮。sign-flip数值另外依赖视频组差值在零假设下的符号可交换性；它不是本数据设计随机分派产生的严格随机化检验，不用其p值代替效应量门槛。

18条件×3重叠人口共54条汇总全部同时有改善和恶化任务；样本加权MAE与任务等权MAE方向在这54条中没有反转，且按相同task样本数可精确重构总体。具体到validation，Qwen image_text k48有13个任务MAE下降、11个上升；Qwen text_video k80有15个下降、9个上升。总体改善绝不等于各task都改善。

全部主条件846/846有效。validation原生中间类别2/3/4的预测条数仍为70–303，具体数值和全部五档分布有导出；这与逐例五类softmax/argmax核验一致。输出集中于端点可由端点标签的监督训练造成，不能将其说成无监督发现，也不能只用中间类别仍存在证明泛化。

## 11项统计谬误核查

| 项目 | 核查结论 |
| --- | --- |
| Simpson反转 | 同task样本数重构54条总体MAE，未出现所有任务同方向而总体反向；54条均有正负任务混合。task-macro另报，不把micro优势泛化到每任务。 |
| 生态谬误 | 基本评分是逐例预测，区间单位是原视频组；没有从模型/任务均值推出每条指令改善。 |
| Berkson选择偏差 | 使用用户接受的grounding cohort，不重审核。该筛选总体、70条训练样本和28训练视频组不能代表全部机器人任务分布。 |
| Collider偏差 | 主target要求完整固定人口，未按预测正确与否纳入样本。已通过候选的选择仍有赢家偏差；ROI分层是事后观察诊断，不能视作随机干预。 |
| 忽略基率 | validation209 suc/451 fail，full268/578，old_holdout234/496；必须同时提高两类。训练类别平衡不改变实际评价基率。 |
| 均值回归 | 比较同输入、同模型的原生baseline和干预。discovery最低MAE选中心仍有极值选择偏差，RR video_text在full增益减弱即为明确反例。 |
| 幸存者偏差 | 15228条主干预及六个完整baseline均通过逐例核验；原三k、所有失败均保留，RR不因未达标被排除。控制中的不可用区域和输入padding差异须另行报告。 |
| 多重寻找效应 | 包含32轮迭代、多个模型/输入/scope/k，且validation被反复查看。报告完整矩阵与负结果，拒绝独立确认性显著宣称；checkpoint级Holm不能消除全研究自适应偏差。 |
| 分析路径自由度 | 第32轮训练步数/最终checkpoint/损失/参数与选择规则先登记；原六候选失败后的32b另外登记，不修改原失败结论，明确validation影响后续研究路径。 |
| 相关与因果混淆 | 同输入attention干预与输出变化可直接比较；目前相同ROI配对多处CI跨零，不能反推出任务语义绑定普遍解决，亦不能在控制完成前把收益全归ROI。 |
| 反向因果 | 计算干预先于输出、推理不读标签，方向明确；由准确率收益推断哪个head已学到因果语义仍无充分依据。 |

11/11已核查。当前是数据集内少样本监督、自适应研究结果；原第32轮未满足两模型各三输入的总目标，第32b与原全部对照继续。


## 15:22 完整控制和最终统计补充

原六输入及全部控制均已完成，完整审计索引与三人口统计/导出位于原watch_full/watch_statistics的complete文件。final主target效应量/CI与三批先前报告逐值一致，Holm因比较范围扩大另计。108阈值记录只有18个不同target配置，其中15通过描述门槛；Qwen video_text k48的validation MAE CI仍跨零。

对同head原±6，五个输入所有k的四指标CI有利；**RR interleaved例外**：三个k的suc显著增加、fail与总准确率明显下降，MAE区间跨零。该类权衡直接阻止“全面优于原方法”的表述。ROI对照也不是普遍全指标优胜：Qwen video_text明确suc/fail权衡、RR interleaved的fail不确定、RR video_text的MAE不确定、Qwen image_text的validation MAE/suc不确定；Qwen text_video仅k64/80获得四指标支持。

wrong区域不可用及其retry产生的padding变化均在读标签前固定排除，严格common与完整人口单独报告。排除多落在suc，改变了子集基率；不能把严格子集结果当完整人口估计。low-rank包含未参与数据梯度、只受正则的head，改变身份与训练覆盖，不是纯排名因果对照。11项核查的生存/选择/因果边界据此补齐，原第32轮仍未达总目标。
