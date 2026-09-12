# 第33轮方法说明：固定attention门控增量的低秩变换

本方法的完整主target已经在当前数据集内达到Qwen4输入、RoboReward5输入的覆盖门槛；此说明记录算法与适用边界。完整主效果见[17:00记录](PROGRESS_20260912_1700.md)，全部控制完成后的科学结论须结合最终总结阅读。它是分别对两模型训练同一种算法，不能称一份参数跨模型迁移，也未证明对Robometer/SOLE同样有效。

## 为什么研究这个变化

原±6同时增强目标、压低其它视觉key，目标/背景相对赔率乘exp(12)，约162755倍。定位增强可能损害参照物、背景关系或原先正确的成功判断。早期研究分别尝试模态质量守恒、soft区域、功能head选择、视觉/任务对比分支与高分辨率；有局部收益，但没有同一方案覆盖两模型各三输入。

第32轮把固定强度改成1792个逐head全局门控，缓解了部分冲突，却在RoboReward video→text达不到总准确率+10pp；完整k扩展后仍为Qwen3/RR2。第33轮保留这些最终门控，对它们产生的内部证据差学习方向变换，使冻结的后续网络处理变换后的证据。这个机制借鉴[ReFT](https://arxiv.org/abs/2404.03592)的内部表示干预位置，结合attention增量约束；不是LoReFT/DiReFT原样复现。相关文献与已核实原文见[文献记录](LITERATURE_20260911.md)。

## 计算定义

对层l、选中head h和当前query q，令S为已有causal/padding mask的QK分数，V为当前value。空间向量r在该scope的目标ROI为+1、其它视觉key为−1、域外为0；任务向量t在当前指令key为1、其它为0。固定门控g来自第32轮最终checkpoint，分别为视觉g_v与任务g_t，取值均在[0,1]：

```text
p0 = softmax(S)
pg = softmax(S + 6 * g_v[l,h] * r + 4 * g_t[l,h] * t)
d  = (pg - p0) @ V
o  = (original_SDPA + cast_to_BF16(d)) + cast_to_BF16(B[l] @ A[l] @ d)
```

最后一行以列向量记法表示变换；代码用行向量右乘A转置和B转置。屏蔽key不会被有限offset重新开放，全遮挡行按原算子契约归零。实际新增项在float32计算，向原attention dtype转换后执行**两次独立加法**；不声称与一次高精度求和逐值等价。

每层A为4×128，B为128×4，在layers8–35各有一对，共28672个新参数。A/B跨选中head、query、五种输入、scope和k共享；原1792个gate固定。未选head保留原SDPA；所有选中head的全部query行使用同一公式。各层d来自当前真实Q/K/V，已包含前层干预影响，不能视为第32轮预缓存的固定向量。

没有常数项、输出层更新、类别offset或logit对比。原五类candidate logits、softmax、argmax和`ANSWER: `读出保持，实际每样本一个完整模型forward；局部仍需额外QK、两次softmax、差分乘V及低秩乘法，不能称与baseline同推理成本。外部rank/head/ROI属于固定实验输入契约，不由标签、任务ID、文件名、配对对手或预测类别选择输出规则。

## 可核验性质及其边界

- B=0时新矩阵项为0，完整输出逐值回放第32轮固定门控。真实smoke已核验；全人口B0与已有同scope/k的第32/32b预测逐值比较是本轮控制的一部分。
- gate=0使pg=p0，从而d=0；即使B非零，也逐值返回baseline。该性质已用真实前向核验。它说明新矩阵依赖attention差，不证明该差在语义上正确。
- 在精确实数下，额外修正满足`||BA d|| <= ||BA||_2 ||d||`，秩至多4；整个`I+BA`变换不一定低秩。训练得到BA最大谱范数Qwen10.2893、RR10.5179，I+BA最小奇异值0.12760/0.02153。因此“rank4”不能解释成小扰动、近恒等或不会记忆；上述局部界也不是整个模型的放大倍数。
- 变换后的head输出一般不能再解释为原value上的非负归一化attention平均，亦不保持视觉/文字总注意力。它是内部表示干预，不继续声称第一轮的模态质量守恒。

## 固定训练与选择

两模型使用相同算法与训练超参，但各自训练参数：

| 项目 | 固定设置 |
| --- | --- |
| 数据 | discovery70，25 suc/45 fail，28个视频组；五种布局全部使用 |
| 原模型 | 语言、视觉、输出embedding与第32轮门控全部冻结 |
| 初始化 | CPU seed20260912，A为高斯/sqrt128，B精确0；两模型相同初值 |
| 损失 | 类别平衡的原五类CE + `.01 * mean_l(||B_l A_l||_F² / 128)` |
| 优化 | Adam，lr .001，梯度clip1，batch1累积8 |
| 训练矩阵 | 五输入×两scope×k32/64，3epoch，4200前后向/525更新 |
| checkpoint | 仅固定最后一步；epoch文件只审计，不用于选最优 |
| 总训练预算 | 加上第32轮gate训练，每模型8400前后向/1050更新；不含更早head profiling/研究搜索 |
| discovery评估 | 两模型×五输入×两scope×k8/32/64，共60条件；完整核验后才评分 |
| full选择 | 每输入按discovery最低MAE确定scope；该scope所有通过中心的相邻k并集一次冻结 |

不是零样本或与原±6同预算的比较。discovery重代入选择仍有过拟合风险；validation660不参加本轮梯度，但整个33轮多次查看其结果，故只能称数据集内自适应探索。full846和old_holdout730还包含训练/选择样本，不能当额外独立验证。

## 结果与代码定位

所有结果相对于`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911/`：

| 内容 | 路径 |
| --- | --- |
| 评分前固定policy | `selection_learned_head_delta_reft_v1.json` |
| 冻结最终gate | `learned_head_gates_v1/training/{model}/final_gates.json` |
| 冻结最终A/B | `learned_head_delta_reft_v1/training/{model}/final_adapter.json` |
| 实际梯度/零增量审计 | `learned_head_delta_reft_v1/actual_gradient_gate_v1.json` |
| 完整训练审计 | `learned_head_delta_reft_v1/fixed_training_audit_v1.json` |
| 完整探索/选择 | `learned_head_delta_reft_v1/selection_complete_discovery_v1.json` |
| 九输入与相邻k覆盖 | `learned_head_delta_reft_v1/coverage.json` |
| 每输入全部target审计 | `learned_head_delta_reft_v1/audits/full_{model}_{protocol}.json` |
| 五输入top8与跨模型top8/32/64重合 | `analysis/head_gate_frozen_rank_summary_20260912_150211/` |

排名是stage8的discovery监督功能排名（8-head组的worst-class NLL变化），不是raw attention mass。第33轮原样沿用，所有五输入、两个scope的top8均列于该简表；相同索引的跨模型重合不证明语义功能等价。

实现：[head_delta_reft.py](head_delta_reft.py)；训练/真实smoke：[head_delta_worker.py](head_delta_worker.py)；冻结推理：[head_delta_inference.py](head_delta_inference.py)；完整对照：[head_delta_controls.py](head_delta_controls.py)。GPU worker不得导入会清空GPU可见性的CPU调度模块。用户更新限制后的所有执行仅物理GPU0/1；GPU2禁止使用。

无须重训即可从完整checkpoint与逐样本文件重算统计。所有新增输出保留原文件；此说明没有启动新的参数搜索、外部数据验证或模型迁移实验。端点输出集中、任务异质性和同ROI配对局限见[11项统计核查](STATISTICAL_REVIEW_HEAD_DELTA_REFT_20260912.md)。


## wrong-region对照的机制含义

该控制只把空间向量r的选中区域换成同scope、等大小且不相交的区域；同一完整视频、指令tokens t、固定g_v/g_t、head集合与A/B仍保留。原SDPA路径和未选heads也继续读取原输入。它不是遮去视觉信息的实验，不能由target与wrong接近推出“视觉无用”。而且softmax含视觉与任务联合offset，d一般不能无条件分解成独立视觉项与任务项。首个Qwen text_image完整控制仅支持“当前未证明目标ROI独有贡献”这一窄结论，见[18:10记录](PROGRESS_20260912_1810.md)；额外全因素或视觉移除实验不属于本轮已登记的四类控制。


## 19:39 最终收尾完成

所有九输入61k的主条件、原±6/B0/wrong-region/低排名head控制与三人口最终报告均完成。732个配对控制比较、2928条指标汇总通过来源和算术核验；主target除Holm外全部字段逐值不变，22份已有B0参考逐值回放通过。validation/full八张新控制PNG已实际查看，见OUT的`audit/head_delta_control_visual_review_20260912_193353.json`。

主baseline支持54/61；与相对原±6的四CI及总点增益≥10pp求交为51/61，全部九输入仍有相邻已测k，Qwen4/RR5输入达到计划数值要求。原代表中心、scope、固定checkpoint没有变。最终Holm每指标260个有限p记录，最小.0519896021，四项同时<.05的配置为0，探索性解释与所有任务/配对/消融反例保持。

正式结论、原始效应与CI、四类对照、全部图表/CSV、方法成本和复用命令见[最终总结](FINAL_REPORT_20260912.md)。主计划已增加最终有效方案。277个队列任务均释放，GPU0/1本研究占用归零，CPU完成链结束。本计划三个主线的必需研究工作均已完成；独立外部验证属于后续新研究，未冒称完成。
