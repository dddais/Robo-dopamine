# 跨模型研究与效果分析

遵守本文的的要求，进行auto research，针对现有研究背景与问题进行探索性研究。

进行长时间的充分的调研、思考、理论分析、实验，直到完成所有主线目标。

我要睡觉了，需要我确认的部分先跳过，进行你能进行的内容。

## 研究背景：

- 基于/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan.md的规划，进行了baseline 和attention steering的实验，目前发现在robo-dopamine的GRM上该方法的效果十分明显;
- 但是在qwen3-vl-8b和roboreward-8b的效果不是很明显
- 已有实验可供参考：
  - 目前已进行的GRM实验结果：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_GRM_summary.md
  - 目前已进行的跨模型实验：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel.md
  - 跨模型实验结果：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel_summary.md

## 主线目标1

- 遵守/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_addbase.md 完成新增baseline任务
- 直到新增baseline的所有实验完成，得到完整的实验结果总结mydata_bench/exp_plan_addbase_summary.md



## 主线目标2

- 调研相关文章，调研总结当前有哪些相关方法 ，目前已有部分可参考，见本文档“可参考代码/文献”部分，调研结果更新在“可参考代码/文献”部分。



## 主线目标3

- step1:基于调研的结果和方法，从理论角度思考提出能够改进目前attention steering的方案，如果需要可继续调研相关工作文章
- step2:实现step1提出的改进方案，进行实验验证，分析实验结果，如果效果不好则重复step1提出改进方案
- 验收目标：不断重复上述两个step，直到有一种方案改良原 attention steering方法同时满足下面的所有要求:
  - 四种模型（roboreward-8b 和qwen3-vl-8b ，robometer-4b,SOLE-R1-8B  ）中至少有三种模型满足“稳定有效”。
  - 稳定有效：指在所有输入构造中的至少三种输入下（最好包含官方的输入构造）都能稳定提升在数据集上的表现（MAE下降，suc,fail准确率提高，总准确率提高至少10%）(不要求所有top-k都能满足，至少存在一个top k 的范围满足)。
- 严禁使用端点hard coding这种类似作弊的方法！！！



## 基本原则（必须遵守）

- 目标代码库是/home/dais/workspace/Robo-Dopamine，不允许看其它Robo-Dopamine相关的文件夹，千万不要搞错了！！
- 尽量不修改现有代码库，如果需要修改，进行增量式修改，比如增加可选配置项等；
- 不允许进行git 操作本地已有的仓库，只能git clone开源仓库进行参考；
- 不允许对本地数据，结果等进行删除修改等操作，只能新增；
- 不用担心耗时，进行充分的调研、思考、理论分析，提出有道理的优雅的方案，严禁作弊的方法
- 注意GPU可能被其它程序使用，根据空余显存灵活使用，优先使用空闲GPU



## 可参考相关代码/文献



### 2026-09-08 本轮核验补充（研究进行中）

- 原列表开头的 PAI/PASTA 链接对应关系写反：PASTA 是 [2311.02262](https://arxiv.org/abs/2311.02262)，PAI 是 [2407.21771](https://arxiv.org/abs/2407.21771)。保留原始条目供追溯。
- 已核实计划内 14 篇论文的 arXiv 标题与摘要，并读取主要机制论文正文；新增 [Visual Contrastive Decoding, 2311.16922](https://arxiv.org/abs/2311.16922)。方法、来源与适用边界见 [文献与假设记录](mydata_bench/auto_research_addbase/LITERATURE_AND_HYPOTHESES_20260908.md)。
- 两个新增模型的官方代码通过新 clone 存在本目标目录的 `results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/references/`；未读取其他本地代码仓库。
- /home/dais/workspace/gaze-heads : [https://arxiv.org/pdf/2606.14703v1](https://arxiv.org/pdf/2606.14703v1)
- /home/dais/workspace/PAI : [https://arxiv.org/pdf/2311.02262](https://arxiv.org/pdf/2311.02262)
- /home/dais/workspace/PASTA : [https://arxiv.org/pdf/2407.21771](https://arxiv.org/pdf/2407.21771)
- **PASTA — Post-hoc Attention Steering for LLMs**（ICLR 2024）：[arXiv:2311.02262](https://arxiv.org/abs/2311.02262)。
- **PAI — Paying More Attention to Image**（ECCV 2024）：[arXiv:2407.21771](https://arxiv.org/abs/2407.21771)。
- **ASCD — Attention-Steerable Contrastive Decoding**（AAAI 2026）：[arXiv:2506.14766](https://arxiv.org/abs/2506.14766)，
- **CAST — Caption-Guided Visual Attention Steering**：[arXiv:2605.04641](https://arxiv.org/abs/2605.04641)。
- **Gaze Heads: How VLMs Look at What They Describe**：[arXiv:2606.14703](https://arxiv.org/abs/2606.14703)。
- **HAS — Highlight-guided Attention Steering for Multimodal LLM Video Summarization**：[arXiv:2607.17994](https://arxiv.org/abs/2607.17994)。
- **Arbitration Failure, Not Perceptual Blindness**：[arXiv:2604.09364](https://arxiv.org/abs/2604.09364)。
- **Inference-Time Attention Steering for VLA Driving Models**（ECCV 2026）：[arXiv:2608.17095](https://arxiv.org/abs/2608.17095)。
- **Attention is Case-Sensitive**（ECCV 2026）：[arXiv:2608.03711](https://arxiv.org/abs/2608.03711)。
- Localization heads :Your Large Vision-Language Model Only Needs AFew Attention Heads For Visual Grounding; [https://arxiv.org/pdf/2503.06287](https://arxiv.org/pdf/2503.06287)
- Your Model Already Knows: Attention-Guided Safety Filter for Vision-Language-Action Models : [https://arxiv.org/pdf/2606.09749](https://arxiv.org/pdf/2606.09749)
- Analyzing Multi-Head Self-Attention :[https://arxiv.org/pdf/1905.09418](https://arxiv.org/pdf/1905.09418)



## 原因分析

请你进行调研思考理论分析后补充。

**2026-09-08 阶段分析，待完整实验验证：** 原 +6/−6 bias 将目标/非目标相对赔率放大约 16 万倍，同时改变视觉与文字的总注意力比例。定位增强并不保证完成度判断正确，同视频不同指令的任务绑定可能仍然失败。实际 reward 读出位置、模态融合方式和 temporal patch 对齐在不同模型中也不同。本轮据此提出“保持视觉总注意力不变，仅重新分配视觉内部注意力”的候选；数学不变量和小规模真实前向已验证，尚不能据此宣称指标有效。详见上述文献与假设记录。

## 实验基础设置

增量式修改：代码修改不要影响到之前的实验运行，尽量以增量式的形式增加代码，比如加可选参数配置之类的

**数据集** ：/home/dais/workspace/data/mydata_v2/new ;/home/dais/workspace/Robo-Dopamine/results/mydata_bench/cohorts/auto_grounded_v2 (认为这就是正确的，不需要人工审核)

**config** 放在：/home/dais/workspace/Robo-Dopamine/mydata_bench/configs/v2_crossmodel

**输入**：video->text ; text->video; image->text ; text->image ;interleaved ;以上五种都需要尝试，方法最好能在大部分输入构造下work
**输出** 在：/home/dais/workspace/Robo-Dopamine/results/mydata_bench/experiments_v2_corssmodel/auto_research

**conda环境**：sam3:rewardbench-sam3 ；其它实验：robo-dopamine

**可用GPU**：0，1，2 ,3

**vpn** : proxy_on

**评价指标**：
1.MAE：按照roboreward的原定义
2.准确率：对于suc数据，lable=5,对于fail数据，lable=1；预测结果和lable相同的数量与概率。包括总准确率，suc,fail准确率，各个具体task的准备率分布。对于GRM这种输出连续进度的，用阈值区分开，统计两套阈值的情况：0.125，0.875；0.2，0.8
3.预测分布：统计在suc,fail数据上模型预测的lable分布，以及各个具体task上的模型预测分布；
4.pairwise区分度分析：因为数据集构成原理是1条suc数据，对应了1条或多条相同视频，不同instruction的fail数据，所以需要先找到suc数据所对应的fail数据，分析相同视频下不同instruction带来的影响。对于roboreward-8b,qwen这种输出离散的模型，计算统计配对数据中suc数据的预测值与fail数据的预测值的差值，把差值分成：负，0，1，2，3，4几档统计一下；
5.ranking head统计:列出具体的top 8，统计top 8,32,64在不同模型的重合度

## 最终有效方案

请你进行研究后，在这写明满足主线目标的最终方案



