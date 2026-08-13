# 门控方案实验探究

请你进行auto research，针对现有现象进行探索性研究，直到完成所有主线目标。

必要时进行文献搜索，开源仓库参考。我要睡觉了，需要我确认的部分先跳过，进行你能进行的内容。

既要进行真实实验，也需要进行原理性分析。

## 已有背景：

- 基于/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan.md的规划，进行了baseline 和attention steering的实验，目前发现在robo-dopamine的GRM上该方法的效果十分明显，但是在qwen3-vl-8b和roboreward-8b的效果不是很明显。
- 考虑使用门控方案，核心是利用baseline,wrong region和target的关系来消除非特异化干扰对模型的影响，详情参考 /home/dais/workspace/Robo-Dopamine/mydata_bench/gate_method_design.md

## 主线目标1

基于robo-dopamine的 GRM的attention ranking 和 steering 方法，尝试实现各种门控方案：/home/dais/workspace/Robo-Dopamine/mydata_bench/gate_method_design.md （参考下面列出的“可参考相关代码/文献”以及可以网络调研搜索其它相关的论文，比如调节attention权重等。）

- 分析提出方案，最好原理上比较优雅，看起来能work
- 尝试各种门控方案，使其能够改良原 attention steering方法，在五种输入构造下都能稳定提升roboreward-8b 和qwen3-vl-8b 在数据集上的表现（MAE下降，suc,fail准确率提高）



## 基本原则（必须遵守）

- 尽量不修改现有代码库，如果需要修改，进行增量式修改，比如增加可选配置项等；
- 不允许进行git 操作本地已有的仓库，只能git clone开源仓库进行参考；
- 不允许对本地数据，结果等进行删除修改等操作，只能新增；
- 探索过程有价值的尝试可以详细记录在 /home/dais/workspace/Robo-Dopamine/mydata_bench/auto_explore_gate.md ；本文档简要更新探索的内容及结果；



## 可参考相关代码/文献

- /home/dais/workspace/gaze-heads : [https://arxiv.org/pdf/2606.14703v1](https://arxiv.org/pdf/2606.14703v1)
- /home/dais/workspace/PAI : [https://arxiv.org/pdf/2311.02262](https://arxiv.org/pdf/2311.02262)
- /home/dais/workspace/PASTA : [https://arxiv.org/pdf/2407.21771](https://arxiv.org/pdf/2407.21771)



## 相关文档

- gate方案设计文档（需要更新）：/home/dais/workspace/Robo-Dopamine/mydata_bench/gate_method_design.md
- gate探索文档（需要更新）： /home/dais/workspace/Robo-Dopamine/mydata_bench/auto_explore_gate.md
- cross_model探索文档（之前的结果，不更新）： /home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel_auto_explore.md ；/home/dais/workspace/Robo-Dopamine/mydata_bench/auto_explore_crossmodel.md



## 实验基础设置

增量式修改：代码修改不要影响到之前的实验运行，尽量以增量式的形式增加代码，比如加可选参数配置之类的

**数据集** ：/home/dais/workspace/data/mydata_v2/new ;/home/dais/workspace/Robo-Dopamine/results/mydata_bench/cohorts/auto_grounded_v2 (认为这就是正确的，不需要人工审核)

**config** 放在：/home/dais/workspace/Robo-Dopamine/mydata_bench/configs/v2_gate

**输入**：video->text ; text->video; image->text ; text->image ;interleaved ;以上五种都需要尝试，方法最好能在大部分输入构造下work

**输出** 在：/home/dais/workspace/Robo-Dopamine/results/mydata_bench/experiments_v2_gate

**conda环境**：sam3:rewardbench-sam3 ；其它实验：robo-dopamine

**可用GPU**：0，1，2

**vpn** : proxy_on

**评价指标**：
1.MAE：按照roboreward的原定义
2.准确率：对于suc数据，lable=5,对于fail数据，lable=1；预测结果和lable相同的数量与概率。包括总准确率，suc,fail准确率，各个具体task的准备率分布。对于GRM这种输出连续进度的，用阈值区分开，统计两套阈值的情况：0.125，0.875；0.2，0.8
3.预测分布：统计在suc,fail数据上模型预测的lable分布，以及各个具体task上的模型预测分布；
4.pairwise区分度分析：因为数据集构成原理是1条suc数据，对应了1条或多条相同视频，不同instruction的fail数据，所以需要先找到suc数据所对应的fail数据，分析相同视频下不同instruction带来的影响。对于roboreward-8b,qwen这种输出离散的模型，计算统计配对数据中suc数据的预测值与fail数据的预测值的差值，把差值分成：负，0，1，2，3，4几档统计一下；
5.ranking head统计:列出具体的top 8，统计top 8,32,64在不同模型的重合度

## 待探索的实验：

