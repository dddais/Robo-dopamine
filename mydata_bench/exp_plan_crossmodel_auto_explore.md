# 跨模型研究与效果分析
请你进行auto research，针对现有现象进行探索性研究，直到找到表现好的方法。必要时进行文献搜索，开源仓库参考，我要睡觉了，需要我确认的部分先跳过，进行你能进行的内容。
## 已有背景：

- 基于/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan.md的规划，进行了baseline 和attention steering的实验，目前发现在robo-dopamine的GRM上该方法的效果十分明显，但是在qwen3-vl-8b和roboreward-8b的效果不是很明显。
- 基于/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel.md 进行了初步探索，但是效果仍然不是很理想。 

## 原因分析

### video input
我怀疑有可能和输入形式有关。GRM输入两个时刻的三视图，而另外两个模型直接输入完整视频序列，video processor采样8帧，并且每两帧成为一个temporal span。一方面我觉得这个两帧作为一个temporal span可能会有问题，比如target img token找不准等。因此已经进行了下面的 **改造输入格式video->image：** 实验来验证

### casual masking
自回归的casual masking可能影响了attention mask的作用 。因此已经进行了下面 **改造输入格式：GRM类型输入**的实验。


## 实验基础设置

增量式修改：代码修改不要影响到之前的实验运行，尽量以增量式的形式增加代码，比如加可选参数配置之类的

**数据集** ：/home/dais/workspace/data/mydata_v2/new

**config** 放在：/home/dais/workspace/Robo-Dopamine/mydata_bench/configs/v2_crossmodel

**输出** 在：/home/dais/workspace/Robo-Dopamine/results/mydata_bench/experiments_v2_corssmodel

**conda环境**：
sam3:rewardbench-sam3 
其它实验：robo-dopamine

**可用GPU**：0，1，2，3

**vpn** : proxy_on

**评价指标**：
1.MAE：按照roboreward的原定义
2.准确率：对于suc数据，lable=5,对于fail数据，lable=1；预测结果和lable相同的数量与概率。包括总准确率，suc,fail准确率，各个具体task的准备率分布。对于GRM这种输出连续进度的，用阈值区分开，统计两套阈值的情况：0.125，0.875；0.2，0.8
3.预测分布：统计在suc,fail数据上模型预测的lable分布，以及各个具体task上的模型预测分布；
4.pairwise区分度分析：因为数据集构成原理是1条suc数据，对应了1条或多条相同视频，不同instruction的fail数据，所以需要先找到suc数据所对应的fail数据，分析相同视频下不同instruction带来的影响。对于roboreward-8b,qwen这种输出离散的模型，计算统计配对数据中suc数据的预测值与fail数据的预测值的差值，把差值分成：负，0，1，2，3，4几档统计一下；
5.ranking head统计:列出具体的top 8，统计top 8,32,64在不同模型的重合度

## 实验具体安排



### 改造输入格式video->image：

不直接输入video，而是先采样img，以img的形式输入，类似于GRM，只是输入的内容不同（一个是三视角after，before,一个是时序的单视角）。然后以这个输入格式进行实验：
1.roboreward-8b baseline :先输入text再输入img
2.roboreward-8b baseline :先输入img再输入text
3.roboreward-8b +attention ranking + steering ：先输入text再输入img
4.roboreward-8b +attention ranking + steering ：先输入img再输入text
5.qwen3-vl-8b baseline: 先输入text再输入img
6.qwen3-vl-8b baseline:先输入img再输入text
7.qwen3-vl-8b +attention ranking + steering ：先输入text再输入img
8.qwen3-vl-8b +attention ranking + steering ：先输入img再输入text

9.roboreward-8b +attention ranking + steering ：先输入text再输入img;-bias加在所有帧的非target区域，+bias加在所有帧的target区域
10.roboreward-8b +attention ranking + steering ：先输入img再输入text;-bias加在所有帧的非target区域，+bias加在所有帧的target区域
11.qwen3-vl-8b +attention ranking + steering ：先输入text再输入img;-bias加在所有帧的非target区域，+bias加在所有帧的target区域
12.qwen3-vl-8b +attention ranking + steering ：先输入img再输入text;-bias加在所有帧的非target区域，+bias加在所有帧的target区域

### 改造输入格式：GRM类型输入
考虑到自回归模型casual masking的影响，不把image text完全分开，不直接输入video，而是类似于GRM那种，把img嵌入到text prompt当中。当然具体的system prompt内容需要改一下。

1.roboreward-8b +attention ranking + steering ：-bias加在所有帧的非target区域，+bias加在所有帧的target区域
2.roboreward-8b +attention ranking + steering ：-bias加在最后一帧的非target区域，+bias加在最后一帧的target区域
3.qwen3-vl-8b +attention ranking + steering ：-bias加在所有帧的非target区域，+bias加在所有帧的target区域
4.qwen3-vl-8b +attention ranking + steering ：-bias加在最后一帧的非target区域，+bias加在最后一帧的target区域

上述实验已做完，接下来请你进行探索的实验：