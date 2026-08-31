# 跨模型研究与效果分析

遵守本文的的要求，进行auto research，针对现有研究背景与问题进行探索性研究。

进行长时间的充分的调研、思考、理论分析、实验，直到完成所有主线目标。

我要睡觉了，需要我确认的部分先跳过，进行你能进行的内容。

## 研究背景：

- 基于/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan.md的规划，进行了baseline 和attention steering的实验，目前发现在robo-dopamine的GRM上该方法的效果十分明显，但是在qwen3-vl-8b和roboreward-8b的效果不是很明显,只有/home/dais/workspace/Robo-Dopamine/results/mydata_bench/experiments_v2_corssmodel/attention_13_roboreward_interleaved_all_frames/exp_record.md 这一个配置能够给roboreward-8b带来提升，其它的配置效果都比较差。
- 目前进行了初步探索（见本文档的 “原因分析” 和 “已经完成的实验”部分 ），但是效果仍然不是很理想。

## 主线目标1

- 调研相关文章，调研总结当前有哪些相关方法 ，目前已有部分可参考，见本文档“可参考代码/文献”部分，调研结果更新在“可参考代码/文献”部分。



## 主线目标2

- 基于调研的结果和方法，从理论角度思考能够改进目前attention steering的方法
- 不断尝试各种方法，直到能够改良原 attention steering方法:两种模型（roboreward-8b 和qwen3-vl-8b ）各自在五种输入构造中的至少两种输入下都满足top k =8 ,32 ,64能稳定提升在数据集上的表现（MAE下降，suc,fail准确率提高），像GRM的表现那样。
- 严禁使用端点hard coding这种类似作弊的方法！！！



## 基本原则（必须遵守）

- 尽量不修改现有代码库，如果需要修改，进行增量式修改，比如增加可选配置项等；
- 不允许进行git 操作本地已有的仓库，只能git clone开源仓库进行参考；
- 不允许对本地数据，结果等进行删除修改等操作，只能新增；
- 不用担心耗时，进行充分的调研、思考、理论分析，提出有道理的优雅的方案，严禁作弊的方法



## 可参考相关代码/文献

- /home/dais/workspace/gaze-heads : [https://arxiv.org/pdf/2606.14703v1](https://arxiv.org/pdf/2606.14703v1)
- /home/dais/workspace/PAI : [https://arxiv.org/pdf/2311.02262](https://arxiv.org/pdf/2311.02262)
- /home/dais/workspace/PASTA : [https://arxiv.org/pdf/2407.21771](https://arxiv.org/pdf/2407.21771)



## 原因分析



### video input

我怀疑有可能和输入形式有关。GRM输入两个时刻的三视图，而另外两个模型直接输入完整视频序列，video processor采样8帧，并且每两帧成为一个temporal span。一方面我觉得这个两帧作为一个temporal span可能会有问题，比如target img token找不准等。因此已经进行了下面的 **改造输入格式video->image：** 实验来验证

### casual masking

自回归的casual masking可能影响了attention mask的作用 。因此已经进行了下面 **改造输入格式：GRM类型输入**的实验。

## temporal prior

GRM一次推理只获得before 和after 两个时刻的图像，而roboreward-8b获取了8个时刻的图像，这有可能时序的进展成为模型的主要判断依据，指令token的贡献被稀释。目前还没进行实验验证。

## 其它原因

请你进行调研思考理论分析后补充。

## 实验基础设置

增量式修改：代码修改不要影响到之前的实验运行，尽量以增量式的形式增加代码，比如加可选参数配置之类的

**数据集** ：/home/dais/workspace/data/mydata_v2/new ;/home/dais/workspace/Robo-Dopamine/results/mydata_bench/cohorts/auto_grounded_v2 (认为这就是正确的，不需要人工审核)

**config** 放在：/home/dais/workspace/Robo-Dopamine/mydata_bench/configs/v2_crossmodel

**输入**：video->text ; text->video; image->text ; text->image ;interleaved ;以上五种都需要尝试，方法最好能在大部分输入构造下work
**输出** 在：/home/dais/workspace/Robo-Dopamine/results/mydata_bench/experiments_v2_corssmodel/auto_research

**conda环境**：sam3:rewardbench-sam3 ；其它实验：robo-dopamine

**可用GPU**：0，1，2

**vpn** : proxy_on

**评价指标**：
1.MAE：按照roboreward的原定义
2.准确率：对于suc数据，lable=5,对于fail数据，lable=1；预测结果和lable相同的数量与概率。包括总准确率，suc,fail准确率，各个具体task的准备率分布。对于GRM这种输出连续进度的，用阈值区分开，统计两套阈值的情况：0.125，0.875；0.2，0.8
3.预测分布：统计在suc,fail数据上模型预测的lable分布，以及各个具体task上的模型预测分布；
4.pairwise区分度分析：因为数据集构成原理是1条suc数据，对应了1条或多条相同视频，不同instruction的fail数据，所以需要先找到suc数据所对应的fail数据，分析相同视频下不同instruction带来的影响。对于roboreward-8b,qwen这种输出离散的模型，计算统计配对数据中suc数据的预测值与fail数据的预测值的差值，把差值分成：负，0，1，2，3，4几档统计一下；
5.ranking head统计:列出具体的top 8，统计top 8,32,64在不同模型的重合度

## 已经完成的实验



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

### 改造输入格式：GRM类型嵌入式输入

考虑到自回归模型casual masking的影响，不把image text完全分开，不直接输入video，而是类似于GRM那种，把img嵌入到text prompt当中。当然具体的system prompt内容需要改一下。

1.roboreward-8b +attention ranking + steering ：-bias加在所有帧的非target区域，+bias加在所有帧的target区域
2.roboreward-8b +attention ranking + steering ：-bias加在最后一帧的非target区域，+bias加在最后一帧的target区域
3.qwen3-vl-8b +attention ranking + steering ：-bias加在所有帧的非target区域，+bias加在所有帧的target区域
4.qwen3-vl-8b +attention ranking + steering ：-bias加在最后一帧的非target区域，+bias加在最后一帧的target区域

## 待探索的实验：

