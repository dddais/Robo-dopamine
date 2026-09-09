# 跨模型研究与效果分析

遵守本文的的要求，进行auto research，针对现有研究背景与问题进行探索性研究。

进行长时间的充分的调研、思考、理论分析、实验，直到完成所有主线目标。

我要睡觉了，需要我确认的部分先跳过，进行你能进行的内容。

## 基本原则（必须遵守）

- 本地代码只能参考/home/dais/workspace/Robo-Dopamine的相关代码，其它仓库不允许看
- 尽量不修改现有代码库，如果需要修改，进行增量式修改，比如增加可选配置项等；
- 不允许进行git 操作本地已有的仓库，只能git clone开源仓库进行参考；
- 不允许对本地数据，结果等进行删除修改等操作，只能新增；
- 不用担心耗时，进行充分的调研、思考、理论分析，提出有道理的优雅的方案，严禁作弊的方法

## 研究背景：

- 基于/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan.md的规划，进行了baseline 和attention steering的实验，目前发现在robo-dopamine的GRM上该方法的效果十分明显;
- 但是在qwen3-vl-8b和roboreward-8b的效果不是很明显
- 已有实验可供参考：
  - 目前已进行的GRM实验结果：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_GRM_summary.md
  - 目前已进行的跨模型实验：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel.md
  - 跨模型实验结果：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel_summary.md



## 主线目标1

- 补充baseline：补充robometer，SOLE-R1-8B 作为baseline。
  - robometer:[https://arxiv.org/pdf/2603.02115](https://arxiv.org/pdf/2603.02115) ;/home/dais/workspace/model/Robometer-4B
  - SOLE-R1-8B:[https://arxiv.org/pdf/2603.28730v2](https://arxiv.org/pdf/2603.28730v2) ;/home/dais/workspace/model/SOLE-R1-8B
  - step1:在现有代码的基础上，参考mydata_bench/qwen_eval ； mydata_bench/roboreward_eval ；mydata_bench/attention_eval 等代码构建方法，构建mydata_bench/meter_eval ； mydata_bench/top_eval
  - step2:对照要求：/home/dais/workspace/Robo-Dopamine/mydata_bench/check.md 中的“attention mask相关”和“baseline相关”，检查代码实现是否有问题
  - step3：类似于已有的GRM实验和roboreward-8b的实验，进行新增baseline的实验：包含baseline和attention mask的实验，attention mask的实验包含不同输入构造（必须包含官方输入构造）
  - step4:类似于/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel_summary.md，把新增的两个baseline的实验结果做一个文档mydata_bench/exp_plan_addbase_summary.md总结记录下来
  - 注意：新增的模型可能输入构造，token结构等存在差异，请你自行思考决定代码具体实现方式，必要情况下可以用可修改配置的方式实现多种方案。



## 基本原则（必须遵守）

- 目标代码库是/home/dais/workspace/Robo-Dopamine，千万不要搞错了！！
- 尽量不修改现有代码库，如果需要修改，进行增量式修改，比如增加可选配置项等；
- 不允许进行git 操作本地已有的仓库，只能git clone开源仓库进行参考；
- 不允许对本地数据，结果等进行删除修改等操作，只能新增；
- 不用担心耗时，进行充分的调研、思考、理论分析，提出有道理的优雅的方案，严禁作弊的方法



## 实验基础设置

增量式修改：代码修改不要影响到之前的实验运行，尽量以增量式的形式增加代码，比如加可选参数配置之类的

**数据集** ：/home/dais/workspace/data/mydata_v2/new ;/home/dais/workspace/Robo-Dopamine/results/mydata_bench/cohorts/auto_grounded_v2 (认为这就是正确的，不需要人工审核)

**config** 放在：/home/dais/workspace/Robo-Dopamine/mydata_bench/configs/v2_crossmodel_addbase

**输入**：video->text ; text->video; image->text ; text->image ;interleaved ;以上五种都需要尝试，得到结果
**输出** 在：/home/dais/workspace/Robo-Dopamine/results/mydata_bench/experiments_v2_addbase/

**conda环境**：sam3:rewardbench-sam3 ；其它实验：robo-dopamine

**可用GPU**：0，1，2

**vpn** : proxy_on

**评价指标**：
1.MAE：按照roboreward的原定义
2.准确率：对于suc数据，lable=5,对于fail数据，lable=1；预测结果和lable相同的数量与概率。包括总准确率，suc,fail准确率，各个具体task的准备率分布。对于GRM这种输出连续进度的，用阈值区分开，统计两套阈值的情况：0.125，0.875；0.2，0.8
3.预测分布：统计在suc,fail数据上模型预测的lable分布，以及各个具体task上的模型预测分布；
4.pairwise区分度分析：因为数据集构成原理是1条suc数据，对应了1条或多条相同视频，不同instruction的fail数据，所以需要先找到suc数据所对应的fail数据，分析相同视频下不同instruction带来的影响。对于roboreward-8b,qwen这种输出离散的模型，计算统计配对数据中suc数据的预测值与fail数据的预测值的差值，把差值分成：负，0，1，2，3，4几档统计一下；
5.ranking head统计:列出具体的top 8，统计top 8,32,64在不同模型的重合度


