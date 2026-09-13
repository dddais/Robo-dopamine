# 跨模型研究与效果分析

遵守本文的的要求，进行auto exp，针对现有研究背景与问题进行实验，直到完成所有主线目标

我要睡觉了，需要我确认的部分先跳过，进行你能进行的内容。

## 基本原则（必须遵守）

- 本地代码只能参考/home/dais/workspace/Robo-Dopamine的相关代码，其它仓库不允许看
- 尽量不修改现有代码库，如果需要修改，进行增量式修改，比如增加可选配置项等；
- 不允许进行git 操作本地已有的仓库，只能git clone开源仓库进行参考；
- 不允许对本地数据，结果等进行删除修改等操作，只能新增；

## 研究背景：

- 基于/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan.md的规划，进行了baseline 和attention steering的实验，目前发现在robo-dopamine的GRM上该方法的效果十分明显;
- 但是在qwen3-vl-8b和roboreward-8b的效果不是很明显
- 已有实验可供参考：
  - 目前已进行的GRM实验结果：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_GRM_summary.md
  - 目前已进行的跨模型实验：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel.md
  - 跨模型实验结果：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel_summary.md ；/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_addbase_summary.md



## 主线目标1

- 目前不是在整个数据集上进行的实验，而是在部分双端点满足的情况下进行的实验。接下来需要把整个实验再在完整的数据集上跑一遍，grounding结果用results/mydata_bench/cohorts/auto_grounded_v2_release；没有可用grounding结果的数据就遵循baseline的结果
- 
- 进行完整实验
  - baseline:GRM-8B（只做forward模式），roboreward-8b，qwen3-vl-8b，robometer-4b，sole-r1。（下文用模型A代替这五个模型）
  - SAS方法（semantic attention steering）：即原attention steering方法，对五个模型分别进行ranking，steering
  - 输入构造：
    - 各个模型的官方输入构造；
    - 先text再image
    - 先image再text
    - 类似GRM的交错输入构造
  - 施加bias：
    - -bias只加在最后一帧（对GRM是after high）的非target区域，+bias只加在最后一帧的target区域
    - -bias加在所有帧的非target区域，+bias加在所有帧的target区域
  - 对照：
    - wrong target
    - low rank
  - top k :
    - 8
    - 32
    - 64
  - 总计：
    - baseline: 5个模型*3种输入（大约）=15组实验
    - SAS:5个模型**1个方法**2种bias**2种对照*3种topk**3种输入（大约）=180组实验



## 主线目标2

- 总结实验结果
  - 每个模型的所有实验分别汇总成一个该模型的summary文档，一共五个文档，每个文档包括：
    - baseline 在整个数据集上的MAE，准确率（总，成功，失败）
    - SAS 方法 在整个数据集上的MAE，准确率（总，成功，失败）
    - SAS 方法的对照组 在整个数据集上的MAE，准确率（总，成功，失败）
    - 效果最好的top k，bias，输入构造及其效果（MAE，准确率，pairwise区分度）
    - 
  - 所有模型的结果汇总成一个总文档，包括：
    - 各个模型baseline的效果
    - 各个模型效果最好的SAS配置及其效果



## 实验基础设置

增量式修改：代码修改不要影响到之前的实验运行，尽量以增量式的形式增加代码，比如加可选参数配置之类的

**数据集** ：/home/dais/workspace/data/mydata_v2/new 

grounding结果：/home/dais/workspace/Robo-Dopamine/results/mydata_bench/cohorts/auto_grounded_v2_release (认为这就是正确的，不需要人工审核) 数据集中没有grounding结果的部分就用baseline替代

**config** 放在：/home/dais/workspace/Robo-Dopamine/mydata_bench/configs/v2_basic_method

**输入**：image->text ; text->image ;interleaved；官方 ;以上四种都需要尝试，得到结果 **输出** 在：/home/dais/workspace/Robo-Dopamine/results/mydata_bench/experiments_v2_basic_method/

**conda环境**：sam3:rewardbench-sam3 ；其它实验：robo-dopamine

**可用GPU**：0，1，2 ，3

**vpn** : proxy_on

**评价指标**：
1.MAE：按照roboreward的原定义
2.准确率：对于suc数据，lable=5,对于fail数据，lable=1；预测结果和lable相同的数量与概率。包括总准确率，suc,fail准确率，各个具体task的准备率分布。对于GRM这种输出连续进度的，用阈值区分开，统计两套阈值的情况：0.125，0.875；0.2，0.8
3.预测分布：统计在suc,fail数据上模型预测的lable分布，以及各个具体task上的模型预测分布；
4.pairwise区分度分析：因为数据集构成原理是1条suc数据，对应了1条或多条相同视频，不同instruction的fail数据，所以需要先找到suc数据所对应的fail数据，分析相同视频下不同instruction带来的影响。对于roboreward-8b,qwen这种输出离散的模型，计算统计配对数据中suc数据的预测值与fail数据的预测值的差值，把差值分成：负，0，1，2，3，4几档统计一下；
5.ranking head统计:列出具体的top 8，统计top 8,32,64在不同模型的重合度