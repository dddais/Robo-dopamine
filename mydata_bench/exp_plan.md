进行下面的实验：（先默认当前sam3首帧+后续帧tracking方案得到的结果是正确的，不进行人工审核，直接实验）
每个实验单独一个config，输出在对应的不同的目录下

conda环境：
sam3:rewardbench-sam3 
其它实验：robo-dopamine

可用GPU：0，1，2，3

使用/参考文档：
Robo-Dopamine/mydata_bench/exp_use.md
Robo-Dopamine/mydata_bench/experience.md

具体实验：
baseline实验：尽量符合原模型各自的要求
1.roboreward-8b baseline 实验1：注意输入格式是先text再video
2.roboreward-8b baseline 实验2：注意输入格式是先video再text
3.qwen-3-vl-8b baseline实验1：注意输入格式是先text再video
4.qwen-3-vl-8b baseline实验2：注意输入格式是先video再text
5.robo-dopamine GRM baseline 实验1：forward模式
6.robo-dopamine GRM baseline 实验2：incremental模式


attention实验：
共同设置：
包含wrong target,low rank等对照实验；
注意ranking都用data/ljx_lfz_task/new/ranking_data.jsonl里面的数据；
steer时 top-k head选择top8,top 32,top 64
6.roboreward-8b attention实验1：-bias只加在最后一帧的非target区域，+bias只加在最后一帧的target区域
7.roboreward-8b attention实验2：-bias加在所有帧的非target区域，+bias加在所有帧的target区域
8.qwen-3-vl-8b attention实验1：-bias只加在最后一帧的非target区域，+bias只加在最后一帧的target区域
9.qwen-3-vl-8b attention实验2：-bias加在所有帧的非target区域，+bias加在所有帧的target区域
10.GRM attention实验1：forward模式；-bias只加在after_high的非target区域，+bias只加在after_high的target区域
11.GRM attention实验2：forward模式；-bias加在after_high，before_high的非target区域，+bias加在after_high,before_high的target区域
12.GRM attention实验1：incremental模式；-bias只加在after_high的非target区域，+bias只加在after_high的target区域
13.GRM attention实验2：incremental模式；-bias加在after_high，before_high的非target区域，+bias加在after_high,before_high的target区域
14.roboreward-8b attention实验3：注意输入格式是先video再text:-bias只加在最后一帧的非target区域，+bias只加在最后一帧的target区域
15.roboreward-8b attention实验4：注意输入格式是先video再text:-bias加在所有帧的非target区域，+bias加在所有帧的target区域


评价指标：
1.MAE：按照roboreward的原定义
2.准确率：对于suc数据，lable=5,对于fail数据，lable=1；预测结果和lable相同的数量与概率。包括总准确率，suc,fail准确率，各个具体task的准备率分布。对于GRM这种输出连续进度的，用阈值区分开，统计两套阈值的情况：0.125，0.875；0.2，0.8
3.预测分布：统计在suc,fail数据上模型预测的lable分布，以及各个具体task上的模型预测分布；GRM相应的用阈值转换为离散的lable（五档）。
4.pairwise区分度分析：因为数据集构成原理是1条suc数据，对应了1条或多条相同视频，不同instruction的fail数据，所以需要先找到suc数据所对应的fail数据，分析相同视频下不同instruction带来的影响。对于roboreward-8b,qwen这种输出离散的模型，计算统计配对数据中suc数据的预测值与fail数据的预测值的差值，把差值分成：负，0，1，2，3，4几档统计一下；对于GRM这种输出连续进度的模型，类似地计算计算统计配对数据中suc数据的进度值与fail数据的进度值的差值，把差值分成：负，10%，20%，30%，50%，60%，70%，80%，90% 几档统计一下
5.ranking head统计:列出具体的top 8，统计top 8,32,64在不同模型的重合度
