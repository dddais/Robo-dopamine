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
**输出** 在：/home/dais/workspace/Robo-Dopamine/results/mydata_bench/experiments_v2_corssmodel

**conda环境**：sam3:rewardbench-sam3 ；其它实验：robo-dopamine

**可用GPU**：0，1，2

**vpn** : proxy_on

**评价指标**：
1.MAE：按照roboreward的原定义
2.准确率：对于suc数据，lable=5,对于fail数据，lable=1；预测结果和lable相同的数量与概率。包括总准确率，suc,fail准确率，各个具体task的准备率分布。对于GRM这种输出连续进度的，用阈值区分开，统计两套阈值的情况：0.125，0.875；0.2，0.8
3.预测分布：统计在suc,fail数据上模型预测的lable分布，以及各个具体task上的模型预测分布；
4.pairwise区分度分析：因为数据集构成原理是1条suc数据，对应了1条或多条相同视频，不同instruction的fail数据，所以需要先找到suc数据所对应的fail数据，分析相同视频下不同instruction带来的影响。对于roboreward-8b,qwen这种输出离散的模型，计算统计配对数据中suc数据的预测值与fail数据的预测值的差值，把差值分成：负，0，1，2，3，4几档统计一下；
5.ranking head统计:列出具体的top 8，统计top 8,32,64在不同模型的重合度

## 最终改良方法：布局特定的双向因果加权 Attention Steering

> 本节是自动探索完成后的方法补充。完整运行过程、失败分支、统计检验和最终裁决见 `mydata_bench/auto_explore.md`。

### 方法动机

原始 ranking 主要按 target 区域的 attention mass 选 head，容易找到“看见目标”的 localization heads，但这些 heads 不一定会使用目标信息判断指令是否完成。固定方向、固定权重地干预这些 heads，还会出现以下问题：

- 同一个 head 可能同时把 success 和 fail 推向高分，只改善一侧；
- 某些 head 的有效方向实际是负方向，不能根据 `+bias` 的结果直接代数反号；
- top-k 从8增加到64时，弱或有害 heads 会累积过强剂量；
- target-region 与 wrong-region 都有效时，变化可能只是一般视觉扰动，而不是目标区域特异效应。

最终方法不再把“原始 attention 大”直接等同于“对 reward 判断有用”，而是对每个模型、每种输入布局单独执行 paired causal profiling，选择能同时帮助 success/fail 且具有空间特异性的 heads。

### 数据划分与隔离

所有划分以 `video_sha256`/video cluster 为单位，不能让同一视频因换了 instruction 而跨 partition：

| 阶段 | 数据 | 唯一视频簇 | 用途 |
| --- | ---: | ---: | --- |
| ranking / head profiling | 34 suc + 32 paired fail = 66 条 video-instruction records | 34 | 候选 head、方向和权重 |
| screening | 35 records | 11 | 判断冻结方法是否值得进入 held-out |
| final held-out | 697 records（223 suc、474 fail） | 254 | 只做最终确认 |

三部分的 video-cluster 重合为0。冻结 partition fingerprint：

```text
4d65d16add4129b3afc0b16f24495189d47580e5688b7167b4921c0cd528fee2
```

早期发现原846条 evaluation cohort 包含34个 ranking 视频后，已把这些 ranking clusters 和11个 screening clusters 全部排除；最终 H1–H4 只使用修正后的697条 held-out。

### 第一步：构建 layout-specific 候选池

每个模型、每个输入构造独立执行，禁止跨模型或跨布局复用 ranking。候选池取该模型/布局已有三种 ranking 的 top-64 并集：

- raw target attention mass；
- target excess mass；
- visual enrichment。

候选池只用于缩小逐 head 因果实验范围，最终顺序不再由原始 attention mass 决定。

### 第二步：逐 head paired causal profiling

对每个候选 head，在34条 development success 和32条同视频 paired fail 上分别运行：

1. `baseline`：不干预；
2. `target`：只对真实 target region 施加该 head 的 bias；
3. `wrong`：对等 token 数、尽量远离 target 的错误区域施加相同 bias。

使用 teacher-forced `ANSWER: 1..5` choice logits 计算连续 margin：

- success 侧要求 reward-5 margin 增加；
- fail 侧要求 reward-5 margin 降低，即不再把错误指令判断为高完成度；
- target 干预必须优于 wrong-region 干预，保证空间特异性；
- 聚合时先按 task 求均值，再对 task 求均值，避免大 task 支配 ranking。

这里的 development 标签只用于离线选 head；最终推理时不读取 suc/fail 标签。

### 第三步：显式确定正负方向

首先对候选 heads 实测 `+6`：

- success/fail 两侧 correct-margin 与 paired spatial effect 均安全的 heads，保留正方向；
- 若一个 head 在 `+6` 下对 success/fail 两侧都产生负 correct-margin，则只把它列为负方向候选；
- 对负方向候选重新完整运行 `-6` 的 success、fail、target 和 wrong-region profiles；
- 只有 `-6` 实测通过双侧与空间 gate 后，才保留为 negative head。

严禁仅根据 `+6` 的数值做代数反号推断。最终每个 `(layer, head)` 只保留一个实际验证过的方向。

### 第四步：双侧因果强度加权

安全 head 的权重由较弱的一侧决定，避免 success-specialist 或 fail-specialist 独占排序。开发强度定义为：

```text
strength = max(0, min(success_correct_effect, fail_correct_effect))
           + max(0, paired_spatial_effect)

abs(multiplier) = max(0.1, sqrt(strength / max_strength))
```

`multiplier` 的符号继承前一步实际验证的 `+6/-6` 方向。直观上，一个 head 即使对 success 很强，只要对 fail 很弱，其最终权重也不会很高。

### 第五步：安全 padding 与 top-k

如果双侧安全 heads 少于64，则从未进入 causal 候选的低 raw-mass heads 中补齐到64，并固定：

```text
padding multiplier = 0.1
```

同时为 low-rank control 保留互斥 tail。这样 top-k=8/32/64 是嵌套组合，但 k32/k64 新增的 padding 不会承受完整剂量，避免 head 数增加导致干预强度失控。

### 最终推理协议

推理时只加载冻结的 `(layer, head, direction, multiplier)` ranking：

- 不使用评测标签；
- 不根据当前样本预测端点选择 success/fail branch；
- 不使用 held-out 输出调 threshold、head、权重或 bias；
- 对 target region 使用 layout-specific、all-frame attention steering；
- `steering_query_scope=all`，最终强配置使用 `negative_scope=none`；
- 同时评价 top-k=8、32、64，并保留 wrong-region 与 low-rank controls。

最终两个强配置为：

| 模型/布局 | ranking artifact | base bias | held-out config |
| --- | --- | ---: | --- |
| RoboReward interleaved | `causal_bidirectional_weighted_rank_roboreward_interleaved/bidirectional_causal_weighted_ranking.json` | 8 | `attention_48_roboreward_interleaved_bidirectional_weighted_bias8_heldout.yaml` |
| Qwen `images→text` | `causal_bidirectional_weighted_rank_qwen_images_text/bidirectional_causal_weighted_ranking.json` | 6 | `attention_64_qwen_images_text_bidirectional_weighted_bias6_heldout.yaml` |

### 最终确认结果

| 模型与输入 | k | MAE baseline→target | suc accuracy | fail accuracy | 结论 |
| --- | ---: | --- | --- | --- | --- |
| RoboReward interleaved | 8 | 1.5968→1.3974 | 44.39→55.16% | 12.87→18.57% | 强确认 |
|  | 32 | 1.5968→1.4103 | 44.39→54.71% | 12.87→17.72% | 强确认 |
|  | 64 | 1.5968→1.3945 | 44.39→54.26% | 12.87→18.35% | 强确认 |
| Qwen `images→text` | 8 | 1.5222→1.4003 | 52.91→69.06% | 6.75→19.83% | 强确认 |
|  | 32 | 1.5222→1.3816 | 52.91→74.44% | 6.75→21.10% | 强确认 |
|  | 64 | 1.5222→1.3816 | 52.91→73.99% | 6.75→20.89% | 强确认 |

两个强配置各自的九个主检验均通过 cluster-aware bootstrap 与 Holm 校正；四次正式 held-out 都是697 examples×10 conditions=`6970/6970` 有效记录。Qwen native `video→text` 的三个 k 点估计也均满足 MAE↓、suc↑、fail↑，但 k8 MAE cluster CI 跨0，因此只记为 `ANALYZED / CAUTION`，不计入强确认。

严格主线状态：RoboReward 与 Qwen 各只有1个强 held-out-confirmed layout，尚未达到“每个模型至少2种输入布局”的完整目标。不能通过继续消费当前 held-out、移动阈值或放宽统计门槛来补足计数。

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

当前 partition 与 held-out 已完成最终统计审计，不应继续用于调参。若开启下一阶段，应先冻结新的 video-cluster cohort，再研究更大的独立 development set、带不确定性校准的 head 组合/路由，或 attention-probability budget redistribution；这些属于新研究阶段，而不是本轮结果的追加调参。
