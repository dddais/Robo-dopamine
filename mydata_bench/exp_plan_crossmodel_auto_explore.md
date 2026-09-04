# 跨模型研究与效果分析

遵守本文的的要求，进行auto research，针对现有研究背景与问题进行探索性研究。

进行长时间的充分的调研、思考、理论分析、实验，直到完成所有主线目标。

我要睡觉了，需要我确认的部分先跳过，进行你能进行的内容。

## 研究背景：

- 基于/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan.md的规划，进行了baseline 和attention steering的实验，目前发现在robo-dopamine的GRM上该方法的效果十分明显;
- 但是在qwen3-vl-8b和roboreward-8b的效果不是很明显
- 已有实验可供参考：
  - 目前已进行的跨模型实验：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel.md
  - 该实验结果：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel_summary.md

## 主线目标1

- 调研相关文章，调研总结当前有哪些相关方法 ，目前已有部分可参考，见本文档“可参考代码/文献”部分，调研结果更新在“可参考代码/文献”部分。

## 主线目标2

- step1:基于调研的结果和方法，从理论角度思考提出能够改进目前attention steering的方案，如果需要可继续调研相关工作文章
- step2:实现step1提出的改进方案，进行实验验证，分析实验结果，如果效果不好则重复step1提出改进方案
- 验收目标：不断重复上述两个step，直到能够成功改良原 attention steering方法:两种模型（roboreward-8b 和qwen3-vl-8b ）各自在五种输入构造中的至少两种输入下都能稳定提升在数据集上的表现（MAE下降，suc,fail准确率提高）(不要求所有top-k都能满足，至少有一个不小的范围可以吧)，像GRM的表现那样。
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
- **PASTA — Post-hoc Attention Steering for LLMs**（ICLR 2024）：[arXiv:2311.02262](https://arxiv.org/abs/2311.02262)。选择小部分 task-specific/task-agnostic heads，并以 `log(alpha)` 修改 attention mask，相当于对非强调 token 做乘法降权；说明 soft multiplicative reweighting 比固定饱和 bias 更适合保留原能力。
- **PAI — Paying More Attention to Image**（ECCV 2024）：[arXiv:2407.21771](https://arxiv.org/abs/2407.21771)。只在最后 query 放大 image-token attention，并使用 image-free branch 做 logits 对比以抵消 text inertia；启发本任务比较 query scope，并区分“视觉证据增强”与“全局输出校准”。
- **ASCD — Attention-Steerable Contrastive Decoding**（AAAI 2026）：[arXiv:2506.14766](https://arxiv.org/abs/2506.14766)，DOI 10.1609/aaai.v40i12.38000。用 500 张参考图统计 `textAttn/visAttn`，每样本 top-32 投票得到 model-specific text-centric heads；正向分支只增强这些 heads，负向分支只抑制动态选出的关键视觉 token。其核心证据是选择性 head/token 操作优于随机或全量操作。
- **CAST — Caption-Guided Visual Attention Steering**：[arXiv:2605.04641](https://arxiv.org/abs/2605.04641)。以 caption-query vs non-caption-query 的 attention-output 差异训练 probe 选 head，并对选中 head 加入对比均值 steering direction；启发用“对比敏感度”而非 raw target mass 作为 ranking 目标。
- **Gaze Heads: How VLMs Look at What They Describe**：[arXiv:2606.14703](https://arxiv.org/abs/2606.14703)。在独立 comics 语料上发现少量 model-specific gaze heads；top-100 干预有效而 all-head 干预破坏生成。工作区已有 RoboReward/Qwen 各自在 500 个独立 comics 样本上得到的完整 ranking，可作为无评测标签泄漏的外部候选。
- **HAS — Highlight-guided Attention Steering for Multimodal LLM Video Summarization**：[arXiv:2607.17994](https://arxiv.org/abs/2607.17994)。把连续帧重要性平滑到 `[0,1]`，再以 `log(h+epsilon)` 注入 attention logits，等价于温和乘法重标定；支持使用连续时序 prior，避免 hard frame selection。
- **Arbitration Failure, Not Perceptual Blindness**：[arXiv:2604.09364](https://arxiv.org/abs/2604.09364)。报告 VLM 失败时视觉证据通常仍被编码，瓶颈更接近末层 arbitration；last-token patching 仅改变 0–1% 输出，而 full-sequence patching 改变 60–84%。这说明只增加注意力质量未必提高准确率，必须直接验证输出与配对区分度。
- **Inference-Time Attention Steering for VLA Driving Models**（ECCV 2026）：[arXiv:2608.17095](https://arxiv.org/abs/2608.17095)。在 Qwen3-VL 上发现 bias 强度和 late-layer 数量都有单调剂量效应，并强调逐调用 exposure audit；支持本研究进行 K-aware 剂量归一化和 hook 生效审计。
- **Attention is Case-Sensitive**（ECCV 2026）：[arXiv:2608.03711](https://arxiv.org/abs/2608.03711)。注意力集中并不保证准确率提高，强 salience 甚至会降性能；作为本研究不能用 attention-mass 增加替代任务指标的反例证据。
- Localization heads :Your Large Vision-Language Model Only Needs AFew Attention Heads For Visual Grounding; [https://arxiv.org/pdf/2503.06287](https://arxiv.org/pdf/2503.06287)
- Your Model Already Knows: Attention-Guided Safety Filter for Vision-Language-Action Models : [https://arxiv.org/pdf/2606.09749](https://arxiv.org/pdf/2606.09749)
- Analyzing Multi-Head Self-Attention :[https://arxiv.org/pdf/1905.09418](https://arxiv.org/pdf/1905.09418)



## 原因分析

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

## 最终有效方案

请你进行研究后，在这写明满足主线目标的最终方案