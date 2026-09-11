# Robometer-4B 与 SOLE-R1-8B：新增 baseline 与 attention steering 详细报告

> 快速阅读见 [实验总结](exp_plan_addbase_summary.md)。本文保留原完整总结正文，包含全部结果、方法和审计说明。


## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-09-10
- Verification Status: ANALYZED
- Version Label: exp_plan_addbase_summary_v1
- 工作目录：`/mnt/public1/dais/workspace/Robo-Dopamine`，与计划中的 `/home/dais/workspace/Robo-Dopamine` 指向同一仓库。
- 范围：执行根目录 [exp_plan_addbase.md](../exp_plan_addbase.md) 的全部主线要求。矩阵完整性和逐步证据已审计；未做整矩阵第二次推理复现。

## 1. 结论与完成范围

已完成两模型各五种输入顺序和额外官方输入，共 **12 个配置、228 个 baseline/干预条件**。每个配置包含全量 baseline，以及两个时域 × 三个 k × target/wrong-region/low-rank 的 18 个干预。按配置、条件、example_id 取追加日志最新记录，共 **197,292 条终点尝试记录**；该数量包含明确报告的无效输出，不能称为全部有效预测。

其中有效输出 **196,797 条**，答案格式失败 **315 条**，wrong-region 不可构造 **180 条**，其它未分类无效记录为 0。SOLE 官方输入的全部 16,441 条终点均有效。

| 模型 | 样本集 | 完整 target 比较 | MAE 降/平/升（完整比较） | 下降且 Holm p<.05 | 上升且 Holm p<.05 | target MAE 小于两控制（共同有效样本） | MAE 下降且 suc/fail 都严格提高：严格/宽阈值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| meter | cohort | 36/36 | 36/0/0 | 34 | 0 | 36 | 0/0 |
| meter | holdout | 36/36 | 35/1/0 | 34 | 0 | 35 | 0/1 |
| sole | cohort | 22/36 | 11/0/11 | 0 | 1 | 12 | 8/9 |
| sole | holdout | 22/36 | 11/0/11 | 0 | 0 | 12 | 7/10 |

**Robometer 的 MAE 收益较一致，但端点准确率存在类别权衡。** cohort 的 36 个 target 都降低 MAE，也都在四条件共同有效样本上优于两控制；然而两套阈值均没有一个条件同时严格提高 suc 与 fail 准确率。因此本轮支持特定离线误差和区分度收益，不能写成成功、失败判断全面改善。

**SOLE 的效果依赖输入协议、干预范围与 k。** 完整矩阵同时包含改善、退化和格式失败；官方输入与普通单次输入应分别解释。上表对不完整比较保留在 36 项总数中，下文完整列出六个 target，避免只报告最低 MAE 点。

SOLE 官方输入的 cohort baseline MAE 为 1.3121；last-frame k8/32/64 为 1.3132/1.2199/1.2317，all-frames k8/32/64 为 1.3369/1.3972/1.3298。因此官方输入下只有 last-frame k32、k64 降低 MAE，全帧三种 k 均上升；“更多帧受到干预”没有带来一致收益。每个条件的统计不确定性与控制见后表。

SOLE 的 22 个完整比较中，MAE 下降与上升各 11 个；没有 MAE 改善通过 Holm 校正。官方六个 target 的 cohort 95% CI 均包含 0、Holm p 均为 1；例如 last-frame/k32 的 ΔMAE=−0.0922，95% CI [−0.2019, 0.0164]，未校正 p=0.1014。因此这些官方改善点仍是描述性候选，不能写成已证实的稳定收益，也不能用未显著来证明没有效应。

另有一个明确的 cohort 负结果：SOLE interleaved/last_frame/k64 的 ΔMAE=+0.1501，95% CI [0.0674, 0.2303]，Holm p=0.0190。留出同样呈上升方向（+0.1342），但 Holm p=0.1482，未通过校正；两样本集有重叠，不能视为两次独立检验。

这组结果没有确立跨模型通用的最优输入顺序、k 或时域。历史 GRM 的较大收益与本轮新增结果共同表明需要区分“输出分数改变”“指令区分改善”和“双类端点判断改善”。模型训练、读出、分辨率和官方递推均不同，本轮不能单独识别是哪一因素造成模型间收益差异。

| 计划步骤 | 交付 |
| --- | --- |
| step1 新增入口 | [meter_eval](meter_eval/__main__.py)、[top_eval](top_eval/__main__.py)、[共享实现](addbase_eval/README.md)；top_eval 的模型是 SOLE-R1-8B |
| step2 check.md 核对 | 第 3 节实现核对、13 个契约检查、官方 readout 局部 parity、全矩阵证据审计 |
| step3 全部实验 | [12 个配置](configs/v2_crossmodel_addbase)、[完整数值表](../results/mydata_bench/experiments_v2_addbase/analysis_v1/full_tables.md) |
| step4 五类指标与结论 | 本文；每配置 exp_record、task CSV、配对分档、具体 top-8 与 top-8/32/64 重合度 |

## 2. 样本、指标与统计口径

| 样本集 | 总量 | suc / fail | task 子集 | 用途 |
| --- | --- | --- | --- | --- |
| full | 1213 | 407 / 806 | 34 | 每个 baseline 的完整样本 |
| cohort | 846 | 268 / 578 | 28 | 用户认可的 auto_grounded_v2；所有干预与同组 baseline 比较 |
| holdout | 730 | 234 / 496 | 28 | 排除原 ranking 清单全部 36 个视频组；与 cohort 重叠 |

实际有效 ranking 为 34 条；留出排除使用全部 36 个候选视频组，包含未进入有效 ranking 的两组。full 中的 task3_5、task3_9、task4_4、task4_5、task5_5、task5_9 完全不在 cohort；attention 结果覆盖的是 28 个 task 子集。留出仍来自相同任务域，并非外部任务集，也不是一次独立复现。

固定标签为 suc=5、fail=1。主 MAE 使用五档奖励 `r(p)=1+Σ[p≥t]`，`t∈{.125,.375,.625,.875}`，计算 `mean(|r(p)−label|)`。它衡量这批指令一致性标签上的误差，不是有逐帧真值的物理完成百分比回归。另存连续映射 `1+4·clip(p,0,1)` 的 MAE，但不替换主指标。

端点准确率独立采用两套 `(low,high)=(.125,.875)、(.2,.8)`：fail 仅在 `p≤low` 时正确，suc 仅在 `p≥high` 时正确，中间值均不正确。本文准确率以固定期望样本数为分母，无效输出不计正确；分析 JSON 同时保存有效分母的比率。MAE 表示有效输出的描述性均值，并另给无效项误差在 [0,4] 下的全体上下界。

最终 `metrics.csv` 保留旧 `acc_*`（有效分母），并新增含义明确的 `acc_valid_*`、`acc_fixed_*`、`correct_*`、`expected_*` 列；`index.json` 记录列口径。本文和完整 Markdown 表使用 `acc_fixed_*` 对应口径，不能把旧 CSV 的 `acc_*` 当成固定分母准确率。

预测分布同时保存奖励档与等宽进度区间（边界 `.2/.4/.6/.8`）；不能互换，更不能把某分布最低档数量直接当作两套端点准确率的正确数。SOLE 支持负百分比，归一化保留负进度，并按既定解析规则将越界值裁剪到 [−1,1]；负值落最低档。连续配对差使用这一归一化、范围裁剪后的进度，仍可超过 1；未裁剪的原百分比另存，不能与该差值混称。

同视频配对按 `source_suc_id` 连接并核验 SHA256。完整 cohort 最多 543 对，另有 35 个 fail 的来源 suc 未进入 cohort；543 对并非 543 个独立视频。报告连续 suc−fail、每 10% 差值分布及离散差 `<0/0/1/2/3/4`，并在相同有效配对上比较变化、补充视频等权均值。

ΔMAE=干预−baseline，负值表示误差降低。采用视频组 bootstrap 5000 次给出点对点 95% CI，视频组 sign flip 10000 次，seed=20260909；检验依赖视频组可交换性/对称性假设。cohort 和 holdout 各固定 72 个 target 比较做 Holm，不完整比较以 p=1 保留在 family 中。控制、阈值、task 和最佳 k 的描述不额外声称经过这一校正；CI 也不是同时置信区间。描述性补充在部分结果产生后编写，本研究不声称正式预注册。

| 恒定输出 p=0：始终失败 | MAE | 准确率 all / suc / fail |
| --- | --- | --- |
| full | 1.3421 | 66.45% / 0.00% / 100.00% |
| cohort | 1.2671 | 68.32% / 0.00% / 100.00% |
| holdout | 1.2822 | 67.95% / 0.00% / 100.00% |

cohort 中 fail 占 68.32%，始终判失败也能取得该总准确率，但 suc 准确率为 0，balanced accuracy 为 .5。这是解释模型总准确率所需的类别基率参考。

## 3. 官方输入与 check.md 实现核对

Robometer 使用 checkpoint 训练过的 prog_token 和 progress MLP；10-bin logits 经 softmax 后对 [0,1] 等距 bin center 求期望。738 个权重张量严格加载，success probability 单独保存。官方输入为 instruction 后接 image/prog_token 交错序列，使用原生图像尺寸；普通 image_text/video_text 把最终 prog_token 放在指令之后，保证最终读出能读取指令。

SOLE 官方输入直接取 RewardGen 原文 system/user prompt 与首/前/当前帧拼图函数，八个采样时刻对应七次模型预测，首时刻为 0。每个干预条件用自身上一时刻绝对进度递推；**绝对进度不累加**。全部 SOLE 使用 greedy、512 新 token 上限；官方 RewardGen 的temperature=1、top_p=.9、top_k=50、max_tokens=200 与本轮不同，因此这里是官方输入构造，不是官方随机采样结果或论文全部实验的逐参数复现。

SOLE official 的 ranking 在第七步未干预 prompt 上计算，前驱取 baseline 自身第六步预测；34 个发现样本均有有效前驱。得到的固定 head 集合用于各步干预，不在每一步重排，也不使用标签作为进度上下文。

五种普通输入固定八个采样时刻、正面单视角、图像 max_pixels=50176；interleaved 在图像间插入 Robometer prog_token 或 SOLE 帧次文字。native video 将八源帧编码为四个 tubelet。官方 Robometer 保留 640×480 原生大小，SOLE 保留官方拼图上下文，故 official 对普通输入的差异同时涉及分辨率或时序上下文。本轮不改变旧 GRM 三视角填充实现；SOLE 选官方 external-only 分支。

| check.md 检查项 | 实现与验证口径 |
| --- | --- |
| all-query 广播 | B×H×1×K 的 key bias 广播至所有 prefill/decode query，保留原 causal/padding mask；生成文字 key bias 为 0 |
| 视觉 key 对齐 | 逐项核对真实 token span、grid_thw、源帧、tracking 帧与 bbox；native video 使用两源帧 bbox 并集 |
| ranking | 未干预 mean raw mass，排除零基 L0–L7；Robometer 最终 prog_token，SOLE 最后 prompt token |
| wrong-region | 在相同时域按网格距离选远处非目标单元，等 token 数且不相交；不足时仅该控制不可用 |
| low-rank | 同一排名最后 k 个合格 head，仍排除前八层；未逐层匹配，不能视作完全相同的层分布 |
| ±bias 时域 | +6 目标 key；−6 同选定视觉域的非目标 key。last 为末图/末 tubelet；all 为全部输入图域 |
| SOLE 官方 last/all | 每一步 last 为与当前帧内容相交的 key 域，all 为该步拼图三时刻；包含自身前驱反馈 |
| 输入与递推 | 逐样本保存完整 prompt、token 哈希、各步原文、progress_curve；标签文件与模型输入分开 |
| 结果对齐 | 按 example_id 与 condition 审计，无缺失/额外 ID；同视频配对另核验哈希 |

**几何限制：矩形相交不是像素级隔离。** 确定性选取第一条 ranking 样本的网格核查中，SOLE official 当前域有 130 单元，目标 6 单元；34 个域单元不完全位于当前内容内部，其中 10 个也与前一帧内容相交。因此早期协议中“黑边/未选时刻不受负 bias”的绝对措辞过强，应以 [几何审计与可视化](../results/mydata_bench/experiments_v2_addbase/research/geometry_audit_v1.md) 为准。本次没有据此改 mask 或重算预测。视觉编码器此前的上下文混合也意味着目标位置 token 不等于纯目标语义。

## 4. 完整 baseline：两套阈值均报告

| 配置 | 有效/1213 | MAE | 全体 MAE 界 | 严格 all / suc / fail | 宽 all / suc / fail |
| --- | --- | --- | --- | --- | --- |
| meter_video_text | 1213 | 2.0000 | 2.0000–2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| meter_text_video | 1213 | 1.6620 | 1.6620–1.6620 | 2.14% / 0.74% / 2.85% | 16.98% / 24.08% / 13.40% |
| meter_image_text | 1213 | 2.1739 | 2.1739–2.1739 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| meter_text_image | 1213 | 2.0190 | 2.0190–2.0190 | 26.05% / 72.97% / 2.36% | 33.31% / 88.21% / 5.58% |
| meter_interleaved | 1213 | 2.1096 | 2.1096–2.1096 | 30.09% / 88.94% / 0.37% | 34.38% / 93.86% / 4.34% |
| meter_official | 1213 | 2.0058 | 2.0058–2.0058 | 32.07% / 95.09% / 0.25% | 36.93% / 98.53% / 5.83% |
| sole_video_text | 1203 | 1.8454 | 1.8302–1.8631 | 8.49% / 1.97% / 11.79% | 12.70% / 6.14% / 16.00% |
| sole_text_video | 1188 | 1.9714 | 1.9308–2.0132 | 6.92% / 0.74% / 10.05% | 12.45% / 3.93% / 16.75% |
| sole_image_text | 1213 | 1.8079 | 1.8079–1.8079 | 8.16% / 2.46% / 11.04% | 11.21% / 5.65% / 14.02% |
| sole_text_image | 1213 | 1.9959 | 1.9959–1.9959 | 5.69% / 0.98% / 8.06% | 9.07% / 7.62% / 9.80% |
| sole_interleaved | 1213 | 1.8780 | 1.8780–1.8780 | 9.32% / 1.72% / 13.15% | 11.46% / 3.44% / 15.51% |
| sole_official | 1213 | 1.4295 | 1.4295–1.4295 | 42.46% / 23.34% / 52.11% | 54.16% / 26.78% / 67.99% |

Robometer 的非官方 image_text、video_text 明显退化：两套端点准确率均为 0。video_text 的全部 1213 条输出均落奖励档 3；image_text 为 496 条档 3、717 条档 4。text_video 虽有更低的 MAE，其严格 suc/fail 端点准确率仍很低。这说明顺序影响读出与校准，不能仅据 MAE 排名声称指令判断可靠。

SOLE 官方输入的全量 MAE 为 1.4295，严格 all/suc/fail 为 42.46%/23.34%/52.11%，宽阈值为 54.16%/26.78%/67.99%；在本轮六个 SOLE baseline 中，MAE 更低且两类端点准确率均更高。这支持保留模型特有的官方上下文进行评估；但官方拼图、分辨率、七步推理和自身前驱同时改变，不能把差异单独归因于输入顺序，也不等于在数据外普遍优于其它模型。

### Robometer 次要 success head

| 配置 | success head MAE | 严格 all / suc / fail | 宽 all / suc / fail |
| --- | --- | --- | --- |
| meter_video_text | 1.3397 | 63.56% / 0.00% / 95.66% | 66.45% / 0.00% / 100.00% |
| meter_text_video | 1.0923 | 59.69% / 12.29% / 83.62% | 65.54% / 20.39% / 88.34% |
| meter_image_text | 1.5969 | 11.95% / 0.00% / 17.99% | 42.54% / 0.00% / 64.02% |
| meter_text_image | 1.5260 | 41.71% / 63.64% / 30.65% | 48.15% / 73.71% / 35.24% |
| meter_interleaved | 1.7502 | 38.66% / 76.41% / 19.60% | 44.11% / 84.28% / 23.82% |
| meter_official | 1.5664 | 47.57% / 83.54% / 29.40% | 52.60% / 88.70% / 34.37% |

success 与 progress 回答不同问题，故次要头可能呈现不同的端点准确率。这里将 success 概率沿用同一分档和两套阈值作辅助描述，并非官方 success 指标复现。全部干预的 success 统计另列各 exp_record 和分析 JSON，不据此事后替换主指标。Robometer 论文 E-5 的失败检测还组合 success 与进度曲线时间相关性，不能把本轮最终 progress 阈值结果当成论文 E-5 复现。

## 5. 全部 target：不隐去 k、时域或缺失比较

### cohort 的有效输出 MAE

| 配置 | 同组 baseline | last k8 | last k32 | last k64 | all k8 | all k32 | all k64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| meter_video_text | 2.0000 | 1.9953 | 1.9492 | 1.8487 | 1.9976 | 1.9137 | 1.9267 |
| meter_text_video | 1.5910 | 1.2577 | 1.3191 | 1.3593 | 1.2258 | 1.2825 | 1.2967 |
| meter_image_text | 2.2045 | 1.9043 | 1.8511 | 1.7695 | 1.9125 | 1.7128 | 1.7861 |
| meter_text_image | 1.8889 | 1.2589 | 1.1797 | 1.1950 | 0.9823 | 1.0875 | 1.0981 |
| meter_interleaved | 1.9976 | 1.3700 | 1.2506 | 1.2943 | 1.1064 | 1.1123 | 1.1797 |
| meter_official | 1.8132 | 1.3014 | 1.1418 | 1.1939 | 1.1513 | 1.1147 | 1.2009 |
| sole_video_text | 1.8578 | 1.7955† | 1.8544† | 1.8168† | 1.8505† | 1.8095† | 1.9672† |
| sole_text_video | 1.9470 | 1.8240† | 1.6599† | 1.6104† | 1.7983† | 1.7455† | 1.8780† |
| sole_image_text | 1.8073 | 1.7730 | 1.8948 | 1.8404 | 1.7045 | 1.6962 | 2.0047† |
| sole_text_image | 1.9858 | 2.0260 | 2.0461 | 1.9433 | 1.8570 | 1.6970† | 1.9019 |
| sole_interleaved | 1.8593 | 1.9161 | 1.8262 | 2.0095 | 1.8144 | 1.7849 | 1.9184 |
| sole_official | 1.3121 | 1.3132 | 1.2199 | 1.2317 | 1.3369 | 1.3972 | 1.3298 |

### holdout 的有效输出 MAE

| 配置 | 同组 baseline | last k8 | last k32 | last k64 | all k8 | all k32 | all k64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| meter_video_text | 2.0000 | 1.9945 | 1.9452 | 1.8616 | 2.0000 | 1.9164 | 1.9260 |
| meter_text_video | 1.6397 | 1.2767 | 1.3329 | 1.3726 | 1.2493 | 1.2932 | 1.3068 |
| meter_image_text | 2.1932 | 1.9041 | 1.8521 | 1.7616 | 1.9123 | 1.7205 | 1.7877 |
| meter_text_image | 1.9630 | 1.3137 | 1.2000 | 1.2274 | 1.0137 | 1.0986 | 1.1137 |
| meter_interleaved | 2.0616 | 1.4356 | 1.2863 | 1.3438 | 1.1411 | 1.1247 | 1.1849 |
| meter_official | 1.8849 | 1.3712 | 1.1959 | 1.2384 | 1.2014 | 1.1342 | 1.2247 |
| sole_video_text | 1.8668 | 1.7959† | 1.8409† | 1.8274† | 1.8611† | 1.8025† | 1.9551† |
| sole_text_video | 1.9426 | 1.8222† | 1.6667† | 1.5978† | 1.8261† | 1.7683† | 1.9057† |
| sole_image_text | 1.8151 | 1.7863 | 1.8973 | 1.8521 | 1.7000 | 1.6986 | 2.0137† |
| sole_text_image | 1.9986 | 2.0438 | 2.0644 | 1.9397 | 1.8795 | 1.7215† | 1.9082 |
| sole_interleaved | 1.8712 | 1.9233 | 1.8260 | 2.0055 | 1.8164 | 1.7890 | 1.9233 |
| sole_official | 1.3521 | 1.3658 | 1.2630 | 1.2795 | 1.3932 | 1.4575 | 1.3877 |

† 表示 baseline/target 配对覆盖不完整，该有效输出 MAE 不能与不同有效样本集直接作总体因果比较。所有配对 ΔMAE、95% CI、有效/期望数、固定分母双类准确率和 Holm p 见 [全矩阵数值表](../results/mydata_bench/experiments_v2_addbase/analysis_v1/full_tables.md)；缺失误差界和共同样本比较见对应 JSON。

### 两模型官方输入的全部六个 target

| 配置 | target | 有效/期望 | 配对 ΔMAE | 95% CI | Holm p | 严格 all / suc / fail | 宽 all / suc / fail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| meter_official | last_frame:target:8 | 846/846 | -0.5118 | [-0.5738, -0.4504] | 0.0072 | 26.83% / 84.33% / 0.17% | 48.94% / 87.69% / 30.97% |
| meter_official | last_frame:target:32 | 846/846 | -0.6714 | [-0.7441, -0.6002] | 0.0072 | 23.05% / 72.39% / 0.17% | 57.21% / 78.73% / 47.23% |
| meter_official | last_frame:target:64 | 846/846 | -0.6194 | [-0.6905, -0.5485] | 0.0072 | 17.49% / 55.22% / 0.00% | 42.91% / 74.63% / 28.20% |
| meter_official | all_frames:target:8 | 846/846 | -0.6619 | [-0.7319, -0.5924] | 0.0072 | 25.30% / 76.12% / 1.73% | 57.68% / 77.61% / 48.44% |
| meter_official | all_frames:target:32 | 846/846 | -0.6986 | [-0.7781, -0.6186] | 0.0072 | 24.59% / 63.81% / 6.40% | 55.91% / 68.28% / 50.17% |
| meter_official | all_frames:target:64 | 846/846 | -0.6123 | [-0.6927, -0.5335] | 0.0072 | 22.10% / 61.94% / 3.63% | 33.33% / 68.28% / 17.13% |
| sole_official | last_frame:target:8 | 846/846 | 0.0012 | [-0.1207, 0.1224] | 1.0000 | 47.64% / 32.84% / 54.50% | 57.68% / 34.33% / 68.51% |
| sole_official | last_frame:target:32 | 846/846 | -0.0922 | [-0.2019, 0.0164] | 1.0000 | 52.13% / 33.21% / 60.90% | 61.82% / 34.33% / 74.57% |
| sole_official | last_frame:target:64 | 846/846 | -0.0804 | [-0.1841, 0.0247] | 1.0000 | 50.00% / 33.96% / 57.44% | 57.68% / 35.07% / 68.17% |
| sole_official | all_frames:target:8 | 846/846 | 0.0248 | [-0.0846, 0.1334] | 1.0000 | 46.22% / 33.21% / 52.25% | 55.08% / 36.19% / 63.84% |
| sole_official | all_frames:target:32 | 846/846 | 0.0851 | [-0.0226, 0.1961] | 1.0000 | 45.27% / 29.48% / 52.60% | 54.49% / 32.09% / 64.88% |
| sole_official | all_frames:target:64 | 846/846 | 0.0177 | [-0.0972, 0.1305] | 1.0000 | 45.39% / 28.36% / 53.29% | 55.44% / 29.85% / 67.30% |

### 官方输入的区域/head 控制（四条件共同有效样本）

| 配置 | 范围/k | 共同 n | baseline | target | wrong-region | low-rank |
| --- | --- | --- | --- | --- | --- | --- |
| meter_official | last_frame/8 | 846 | 1.8132 | 1.3014 | 1.7754 | 1.8239 |
| meter_official | last_frame/32 | 846 | 1.8132 | 1.1418 | 1.6738 | 1.8499 |
| meter_official | last_frame/64 | 846 | 1.8132 | 1.1939 | 1.5390 | 1.8652 |
| meter_official | all_frames/8 | 846 | 1.8132 | 1.1513 | 1.5615 | 1.8203 |
| meter_official | all_frames/32 | 846 | 1.8132 | 1.1147 | 1.3558 | 1.7766 |
| meter_official | all_frames/64 | 846 | 1.8132 | 1.2009 | 1.3783 | 1.8050 |
| sole_official | last_frame/8 | 846 | 1.3121 | 1.3132 | 1.3097 | 1.3452 |
| sole_official | last_frame/32 | 846 | 1.3121 | 1.2199 | 1.3215 | 1.2813 |
| sole_official | last_frame/64 | 846 | 1.3121 | 1.2317 | 1.2967 | 1.2695 |
| sole_official | all_frames/8 | 846 | 1.3121 | 1.3369 | 1.3156 | 1.3050 |
| sole_official | all_frames/32 | 846 | 1.3121 | 1.3972 | 1.3239 | 1.2861 |
| sole_official | all_frames/64 | 846 | 1.3121 | 1.3298 | 1.2742 | 1.2979 |

控制只匹配 token 数/时域或低排名 head，没有完全匹配语义对象、形状或层分布。target 优于控制支持当前计算设置下的选择性，不能证明收益全部来自正确目标语义；控制本身改善也不应被隐去。全部普通输入控制见完整数值表。

例如 SOLE official/all_frames/k64 的 target 与 wrong-region 均有完整的同一 846 条有效输出：target MAE 为 1.3298，wrong-region 为 1.2742；baseline 为 1.3121。正确区域 target 并未优于该空间控制，这限制了将该设置解释为稳定增强正确目标证据的主张。

### 解释用例：误差、阈值与控制回答不同问题

| 配置 / target | 有效输出 MAE：baseline→target | 严格 all/suc/fail：baseline→target | 宽 all/suc/fail：baseline→target |
| --- | --- | --- | --- |
| meter_text_image / all_frames:target:8 | 1.8889 → 0.9823 | 28.61% / 83.58% / 3.11% → 38.06% / 67.16% / 24.57% | 34.28% / 92.54% / 7.27% → 70.09% / 71.64% / 69.38% |
| meter_official / all_frames:target:32 | 1.8132 → 1.1147 | 30.85% / 96.64% / 0.35% → 24.59% / 63.81% / 6.40% | 36.52% / 98.51% / 7.79% → 55.91% / 68.28% / 50.17% |
| sole_official / last_frame:target:32 | 1.3121 → 1.2199 | 44.44% / 30.60% / 50.87% → 52.13% / 33.21% / 60.90% | 56.86% / 35.07% / 66.96% → 61.82% / 34.33% / 74.57% |
| sole_official / last_frame:target:64 | 1.3121 → 1.2317 | 44.44% / 30.60% / 50.87% → 50.00% / 33.96% / 57.44% | 56.86% / 35.07% / 66.96% → 57.68% / 35.07% / 68.17% |
| sole_official / all_frames:target:8 | 1.3121 → 1.3369 | 44.44% / 30.60% / 50.87% → 46.22% / 33.21% / 52.25% | 56.86% / 35.07% / 66.96% → 55.08% / 36.19% / 63.84% |
| sole_text_video / last_frame:target:64 | 1.9470 → 1.6104 | 7.09% / 0.75% / 10.03% → 15.48% / 4.48% / 20.59% | 12.65% / 4.48% / 16.44% → 25.53% / 10.82% / 32.35% |
| sole_text_video / all_frames:target:32 | 1.9470 → 1.7455 | 7.09% / 0.75% / 10.03% → 18.91% / 0.75% / 27.34% | 12.65% / 4.48% / 16.44% → 27.07% / 1.87% / 38.75% |

这些用例用于解释完整矩阵，属于观察后选例，并非独立确认的最优配置。Robometer text_image/all_frames/k8 是本轮 36 个 Robometer target 中观察到的最低 cohort MAE：1.8889→0.9823；严格 suc 83.58%→67.16%，fail 3.11%→24.57%，宽阈值总准确率虽达 70.09%，suc 仍由 92.54% 降至 71.64%。Robometer official/all_frames/k32 则连严格总准确率也下降，说明降低 MAE 不保证改善端点决策。

两模型官方输入的全部六个条件亦画为 [双类端点准确率变化图](../results/mydata_bench/experiments_v2_addbase/analysis_v1/official_endpoint_tradeoffs_v2.png)（[SVG](../results/mydata_bench/experiments_v2_addbase/analysis_v1/official_endpoint_tradeoffs_v2.svg)）：横轴为 fail 准确率变化，纵轴为 suc 变化，只有右上象限表示两类都严格提高。该图是固定分母的描述性点估计，不表示准确率变化通过显著性检验。

SOLE official/last_frame/k32 的严格 suc/fail 同时提高，但宽阈值 suc 下降；k64 在宽阈值 suc 与 baseline 相等。official/all_frames/k8 的严格双类准确率都提高，MAE 却上升。因此不能把“低 MAE”和“更好的双类端点判断”互相替代，阈值选择也会改变判断。

SOLE 普通 text_video/last_frame/k64 的有效输出 MAE 为 1.9470→1.6104，全体 MAE 界为 [1.9102, 1.9858]→[1.5780, 1.6584]。target 上界仍低于 baseline 下界，故误差改善并非只能依靠删掉无效输出才能成立。但其严格 suc 准确率仅 4.48%，且 baseline/target 配对覆盖不完整；Holm p=1 是保守缺失规则，不能解释为证明没有效果。该协议的 last-frame target 在共同样本上优于两控制，all-frames/k32 的 wrong-region 反而比 target MAE 更低，选择性随时域改变。

### 全部协议的 task 异质性

| 配置 | 时域 | k8：task MAE 降/平/升（完整 task 数） | k32 | k64 |
| --- | --- | --- | --- | --- |
| meter_video_text | last_frame | 3/25/0 (28/28) | 17/8/3 (28/28) | 19/8/1 (28/28) |
| meter_video_text | all_frames | 1/27/0 (28/28) | 22/6/0 (28/28) | 17/11/0 (28/28) |
| meter_text_video | last_frame | 27/0/1 (28/28) | 24/3/1 (28/28) | 25/1/2 (28/28) |
| meter_text_video | all_frames | 26/1/1 (28/28) | 25/0/3 (28/28) | 25/1/2 (28/28) |
| meter_image_text | last_frame | 21/4/3 (28/28) | 23/3/2 (28/28) | 24/3/1 (28/28) |
| meter_image_text | all_frames | 21/4/3 (28/28) | 24/3/1 (28/28) | 21/4/3 (28/28) |
| meter_text_image | last_frame | 26/2/0 (28/28) | 27/0/1 (28/28) | 26/1/1 (28/28) |
| meter_text_image | all_frames | 27/0/1 (28/28) | 27/0/1 (28/28) | 27/0/1 (28/28) |
| meter_interleaved | last_frame | 27/1/0 (28/28) | 26/0/2 (28/28) | 27/0/1 (28/28) |
| meter_interleaved | all_frames | 27/1/0 (28/28) | 27/0/1 (28/28) | 27/0/1 (28/28) |
| meter_official | last_frame | 26/0/2 (28/28) | 26/0/2 (28/28) | 25/1/2 (28/28) |
| meter_official | all_frames | 27/0/1 (28/28) | 24/2/2 (28/28) | 23/3/2 (28/28) |
| sole_video_text | last_frame | 14/3/7 (24/28) | 13/1/12 (26/28) | 15/3/8 (26/28) |
| sole_video_text | all_frames | 13/0/8 (21/28) | 15/4/7 (26/28) | 11/1/9 (21/28) |
| sole_text_video | last_frame | 10/2/4 (16/28) | 11/3/1 (15/28) | 13/0/3 (16/28) |
| sole_text_video | all_frames | 9/2/3 (14/28) | 10/3/5 (18/28) | 9/3/5 (17/28) |
| sole_image_text | last_frame | 13/5/10 (28/28) | 9/3/16 (28/28) | 12/1/15 (28/28) |
| sole_image_text | all_frames | 17/3/8 (28/28) | 16/1/11 (28/28) | 9/2/16 (27/28) |
| sole_text_image | last_frame | 9/4/15 (28/28) | 11/3/14 (28/28) | 11/4/13 (28/28) |
| sole_text_image | all_frames | 19/1/8 (28/28) | 21/0/6 (27/28) | 17/1/10 (28/28) |
| sole_interleaved | last_frame | 12/1/15 (28/28) | 13/5/10 (28/28) | 7/3/18 (28/28) |
| sole_interleaved | all_frames | 15/2/11 (28/28) | 16/0/12 (28/28) | 16/2/10 (28/28) |
| sole_official | last_frame | 16/0/12 (28/28) | 18/1/9 (28/28) | 19/1/8 (28/28) |
| sole_official | all_frames | 12/1/15 (28/28) | 12/1/15 (28/28) | 16/1/11 (28/28) |

task 方向为描述性结果，既不代表每条样本改善，也不代表两个类别都改善。不完整 task 没有强行归入方向统计，其固定分母、无效数和准确率仍完整保留。逐 task 的 full/cohort/holdout 指标和两种分布分别见 [13,176 行 task 指标](../results/mydata_bench/experiments_v2_addbase/analysis_v1/task_metrics_all_populations.csv) 与 [79,056 行 task 分布](../results/mydata_bench/experiments_v2_addbase/analysis_v1/task_distributions_all_populations.csv)。

## 6. 预测分布与同视频指令区分

下表是 full baseline 的奖励档分布，顺序为 `[1,2,3,4,5]`；无效输出不被填入任何档。干预的全部 suc/fail/task 分布与原始连续差分档保存在分析 JSON 和 task CSV。

| 配置 | suc 档 1–5 计数 | fail 档 1–5 计数 |
| --- | --- | --- |
| meter_video_text | 0, 0, 407, 0, 0 | 0, 0, 806, 0, 0 |
| meter_text_video | 5, 81, 96, 222, 3 | 23, 382, 246, 155, 0 |
| meter_image_text | 0, 0, 154, 253, 0 | 0, 0, 342, 464, 0 |
| meter_text_image | 1, 5, 18, 86, 297 | 19, 113, 117, 267, 290 |
| meter_interleaved | 0, 6, 6, 33, 362 | 3, 122, 102, 146, 433 |
| meter_official | 0, 1, 2, 17, 387 | 2, 158, 117, 99, 430 |
| sole_video_text | 34, 110, 115, 135, 8 | 95, 281, 188, 216, 21 |
| sole_text_video | 37, 113, 156, 85, 3 | 81, 192, 315, 188, 18 |
| sole_image_text | 17, 118, 141, 121, 10 | 89, 287, 220, 199, 11 |
| sole_text_image | 22, 41, 211, 129, 4 | 65, 109, 353, 272, 7 |
| sole_interleaved | 36, 69, 218, 77, 7 | 106, 147, 409, 127, 17 |
| sole_official | 176, 94, 22, 20, 95 | 420, 242, 60, 14, 70 |

SOLE official 全量 baseline 仍将 176/407 条 suc 放入最低奖励档，而 Robometer official 将 430/806 条 fail 放入最高奖励档；两者的主要偏差不同。下面给出前述解释用例在完整 cohort 上的分布变化，全部条件的分布仍完整保存在分析文件。

| 解释用例（cohort） | 类别 | baseline 奖励档 1–5 | target 奖励档 1–5 |
| --- | --- | --- | --- |
| meter_text_image / all_frames:target:8 | suc | 0, 1, 6, 37, 224 | 27, 31, 10, 20, 180 |
| meter_text_image / all_frames:target:8 | fail | 18, 103, 101, 183, 173 | 142, 356, 30, 26, 24 |
| meter_official / all_frames:target:32 | suc | 0, 0, 2, 7, 259 | 18, 57, 6, 16, 171 |
| meter_official / all_frames:target:32 | fail | 2, 156, 115, 83, 222 | 37, 479, 20, 15, 27 |
| sole_official / last_frame:target:32 | suc | 108, 47, 14, 17, 82 | 108, 55, 9, 7, 89 |
| sole_official / last_frame:target:32 | fail | 294, 177, 52, 9, 46 | 352, 143, 29, 7, 47 |

| cohort baseline | 有效对 | 来源 suc 视频 | 平均连续差 | 连续负差率 | 离散差 <0,0,1,2,3,4 计数 |
| --- | --- | --- | --- | --- | --- |
| meter_video_text | 543 | 265 | 0.0045 | 34.99% | 0, 543, 0, 0, 0, 0 |
| meter_text_video | 543 | 265 | 0.2802 | 17.31% | 19, 183, 138, 186, 17, 0 |
| meter_image_text | 543 | 265 | 0.0039 | 36.46% | 7, 493, 43, 0, 0, 0 |
| meter_text_image | 543 | 265 | 0.2540 | 20.07% | 30, 172, 136, 97, 90, 18 |
| meter_interleaved | 543 | 265 | 0.2525 | 19.71% | 13, 240, 88, 89, 110, 3 |
| meter_official | 543 | 265 | 0.3312 | 16.02% | 10, 188, 77, 113, 153, 2 |
| sole_video_text | 539 | 263 | 0.0300 | 41.56% | 151, 203, 105, 62, 15, 3 |
| sole_text_video | 518 | 253 | -0.0004 | 46.91% | 161, 196, 104, 47, 9, 1 |
| sole_image_text | 543 | 265 | 0.0727 | 35.54% | 133, 168, 147, 81, 14, 0 |
| sole_text_image | 543 | 265 | 0.0327 | 44.20% | 154, 206, 122, 50, 11, 0 |
| sole_interleaved | 543 | 265 | 0.0381 | 45.12% | 131, 229, 104, 64, 15, 0 |
| sole_official | 543 | 265 | 0.2468 | 37.75% | 117, 166, 44, 48, 66, 102 |

### 官方 target 的共同有效配对变化

| 配置 | target | 共同对 | 平均连续差的变化：配对等权 | 视频等权 | target 连续负差率 | target 离散差六档计数 |
| --- | --- | --- | --- | --- | --- | --- |
| meter_official | last_frame:target:8 | 543 | 0.1600 | 0.1524 | 12.34% | 26, 89, 69, 30, 328, 1 |
| meter_official | last_frame:target:32 | 543 | 0.1802 | 0.2046 | 11.60% | 27, 90, 32, 42, 352, 0 |
| meter_official | last_frame:target:64 | 543 | 0.1356 | 0.1669 | 11.79% | 20, 100, 50, 101, 272, 0 |
| meter_official | all_frames:target:8 | 543 | 0.1838 | 0.2253 | 13.26% | 19, 105, 27, 51, 335, 6 |
| meter_official | all_frames:target:32 | 543 | 0.1389 | 0.1814 | 15.47% | 29, 136, 20, 25, 323, 10 |
| meter_official | all_frames:target:64 | 543 | 0.0841 | 0.1313 | 17.50% | 29, 126, 37, 72, 275, 4 |
| sole_official | last_frame:target:8 | 543 | 0.0037 | -0.0106 | 36.10% | 105, 176, 36, 39, 64, 123 |
| sole_official | last_frame:target:32 | 543 | 0.0235 | 0.0178 | 30.20% | 76, 207, 52, 27, 67, 114 |
| sole_official | last_frame:target:64 | 543 | 0.0320 | 0.0333 | 32.04% | 83, 181, 55, 33, 67, 124 |
| sole_official | all_frames:target:8 | 543 | 0.0024 | -0.0100 | 34.99% | 112, 175, 43, 32, 63, 118 |
| sole_official | all_frames:target:32 | 543 | -0.0516 | -0.0634 | 38.86% | 131, 166, 55, 33, 54, 104 |
| sole_official | all_frames:target:64 | 543 | -0.0192 | -0.0279 | 38.31% | 115, 176, 52, 30, 56, 114 |

配对等权会使拥有更多 fail 指令的视频权重更高，因此补充视频等权；二者都在相同有效配对 ID 上作差。幅度增加与负差率降低是不同指标，不能替代端点准确率。跨模型比较时，SOLE 的负进度使连续差的范围也不同。

Robometer text_image/all_frames/k8 的 543 对平均连续差从 0.2540 增至 0.5071，连续负差率从 20.07% 降至 13.81%，差≥.5 的比例从 25.05% 增至 66.85%。但 Robometer official/all_frames/k32 虽将平均连续差从 0.3312 提高到 0.4701，离散负差数却由 10 增至 29；量化后的平局/反序与连续均值也不能互相替代。SOLE official/last_frame/k32 的均值仅从 0.2468 增至 0.2704，连续负差率从 37.75% 降至 30.20%，仍有大量反序对。这些是配对分布描述，没有另作配对排序指标的显著性声明。

## 7. 具体 top-8 与 top-8/32/64 重合度

| 配置/时域 | 具体 top-8（零基 L/H） |
| --- | --- |
| meter_video_text/last_frame | L22H15, L21H11, L20H29, L21H16, L19H15, L19H21, L16H25, L22H3 |
| meter_video_text/all_frames | L21H11, L20H29, L22H15, L18H30, L12H18, L14H17, L19H15, L20H30 |
| meter_text_video/last_frame | L19H21, L19H10, L19H17, L19H23, L22H15, L21H16, L21H19, L22H26 |
| meter_text_video/all_frames | L19H21, L19H10, L19H17, L19H23, L22H15, L21H16, L18H30, L21H19 |
| meter_image_text/last_frame | L22H15, L22H2, L21H11, L21H16, L21H19, L19H21, L19H23, L19H10 |
| meter_image_text/all_frames | L22H15, L21H11, L19H21, L18H30, L21H16, L22H2, L21H19, L19H15 |
| meter_text_image/last_frame | L19H21, L19H10, L19H17, L19H23, L21H16, L22H15, L21H19, L22H26 |
| meter_text_image/all_frames | L19H21, L19H10, L19H17, L19H23, L21H16, L22H15, L19H31, L21H19 |
| meter_interleaved/last_frame | L19H21, L19H10, L19H17, L18H30, L19H31, L21H16, L17H3, L22H26 |
| meter_interleaved/all_frames | L19H21, L19H10, L18H30, L21H16, L19H31, L19H17, L17H3, L22H26 |
| meter_official/last_frame | L19H21, L19H10, L18H30, L19H17, L17H3, L21H16, L19H31, L10H14 |
| meter_official/all_frames | L19H21, L19H10, L18H30, L19H16, L19H17, L22H15, L21H16, L19H23 |
| sole_video_text/last_frame | L19H28, L29H13, L26H26, L31H9, L17H24, L10H12, L23H24, L31H15 |
| sole_video_text/all_frames | L19H28, L19H31, L31H15, L15H0, L26H26, L17H24, L31H9, L8H26 |
| sole_text_video/last_frame | L15H25, L10H13, L18H30, L21H16, L8H24, L21H27, L19H28, L8H22 |
| sole_text_video/all_frames | L19H28, L19H31, L18H30, L21H19, L21H16, L15H25, L16H15, L21H29 |
| sole_image_text/last_frame | L26H26, L15H1, L24H31, L22H11, L15H3, L19H23, L27H18, L16H15 |
| sole_image_text/all_frames | L19H28, L26H26, L19H31, L14H13, L16H15, L15H0, L18H30, L21H19 |
| sole_text_image/last_frame | L15H25, L17H3, L22H17, L19H17, L10H13, L8H24, L21H27, L17H29 |
| sole_text_image/all_frames | L19H28, L18H30, L19H31, L21H19, L26H26, L17H3, L16H15, L17H29 |
| sole_interleaved/last_frame | L17H29, L19H17, L17H3, L24H10, L10H12, L17H24, L19H23, L22H17 |
| sole_interleaved/all_frames | L19H28, L19H31, L18H30, L16H15, L17H24, L21H16, L19H23, L10H12 |
| sole_official/last_frame | L24H13, L30H8, L29H1, L22H2, L34H0, L16H15, L31H15, L27H16 |
| sole_official/all_frames | L24H13, L29H1, L30H8, L31H15, L34H0, L16H15, L24H28, L29H12 |

| Robometer 与 SOLE 同名协议/时域 | top-8 交集 | top-32 交集 | top-64 交集 |
| --- | --- | --- | --- |
| video_text/last_frame | 0/8 | 9/32 | 21/64 |
| video_text/all_frames | 0/8 | 9/32 | 24/64 |
| text_video/last_frame | 1/8 | 7/32 | 22/64 |
| text_video/all_frames | 3/8 | 6/32 | 22/64 |
| image_text/last_frame | 1/8 | 8/32 | 19/64 |
| image_text/all_frames | 2/8 | 7/32 | 24/64 |
| text_image/last_frame | 1/8 | 4/32 | 15/64 |
| text_image/all_frames | 2/8 | 8/32 | 22/64 |
| interleaved/last_frame | 2/8 | 7/32 | 21/64 |
| interleaved/all_frames | 3/8 | 8/32 | 18/64 |
| official/last_frame | 0/8 | 1/32 | 7/64 |
| official/all_frames | 0/8 | 0/32 | 6/64 |

完整导出有 54 个 ranking 集、2,988 条重合度记录，包含新增 24 集与历史 Qwen/RoboReward/GRM 及 GRM raw-mass 重排。GRM 历史发布顺序以 excess mass 优先，额外 `/reranked_raw` 仅从历史值只读重排，没有改写历史结果。具体历史 top-8、全部交集比例和 Jaccard 见 [ranking 汇总](../results/mydata_bench/experiments_v2_addbase/head_overlap_v1/ranking_overlap.md)、[overlap.csv](../results/mydata_bench/experiments_v2_addbase/head_overlap_v1/overlap.csv)。跨权重和协议的 L/H 编号重合仅作描述，不等价于同一功能 head，也未验证直接复用旧 head 的效果。

## 8. 完整性、无效输出与复现边界

| 配置：baseline+18 干预 | 终点尝试 | 有效 | 格式失败 | 不可构造控制 | 其它无效 | 负进度 | 越界裁剪 | 终点生成达 token cap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meter_video_text | 16441 | 16414 | 0 | 27 | 0 | 0 | 0 | 0 |
| meter_text_video | 16441 | 16414 | 0 | 27 | 0 | 0 | 0 | 0 |
| meter_image_text | 16441 | 16429 | 0 | 12 | 0 | 0 | 0 | 0 |
| meter_text_image | 16441 | 16429 | 0 | 12 | 0 | 0 | 0 | 0 |
| meter_interleaved | 16441 | 16429 | 0 | 12 | 0 | 0 | 0 | 0 |
| meter_official | 16441 | 16441 | 0 | 0 | 0 | 0 | 0 | 0 |
| sole_video_text | 16441 | 16344 | 70 | 27 | 0 | 113 | 1 | 2 |
| sole_text_video | 16441 | 16171 | 243 | 27 | 0 | 117 | 0 | 5 |
| sole_image_text | 16441 | 16428 | 1 | 12 | 0 | 58 | 0 | 1 |
| sole_text_image | 16441 | 16428 | 1 | 12 | 0 | 82 | 2 | 0 |
| sole_interleaved | 16441 | 16429 | 0 | 12 | 0 | 133 | 0 | 0 |
| sole_official | 16441 | 16441 | 0 | 0 | 0 | 1 | 12 | 0 |

审计 `complete=true`，无缺失/额外 ID 和未处理运行错误。SOLE official 共核对 **115,087 个实际步骤**，覆盖同条件前驱、progress_curve、末步 prompt 哈希及实际生成步骤的 attention hooks。无效条件仍保留终点记录；本轮 SOLE 官方终点全部有效且完成七步。token cap 计数仅描述终点实际生成，不能单凭达上限判断答案是否无效。

格式解析失败保留原文，不能从损坏标签中猜数，也不补为 0/reward=1；wrong-region 不可构造时只影响该控制。SOLE 合法数字若超出 [−100,100]% 则按既定解析规则裁剪，原百分比与标志仍保存；普通输入中发现的 3 条为 117%、133%、133%，不能把它们描述为模型天然满足数值范围。不完整比较的缺失 [0,4] 误差界及对所有缺失赋值仍降低 MAE 的描述性判定见 [完整解释 JSON](../results/mydata_bench/experiments_v2_addbase/interpretation_v1.json)。四条件共同有效比较避免分母错配，但仍只代表该有效子集。

最终有 15 条终点记录发生范围裁剪，其中 12 条来自 SOLE official。另对官方全部 115,087 步作数值检查：13 步发生裁剪（12 个终点、1 个中间步骤），越界原值为 101%、102%、105%、110% 或 171%；有 2 个负进度步骤，无生成达 token cap 的步骤。中间步骤裁剪后的值会进入该条件自身递推上下文；本轮没有重跑无裁剪版本。详见 [官方全步骤数值审计](../results/mydata_bench/experiments_v2_addbase/research/official_numeric_audit_v1.json)。

13 个契约检查通过；Robometer 两条样本的 progress/success 与官方纯源码 readout 最大误差均为 0，ranking 观测前后输出误差也为 0。这是局部 readout 验证，**不是整矩阵二次推理复现**。全部 baseline 的新增奖励档分布复算确认原 MAE/准确率/配对统计未变（浮点容差 1e-12），该检查同样只验证统计。

运行环境为 robo-dopamine，Python 3.10.20、PyTorch 2.8.0/CUDA build 12.8、Transformers 4.57.0；仅使用用户授权的 GPU 0、1、2。权重/processor 的 31 个文件指纹、实际 batch 与调度事件均保留。早期 meta buffer、不可行控制阻断和官方 ranking 前驱处理修正及本轮进程重启已留痕；raw 结果只追加，按每条件/ID 最新记录评分，pilot 独立保存。

操作审计：首次读取计划前误执行过一次只读 `git status --short`，未执行版本写入；读到禁令后停止已有仓库 Git 操作，随后仅使用计划允许的新官方开源仓库 clone。未参考其它已有本地代码仓库；原始数据与历史结果没有删除或覆盖。

## 9. 统计谬误检查：11/11

| 检查 | 结论/限制 |
| --- | --- |
| 1 Simpson 悖论 | 已比较总体、suc/fail 与逐 task；存在类别权衡和 task 异质性，不能把总体 MAE 下降称为各层都改善，也不把局部反向直接命名为严格 Simpson 悖论 |
| 2 生态谬误 | task 均值不推断每条样本；保留逐例记录和同视频配对 |
| 3 Berkson 选择偏差 | grounding cohort 为筛选子集；full baseline 与 cohort 分开，未据此外推全部 34 个 task |
| 4 Collider 偏差 | 无基于模型分数的事后筛选；grounding 和有效输出选择仍可能限制推广，共同有效比较不能消除此风险 |
| 5 基率忽视 | 报告 suc/fail 固定分母和两套阈值，给出始终失败参考；总准确率不等价于平衡判断 |
| 6 均值回归 | 未按极差预测选择 episode；所有 k 完整披露，观察后最低 MAE 不当作独立确认，holdout 与 cohort 重叠 |
| 7 生存者偏差 | 格式失败/不可行控制公开；固定分母准确率、MAE 误差界、完整性门槛与共同 ID 比较并行报告 |
| 8 多处寻找效应 | 每 population 固定 72 项 Holm，未完整项 p=1；没有为探索性控制/阈值/task 结果冒称同一校正的显著性 |
| 9 分析岔路 | 参数、输入、发现集冻结，修复留痕；后加分布和解释导出明确为描述性，无正式预注册声明 |
| 10 相关与因果 | 固定视频上的 attention 计算干预可改变该模型输出；不推出机器人闭环成功率或训练机制因果归因 |
| 11 反向因果 | head raw mass 排名与结果关联不证明训练形成机制；跨模型编号相交不作因果或功能身份解释 |

总体解释等级为 **CAUTION**：已完成数值和完整性审计，但探索性设置、同域筛选、缺失输出与协议混杂仍限制结论。没有用“未显著”证明无效，也没有把显著的微小 MAE 变化解释成实际可用性。

## 10. 机制解释与下一步边界

对某个可见 query，原始注意力质量为目标 mT、受负 bias 域 mN、其它 mO，则施加 b 后：`mT′=exp(b)mT/[exp(b)mT+exp(−b)mN+mO]`。b=6 将目标对负域的相对 odds 乘以 exp(12)≈162755，对其它可见域乘以 exp(6)≈403；causal 禁止 key 仍不可见。这说明相同数值 bias 并非跨模型等效剂量，目标面积、原 mass、层位置和最终读出均改变作用。

Robometer 通过专门训练的 prog_token 和分类头读进度；SOLE 在最后 prompt token 排名后还生成 reasoning/answer，官方输入又包含逐步自反馈。普通图像先于指令时，较早视觉 token 在因果语言层不能读取后续指令；这些差异提供了协议敏感性的合理解释，但本轮没有用独立消融识别其相对贡献。ViT 混合、tubelet 并集和拼图边界也限制“纯目标语义增强”的机制主张。

若继续独立研究，优先在新视频组/新任务上固定候选协议与 k，同时检查双类准确率、共同控制和配对；再单独消融 bias 强度、ranking query、层分布与官方递推，避免同时改变多个因素。这些是后续假设，未算作本轮已执行的实验；本轮计划要求的矩阵与五类指标已完成。

## 11. 证据与复查入口

| 材料 | 位置 |
| --- | --- |
| 完整性与终止状态 | [completion audit](../results/mydata_bench/experiments_v2_addbase/completion_audit_v1.json)；[scheduler completion](../results/mydata_bench/experiments_v2_addbase/research/scheduler_completion.json) |
| 完整数值与统计 | [full_tables.md](../results/mydata_bench/experiments_v2_addbase/analysis_v1/full_tables.md)；[analysis index](../results/mydata_bench/experiments_v2_addbase/analysis_v1/index.json)；[Holm](../results/mydata_bench/experiments_v2_addbase/analysis_v1/holm.json) |
| 全矩阵科学图 | [ΔMAE PNG](../results/mydata_bench/experiments_v2_addbase/analysis_v1/paired_mae_changes.png)；[可导出 SVG](../results/mydata_bench/experiments_v2_addbase/analysis_v1/paired_mae_changes.svg) |
| 逐 task 全样本集 | [metrics CSV](../results/mydata_bench/experiments_v2_addbase/analysis_v1/task_metrics_all_populations.csv)；[distributions CSV](../results/mydata_bench/experiments_v2_addbase/analysis_v1/task_distributions_all_populations.csv) |
| 契约与局部官方一致性 | [13 项检查](../results/mydata_bench/experiments_v2_addbase/research/contracts_v9.log)；[官方 readout parity](../results/mydata_bench/experiments_v2_addbase/research/official_readout_parity.json) |
| 统计定义与几何 | [分布复算审计](../results/mydata_bench/experiments_v2_addbase/research/distribution_definition_audit_v1.json)；[几何审计](../results/mydata_bench/experiments_v2_addbase/research/geometry_audit_v1.md) |
| 环境与来源 | [环境](../results/mydata_bench/experiments_v2_addbase/research/runtime_environment_v2.json)；[模型指纹](../results/mydata_bench/experiments_v2_addbase/research/model_fingerprints.json)；[输入/配置/官方源码指纹](../results/mydata_bench/experiments_v2_addbase/research/source_provenance_v1.json)；[文献和估计目标](../results/mydata_bench/experiments_v2_addbase/research/literature_and_estimands_v2.md) |

| 配置 | 实验记录 |
| --- | --- |
| meter_video_text | [exp_record.md](../results/mydata_bench/experiments_v2_addbase/meter_video_text/exp_record.md) |
| meter_text_video | [exp_record.md](../results/mydata_bench/experiments_v2_addbase/meter_text_video/exp_record.md) |
| meter_image_text | [exp_record.md](../results/mydata_bench/experiments_v2_addbase/meter_image_text/exp_record.md) |
| meter_text_image | [exp_record.md](../results/mydata_bench/experiments_v2_addbase/meter_text_image/exp_record.md) |
| meter_interleaved | [exp_record.md](../results/mydata_bench/experiments_v2_addbase/meter_interleaved/exp_record.md) |
| meter_official | [exp_record.md](../results/mydata_bench/experiments_v2_addbase/meter_official/exp_record.md) |
| sole_video_text | [exp_record.md](../results/mydata_bench/experiments_v2_addbase/sole_video_text/exp_record.md) |
| sole_text_video | [exp_record.md](../results/mydata_bench/experiments_v2_addbase/sole_text_video/exp_record.md) |
| sole_image_text | [exp_record.md](../results/mydata_bench/experiments_v2_addbase/sole_image_text/exp_record.md) |
| sole_text_image | [exp_record.md](../results/mydata_bench/experiments_v2_addbase/sole_text_image/exp_record.md) |
| sole_interleaved | [exp_record.md](../results/mydata_bench/experiments_v2_addbase/sole_interleaved/exp_record.md) |
| sole_official | [exp_record.md](../results/mydata_bench/experiments_v2_addbase/sole_official/exp_record.md) |

一手来源：[Robometer 论文](https://arxiv.org/html/2603.02115)、[官方 Robometer 实现](addbase_eval/references/robometer/robometer/models/rbm.py)、[SOLE-R1 论文](https://arxiv.org/html/2603.28730v2)、[官方 RewardGen SOLE 实现](addbase_eval/references/rewardgen/rewardgen/sole.py)。历史背景为 [GRM 总结](exp_plan_GRM_summary.md) 和 [跨模型总结](exp_plan_crossmodel_summary.md)；本轮没有重跑这些历史模型，也没有将不同协议的结果混称同一受控比较。

<a id="success-head-supplement"></a>

## 12. Robometer success head 补充（2026-09-11）

本节按新增要求补充 success head、二分类指标和全部逐任务结果。只读取原始预测，未重跑模型；原 progress 主分析及其 Holm 检验保持原口径。success head 的五档 MAE 和端点统计与原分析 JSON 的已有结果完全一致。

### 12.1 head 含义与评分口径

本轮 [双 head 读出](meter_eval/model.py) 从同一组 prog_token 隐状态同时得到 progress 和 success，取末帧作为终点。[官方 success 训练代码](addbase_eval/references/robometer/robometer/trainers/rbm_heads_trainer.py) 使用二分类交叉熵；[官方模型](addbase_eval/references/robometer/robometer/models/rbm.py) 也将相同帧表征送入两个 head。因此更贴合本任务的是 success 的训练目标和输出语义，并没有额外更换输入。对于本数据的“正确指令 suc / 同视频错误指令 fail”标签，终点成功概率比连续进度更直接，但不能据训练目标推断所有任务都表现更好。

- 为与原主表对照，success 概率沿用五档奖励、严格 .125/.875 和宽 .2/.8 阈值。这里的 success MAE 是映射后的五档误差，不是成功概率的校准误差。
- 另补充固定二分类分界：分数>.5 判 suc，否则判 fail；与 [官方工具的 success 分界](addbase_eval/references/robometer/robometer/evals/compile_results.py) 一致。两个 head 均采用相同分界作诊断，未利用本数据搜索阈值。progress>.5 本身并不等价于任务完成。
- 平衡准确率为 suc 召回率与 fail 召回率的平均；准确率采用固定期望分母，无效输出不计正确。各类别分母为零时记“—”，不补为零。
- Baseline full=1213（407/806，34 task）；cohort=846（268/578，28 task）；holdout=730（234/496）。所有 steering 与同 cohort、同 head baseline 比较。

### 12.2 二分类 baseline 对比（full）

| 输入 | head | 二分类总准确率 | suc 召回率 | fail 召回率 | 平衡准确率 |
| --- | --- | --- | --- | --- | --- |
| video → text | progress | 44.93% | 76.66% | 28.91% | 52.78% |
| video → text | success | 66.45% | 0.00% | 100.00% | 50.00% |
| text → video | progress | 67.68% | 66.58% | 68.24% | 67.41% |
| text → video | success | 73.70% | 31.20% | 95.16% | 63.18% |
| images → text | progress | 33.64% | 99.75% | 0.25% | 50.00% |
| images → text | success | 66.45% | 0.00% | 100.00% | 50.00% |
| text → images | progress | 48.56% | 97.05% | 24.07% | 60.56% |
| text → images | success | 61.91% | 86.98% | 49.26% | 68.12% |
| 交错输入 | progress | 47.73% | 98.03% | 22.33% | 60.18% |
| 交错输入 | success | 56.06% | 93.86% | 36.97% | 65.42% |
| 官方输入 | progress | 52.35% | 99.51% | 28.54% | 64.02% |
| 官方输入 | success | 60.59% | 96.31% | 42.56% | 69.44% |

官方输入的 success head 相比 progress head 将严格总准确率提高 15.50 个百分点，MAE 降低 0.4394；二分类总准确率提高 8.24 个百分点，平衡准确率提高 5.42 个百分点。text → video 的 success 总准确率更高，但平衡准确率从 67.41% 降至 63.18%；video → text 和 images → text 的 success 二分类结果均为始终失败，平衡准确率只有 50%。不能把 head 语义更贴合表述为每种协议均更均衡。六种 baseline 的两套端点阈值表保留在第 4 节。

### 12.3 success steering、控制与配对

| Robometer 输入 / 范围 | k | success MAE | 严格总准确率 | 严格 suc 准确率 | 严格 fail 准确率 |
| --- | --- | --- | --- | --- | --- |
| video → text / 全帧 | 32 | 1.2648 → 1.1879 | 64.42% → 65.96% | 0.00% → 0.00% | 94.29% → 96.54% |
| text → video / 最后一帧 | 8 | 0.9102 → 0.7340 | 62.17% → 71.75% | 18.66% → 17.54% | 82.35% → 96.89% |
| images → text / 最后一帧 | 32 | 1.5662 → 1.0165 | 8.75% → 54.26% | 0.00% → 0.00% | 12.80% → 79.41% |
| text → images / 全帧 | 8 | 1.2837 → 0.5378 | 50.59% → 83.22% | 76.49% → 66.42% | 38.58% → 91.00% |
| 交错输入 / 全帧 | 8 | 1.5378 → 0.5816 | 44.80% → 79.08% | 85.82% → 69.03% | 25.78% → 83.74% |
| 官方输入 / 最后一帧 | 32 | 1.2258 → 0.5201 | 56.03% → 80.97% | 89.93% → 70.15% | 40.31% → 85.99% |

各行是该协议在 6 个 target 中按 cohort success MAE 观察后选择的最低点；它们不一定与 progress head 最低点一致。全部 36 个 target 在 cohort 和 holdout 均为 32 个 MAE 下降、4 个上升，4 个上升都来自 text → video。cohort 中 33/36 个 target 在四条件共同有效样本上低于两控制，holdout 为 35/36。没有对本补充新增显著性检验，不能套用原 progress 的 72 项 Holm family。

cohort 严格阈值下，只有 text → video / 全帧 / k8 同时提高两类准确率；宽阈值下为 images → text / 全帧 / k64 与最后一帧 / k32；二分类 .5 下为 text → video / 全帧 / k8 与最后一帧 / k8。双类方向依赖评分规则，不能混称为一组稳定改善点。

| 输入 | 共同有效条件 | n | success MAE | 严格 all/suc/fail |
| --- | --- | --- | --- | --- |
| 官方输入 | baseline | 846 | 1.2258 | 56.03% / 89.93% / 40.31% |
| 官方输入 | last_frame:target:32 | 846 | 0.5201 | 80.97% / 70.15% / 85.99% |
| 官方输入 | last_frame:wrong_region:32 | 846 | 1.1655 | 56.26% / 83.58% / 43.60% |
| 官方输入 | last_frame:low_rank:32 | 846 | 1.2660 | 53.90% / 88.81% / 37.72% |
| text → images | baseline | 842 | 1.2898 | 50.36% / 76.14% / 38.58% |
| text → images | all_frames:target:8 | 842 | 0.5404 | 83.14% / 65.91% / 91.00% |
| text → images | all_frames:wrong_region:8 | 842 | 1.1580 | 49.88% / 21.59% / 62.80% |
| text → images | all_frames:low_rank:8 | 842 | 1.3112 | 50.00% / 77.27% / 37.54% |

官方最后一帧 k32 的 543 对、265 个来源 suc 视频中，平均连续 suc−fail 从 0.5392 增至 0.6960，负差率从 16.39% 降至 12.34%。cohort 二分类准确率从 70.21% 增至 87.59%，平衡准确率从 77.40% 增至 85.11%，但 suc 召回率从 97.01% 降至 78.36%。holdout 的 success MAE 为 0.5822，严格准确率 all/suc/fail=79.32%/67.95%/84.68%；holdout 与 cohort 重叠，未构成独立复现。

### 12.4 全部任务：各输入 baseline 与其 success MAE 最低 target

每组第一张表使用 full，比较相同 1213 条上的两个 head；第二张表使用 cohort，箭头为该输入同 cohort 的 success baseline → success target。task 编号沿用数据集 subset；n 后括号为 suc/fail 样本数。未进入 cohort 的 task 没有 steering 结果。

官方 baseline 的 success MAE 相比 progress 在 20/34 个 task 降低、11 个持平、3 个升高（task3_1、task4_1、task4_7）。官方最后一帧 k32 在 24/28 个 cohort task 降低 success MAE，4 个退化 task 为 task2_2、task2_3、task3_6、task3_7。下面保留全部任务，不按结果好坏筛选。

<details>
<summary>video → text：34 个 full task 的双 head baseline，以及 28 个 cohort task 的 all_frames:target:32</summary>

**Baseline（full）**

| task | n（suc/fail） | progress MAE | success MAE | progress 严格 all/suc/fail | success 严格 all/suc/fail |
| --- | --- | --- | --- | --- | --- |
| task1_1 | 128 (32/96) | 2.0000 | 0.9453 | 0.00% / 0.00% / 0.00% | 72.66% / 0.00% / 96.88% |
| task1_2 | 45 (15/30) | 2.0000 | 1.3333 | 0.00% / 0.00% / 0.00% | 66.67% / 0.00% / 100.00% |
| task1_3 | 96 (24/72) | 2.0000 | 1.0000 | 0.00% / 0.00% / 0.00% | 75.00% / 0.00% / 100.00% |
| task2_1 | 64 (16/48) | 2.0000 | 0.9844 | 0.00% / 0.00% / 0.00% | 57.81% / 0.00% / 77.08% |
| task2_2 | 36 (12/24) | 2.0000 | 1.3333 | 0.00% / 0.00% / 0.00% | 44.44% / 0.00% / 66.67% |
| task2_3 | 16 (8/8) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 37.50% / 0.00% / 75.00% |
| task2_4 | 96 (24/72) | 2.0000 | 1.0000 | 0.00% / 0.00% / 0.00% | 75.00% / 0.00% / 100.00% |
| task2_5 | 54 (18/36) | 2.0000 | 1.3333 | 0.00% / 0.00% / 0.00% | 66.67% / 0.00% / 100.00% |
| task3_1 | 27 (9/18) | 2.0000 | 1.5185 | 0.00% / 0.00% / 0.00% | 40.74% / 0.00% / 61.11% |
| task3_10 | 54 (18/36) | 2.0000 | 1.3333 | 0.00% / 0.00% / 0.00% | 66.67% / 0.00% / 100.00% |
| task3_2 | 64 (16/48) | 2.0000 | 0.9844 | 0.00% / 0.00% / 0.00% | 75.00% / 0.00% / 100.00% |
| task3_3 | 12 (4/8) | 2.0000 | 1.3333 | 0.00% / 0.00% / 0.00% | 66.67% / 0.00% / 100.00% |
| task3_4 | 36 (12/24) | 2.0000 | 1.3333 | 0.00% / 0.00% / 0.00% | 66.67% / 0.00% / 100.00% |
| task3_5 | 64 (16/48) | 2.0000 | 1.0000 | 0.00% / 0.00% / 0.00% | 75.00% / 0.00% / 100.00% |
| task3_6 | 12 (6/6) | 2.0000 | 2.0833 | 0.00% / 0.00% / 0.00% | 41.67% / 0.00% / 83.33% |
| task3_7 | 12 (6/6) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task3_8 | 45 (15/30) | 2.0000 | 1.3333 | 0.00% / 0.00% / 0.00% | 66.67% / 0.00% / 100.00% |
| task3_9 | 80 (20/60) | 2.0000 | 1.0000 | 0.00% / 0.00% / 0.00% | 75.00% / 0.00% / 100.00% |
| task4_1 | 16 (8/8) | 2.0000 | 2.0625 | 0.00% / 0.00% / 0.00% | 37.50% / 0.00% / 75.00% |
| task4_2 | 12 (6/6) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task4_3 | 12 (6/6) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task4_4 | 12 (6/6) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task4_5 | 16 (8/8) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task4_6 | 8 (4/4) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task4_7 | 12 (6/6) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task5_1 | 32 (16/16) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 46.88% / 0.00% / 93.75% |
| task5_2 | 32 (16/16) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task5_3 | 32 (16/16) | 2.0000 | 1.9688 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task5_4 | 16 (8/8) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task5_5 | 16 (8/8) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task5_6 | 12 (6/6) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task5_7 | 12 (6/6) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task5_8 | 16 (8/8) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task5_9 | 16 (8/8) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |

**Success steering（cohort）**

| task | n（suc/fail） | success MAE | 严格总准确率 | 严格 suc 准确率 | 严格 fail 准确率 |
| --- | --- | --- | --- | --- | --- |
| task1_1 | 127 (32/95) | 0.9528 → 0.8425 | 72.44% → 74.80% | 0.00% → 0.00% | 96.84% → 100.00% |
| task1_2 | 15 (4/11) | 1.0667 → 1.0667 | 73.33% → 66.67% | 0.00% → 0.00% | 100.00% → 90.91% |
| task1_3 | 95 (23/72) | 0.9684 → 0.8947 | 75.79% → 73.68% | 0.00% → 0.00% | 100.00% → 97.22% |
| task2_1 | 64 (16/48) | 0.9844 → 0.9062 | 57.81% → 70.31% | 0.00% → 0.00% | 77.08% → 93.75% |
| task2_2 | 36 (12/24) | 1.3333 → 1.3333 | 44.44% → 66.67% | 0.00% → 0.00% | 66.67% → 100.00% |
| task2_3 | 16 (8/8) | 2.0000 → 2.0000 | 37.50% → 31.25% | 0.00% → 0.00% | 75.00% → 62.50% |
| task2_4 | 96 (24/72) | 1.0000 → 0.9375 | 75.00% → 75.00% | 0.00% → 0.00% | 100.00% → 100.00% |
| task2_5 | 51 (17/34) | 1.3333 → 1.3529 | 66.67% → 60.78% | 0.00% → 0.00% | 100.00% → 91.18% |
| task3_1 | 24 (8/16) | 1.5417 → 1.3750 | 41.67% → 58.33% | 0.00% → 0.00% | 62.50% → 87.50% |
| task3_10 | 51 (17/34) | 1.3333 → 1.3333 | 66.67% → 66.67% | 0.00% → 0.00% | 100.00% → 100.00% |
| task3_2 | 16 (4/12) | 1.0000 → 1.0000 | 75.00% → 75.00% | 0.00% → 0.00% | 100.00% → 100.00% |
| task3_3 | 12 (4/8) | 1.3333 → 1.4167 | 66.67% → 50.00% | 0.00% → 0.00% | 100.00% → 75.00% |
| task3_4 | 30 (10/20) | 1.3333 → 1.3667 | 66.67% → 60.00% | 0.00% → 0.00% | 100.00% → 90.00% |
| task3_6 | 10 (5/5) | 2.0000 → 2.0000 | 50.00% → 30.00% | 0.00% → 0.00% | 100.00% → 60.00% |
| task3_7 | 10 (5/5) | 2.0000 → 2.0000 | 50.00% → 50.00% | 0.00% → 0.00% | 100.00% → 100.00% |
| task3_8 | 39 (13/26) | 1.3333 → 1.3333 | 66.67% → 66.67% | 0.00% → 0.00% | 100.00% → 100.00% |
| task4_1 | 16 (8/8) | 2.0625 → 1.6875 | 37.50% → 50.00% | 0.00% → 0.00% | 75.00% → 100.00% |
| task4_2 | 5 (2/3) | 1.6000 → 1.6000 | 60.00% → 60.00% | 0.00% → 0.00% | 100.00% → 100.00% |
| task4_3 | 12 (6/6) | 2.0000 → 2.0000 | 50.00% → 50.00% | 0.00% → 0.00% | 100.00% → 100.00% |
| task4_6 | 5 (1/4) | 0.8000 → 0.8000 | 80.00% → 80.00% | 0.00% → 0.00% | 100.00% → 100.00% |
| task4_7 | 11 (5/6) | 1.8182 → 1.6364 | 54.55% → 54.55% | 0.00% → 0.00% | 100.00% → 100.00% |
| task5_1 | 24 (8/16) | 1.3750 → 1.1667 | 62.50% → 66.67% | 0.00% → 0.00% | 93.75% → 100.00% |
| task5_2 | 15 (7/8) | 1.8667 → 1.5333 | 53.33% → 53.33% | 0.00% → 0.00% | 100.00% → 100.00% |
| task5_3 | 31 (15/16) | 1.9032 → 1.6129 | 51.61% → 51.61% | 0.00% → 0.00% | 100.00% → 100.00% |
| task5_4 | 8 (2/6) | 1.0000 → 1.0000 | 75.00% → 75.00% | 0.00% → 0.00% | 100.00% → 100.00% |
| task5_6 | 12 (6/6) | 2.0000 → 1.8333 | 50.00% → 50.00% | 0.00% → 0.00% | 100.00% → 100.00% |
| task5_7 | 11 (5/6) | 1.8182 → 1.5455 | 54.55% → 54.55% | 0.00% → 0.00% | 100.00% → 100.00% |
| task5_8 | 4 (1/3) | 1.0000 → 1.0000 | 75.00% → 75.00% | 0.00% → 0.00% | 100.00% → 100.00% |

</details>

<details>
<summary>text → video：34 个 full task 的双 head baseline，以及 28 个 cohort task 的 last_frame:target:8</summary>

**Baseline（full）**

| task | n（suc/fail） | progress MAE | success MAE | progress 严格 all/suc/fail | success 严格 all/suc/fail |
| --- | --- | --- | --- | --- | --- |
| task1_1 | 128 (32/96) | 1.2344 | 0.2812 | 6.25% / 0.00% / 8.33% | 82.03% / 56.25% / 90.62% |
| task1_2 | 45 (15/30) | 1.8889 | 1.3778 | 0.00% / 0.00% / 0.00% | 53.33% / 0.00% / 80.00% |
| task1_3 | 96 (24/72) | 1.1042 | 0.3438 | 4.17% / 0.00% / 5.56% | 81.25% / 29.17% / 98.61% |
| task2_1 | 64 (16/48) | 1.5312 | 0.2969 | 0.00% / 0.00% / 0.00% | 76.56% / 31.25% / 91.67% |
| task2_2 | 36 (12/24) | 1.5278 | 0.1667 | 0.00% / 0.00% / 0.00% | 83.33% / 58.33% / 95.83% |
| task2_3 | 16 (8/8) | 1.5000 | 1.0625 | 0.00% / 0.00% / 0.00% | 43.75% / 12.50% / 75.00% |
| task2_4 | 96 (24/72) | 1.3333 | 0.6250 | 2.08% / 4.17% / 1.39% | 77.08% / 20.83% / 95.83% |
| task2_5 | 54 (18/36) | 1.3519 | 0.3519 | 3.70% / 11.11% / 0.00% | 75.93% / 38.89% / 94.44% |
| task3_1 | 27 (9/18) | 2.3333 | 1.8889 | 0.00% / 0.00% / 0.00% | 7.41% / 0.00% / 11.11% |
| task3_10 | 54 (18/36) | 1.8519 | 1.3519 | 3.70% / 0.00% / 5.56% | 61.11% / 0.00% / 91.67% |
| task3_2 | 64 (16/48) | 1.7500 | 1.0156 | 0.00% / 0.00% / 0.00% | 71.88% / 0.00% / 95.83% |
| task3_3 | 12 (4/8) | 1.9167 | 1.4167 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 75.00% |
| task3_4 | 36 (12/24) | 1.9722 | 1.5000 | 0.00% / 0.00% / 0.00% | 47.22% / 0.00% / 70.83% |
| task3_5 | 64 (16/48) | 1.7344 | 1.1250 | 10.94% / 0.00% / 14.58% | 60.94% / 0.00% / 81.25% |
| task3_6 | 12 (6/6) | 2.0833 | 1.9167 | 0.00% / 0.00% / 0.00% | 33.33% / 0.00% / 66.67% |
| task3_7 | 12 (6/6) | 1.8333 | 1.9167 | 0.00% / 0.00% / 0.00% | 33.33% / 0.00% / 66.67% |
| task3_8 | 45 (15/30) | 1.8000 | 1.4000 | 0.00% / 0.00% / 0.00% | 60.00% / 0.00% / 90.00% |
| task3_9 | 80 (20/60) | 1.7500 | 1.0375 | 1.25% / 0.00% / 1.67% | 70.00% / 0.00% / 93.33% |
| task4_1 | 16 (8/8) | 1.8125 | 1.8750 | 0.00% / 0.00% / 0.00% | 37.50% / 0.00% / 75.00% |
| task4_2 | 12 (6/6) | 1.9167 | 2.0000 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task4_3 | 12 (6/6) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 16.67% / 0.00% / 33.33% |
| task4_4 | 12 (6/6) | 1.9167 | 2.0000 | 0.00% / 0.00% / 0.00% | 33.33% / 0.00% / 66.67% |
| task4_5 | 16 (8/8) | 2.1250 | 2.0000 | 0.00% / 0.00% / 0.00% | 37.50% / 0.00% / 75.00% |
| task4_6 | 8 (4/4) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 25.00% / 0.00% / 50.00% |
| task4_7 | 12 (6/6) | 2.2500 | 2.2500 | 0.00% / 0.00% / 0.00% | 8.33% / 0.00% / 16.67% |
| task5_1 | 32 (16/16) | 2.0312 | 2.0000 | 0.00% / 0.00% / 0.00% | 31.25% / 0.00% / 62.50% |
| task5_2 | 32 (16/16) | 1.9688 | 2.0000 | 0.00% / 0.00% / 0.00% | 46.88% / 0.00% / 93.75% |
| task5_3 | 32 (16/16) | 1.9688 | 2.0938 | 0.00% / 0.00% / 0.00% | 9.38% / 0.00% / 18.75% |
| task5_4 | 16 (8/8) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 31.25% / 0.00% / 62.50% |
| task5_5 | 16 (8/8) | 2.0625 | 2.0625 | 0.00% / 0.00% / 0.00% | 37.50% / 0.00% / 75.00% |
| task5_6 | 12 (6/6) | 1.9167 | 1.9167 | 0.00% / 0.00% / 0.00% | 8.33% / 0.00% / 16.67% |
| task5_7 | 12 (6/6) | 1.9167 | 2.1667 | 0.00% / 0.00% / 0.00% | 8.33% / 0.00% / 16.67% |
| task5_8 | 16 (8/8) | 2.0000 | 1.9375 | 0.00% / 0.00% / 0.00% | 50.00% / 0.00% / 100.00% |
| task5_9 | 16 (8/8) | 1.9375 | 2.0000 | 0.00% / 0.00% / 0.00% | 37.50% / 0.00% / 75.00% |

**Success steering（cohort）**

| task | n（suc/fail） | success MAE | 严格总准确率 | 严格 suc 准确率 | 严格 fail 准确率 |
| --- | --- | --- | --- | --- | --- |
| task1_1 | 127 (32/95) | 0.2835 → 0.1339 | 81.89% → 89.76% | 56.25% → 71.88% | 90.53% → 95.79% |
| task1_2 | 15 (4/11) | 1.3333 → 0.9333 | 46.67% → 73.33% | 0.00% → 0.00% | 63.64% → 100.00% |
| task1_3 | 95 (23/72) | 0.3263 → 0.2211 | 82.11% → 84.21% | 30.43% → 34.78% | 98.61% → 100.00% |
| task2_1 | 64 (16/48) | 0.2969 → 0.4375 | 76.56% → 76.56% | 31.25% → 6.25% | 91.67% → 100.00% |
| task2_2 | 36 (12/24) | 0.1667 → 0.5000 | 83.33% → 69.44% | 58.33% → 8.33% | 95.83% → 100.00% |
| task2_3 | 16 (8/8) | 1.0625 → 1.3125 | 43.75% → 43.75% | 12.50% → 25.00% | 75.00% → 62.50% |
| task2_4 | 96 (24/72) | 0.6250 → 0.5417 | 77.08% → 81.25% | 20.83% → 25.00% | 95.83% → 100.00% |
| task2_5 | 51 (17/34) | 0.3529 → 0.8431 | 76.47% → 70.59% | 41.18% → 17.65% | 94.12% → 97.06% |
| task3_1 | 24 (8/16) | 1.8750 → 1.3333 | 8.33% → 58.33% | 0.00% → 0.00% | 12.50% → 87.50% |
| task3_10 | 51 (17/34) | 1.3529 → 1.3922 | 60.78% → 54.90% | 0.00% → 0.00% | 91.18% → 82.35% |
| task3_2 | 16 (4/12) | 1.0000 → 1.0000 | 75.00% → 75.00% | 0.00% → 0.00% | 100.00% → 100.00% |
| task3_3 | 12 (4/8) | 1.4167 → 1.1667 | 50.00% → 58.33% | 0.00% → 0.00% | 75.00% → 87.50% |
| task3_4 | 30 (10/20) | 1.5333 → 1.3333 | 43.33% → 66.67% | 0.00% → 0.00% | 65.00% → 100.00% |
| task3_6 | 10 (5/5) | 1.9000 → 1.4000 | 40.00% → 50.00% | 0.00% → 0.00% | 80.00% → 100.00% |
| task3_7 | 10 (5/5) | 2.0000 → 1.9000 | 30.00% → 50.00% | 0.00% → 0.00% | 60.00% → 100.00% |
| task3_8 | 39 (13/26) | 1.4103 → 1.3333 | 58.97% → 66.67% | 0.00% → 0.00% | 88.46% → 100.00% |
| task4_1 | 16 (8/8) | 1.8750 → 1.3125 | 37.50% → 50.00% | 0.00% → 0.00% | 75.00% → 100.00% |
| task4_2 | 5 (2/3) | 1.6000 → 1.6000 | 60.00% → 60.00% | 0.00% → 0.00% | 100.00% → 100.00% |
| task4_3 | 12 (6/6) | 2.0000 → 1.1667 | 16.67% → 50.00% | 0.00% → 0.00% | 33.33% → 100.00% |
| task4_6 | 5 (1/4) | 1.0000 → 0.2000 | 40.00% → 80.00% | 0.00% → 0.00% | 50.00% → 100.00% |
| task4_7 | 11 (5/6) | 2.1818 → 1.3636 | 9.09% → 45.45% | 0.00% → 0.00% | 16.67% → 83.33% |
| task5_1 | 24 (8/16) | 1.3750 → 0.5417 | 41.67% → 70.83% | 0.00% → 12.50% | 62.50% → 100.00% |
| task5_2 | 15 (7/8) | 1.8000 → 1.8000 | 53.33% → 53.33% | 0.00% → 0.00% | 100.00% → 100.00% |
| task5_3 | 31 (15/16) | 2.1290 → 0.6774 | 9.68% → 58.06% | 0.00% → 13.33% | 18.75% → 100.00% |
| task5_4 | 8 (2/6) | 1.2500 → 1.0000 | 50.00% → 75.00% | 0.00% → 0.00% | 66.67% → 100.00% |
| task5_6 | 12 (6/6) | 1.9167 → 0.8333 | 8.33% → 50.00% | 0.00% → 0.00% | 16.67% → 100.00% |
| task5_7 | 11 (5/6) | 2.0000 → 0.6364 | 9.09% → 54.55% | 0.00% → 0.00% | 16.67% → 100.00% |
| task5_8 | 4 (1/3) | 1.0000 → 1.0000 | 75.00% → 75.00% | 0.00% → 0.00% | 100.00% → 100.00% |

</details>

<details>
<summary>images → text：34 个 full task 的双 head baseline，以及 28 个 cohort task 的 last_frame:target:32</summary>

**Baseline（full）**

| task | n（suc/fail） | progress MAE | success MAE | progress 严格 all/suc/fail | success 严格 all/suc/fail |
| --- | --- | --- | --- | --- | --- |
| task1_1 | 128 (32/96) | 2.4609 | 1.4688 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task1_2 | 45 (15/30) | 2.2222 | 1.6444 | 0.00% / 0.00% / 0.00% | 2.22% / 0.00% / 3.33% |
| task1_3 | 96 (24/72) | 2.1979 | 1.4375 | 0.00% / 0.00% / 0.00% | 5.21% / 0.00% / 6.94% |
| task2_1 | 64 (16/48) | 2.5000 | 1.4844 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task2_2 | 36 (12/24) | 2.3333 | 1.6667 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task2_3 | 16 (8/8) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task2_4 | 96 (24/72) | 2.1354 | 1.3125 | 0.00% / 0.00% / 0.00% | 22.92% / 0.00% / 30.56% |
| task2_5 | 54 (18/36) | 2.0000 | 1.5185 | 0.00% / 0.00% / 0.00% | 20.37% / 0.00% / 30.56% |
| task3_1 | 27 (9/18) | 2.3333 | 1.6667 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task3_10 | 54 (18/36) | 2.0370 | 1.5741 | 0.00% / 0.00% / 0.00% | 20.37% / 0.00% / 30.56% |
| task3_2 | 64 (16/48) | 2.4062 | 1.5000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task3_3 | 12 (4/8) | 2.3333 | 1.6667 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task3_4 | 36 (12/24) | 2.2222 | 1.6389 | 0.00% / 0.00% / 0.00% | 5.56% / 0.00% / 8.33% |
| task3_5 | 64 (16/48) | 2.2656 | 1.4375 | 0.00% / 0.00% / 0.00% | 9.38% / 0.00% / 12.50% |
| task3_6 | 12 (6/6) | 1.9167 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task3_7 | 12 (6/6) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task3_8 | 45 (15/30) | 2.0222 | 1.4000 | 0.00% / 0.00% / 0.00% | 53.33% / 0.00% / 80.00% |
| task3_9 | 80 (20/60) | 2.0000 | 1.1250 | 0.00% / 0.00% / 0.00% | 55.00% / 0.00% / 73.33% |
| task4_1 | 16 (8/8) | 1.9375 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task4_2 | 12 (6/6) | 1.9167 | 2.0000 | 0.00% / 0.00% / 0.00% | 8.33% / 0.00% / 16.67% |
| task4_3 | 12 (6/6) | 1.9167 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task4_4 | 12 (6/6) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task4_5 | 16 (8/8) | 2.0625 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task4_6 | 8 (4/4) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task4_7 | 12 (6/6) | 2.0833 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task5_1 | 32 (16/16) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task5_2 | 32 (16/16) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task5_3 | 32 (16/16) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task5_4 | 16 (8/8) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 18.75% / 0.00% / 37.50% |
| task5_5 | 16 (8/8) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 18.75% / 0.00% / 37.50% |
| task5_6 | 12 (6/6) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task5_7 | 12 (6/6) | 2.0833 | 2.0000 | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| task5_8 | 16 (8/8) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 37.50% / 0.00% / 75.00% |
| task5_9 | 16 (8/8) | 2.0000 | 2.0000 | 0.00% / 0.00% / 0.00% | 37.50% / 0.00% / 75.00% |

**Success steering（cohort）**

| task | n（suc/fail） | success MAE | 严格总准确率 | 严格 suc 准确率 | 严格 fail 准确率 |
| --- | --- | --- | --- | --- | --- |
| task1_1 | 127 (32/95) | 1.4724 → 0.4331 | 0.00% → 69.29% | 0.00% → 0.00% | 0.00% → 92.63% |
| task1_2 | 15 (4/11) | 1.5333 → 1.0667 | 0.00% → 53.33% | 0.00% → 0.00% | 0.00% → 72.73% |
| task1_3 | 95 (23/72) | 1.4211 → 0.5158 | 5.26% → 66.32% | 0.00% → 0.00% | 6.94% → 87.50% |
| task2_1 | 64 (16/48) | 1.4844 → 1.0625 | 0.00% → 34.38% | 0.00% → 0.00% | 0.00% → 45.83% |
| task2_2 | 36 (12/24) | 1.6667 → 1.4444 | 0.00% → 33.33% | 0.00% → 0.00% | 0.00% → 50.00% |
| task2_3 | 16 (8/8) | 2.0000 → 2.0000 | 0.00% → 12.50% | 0.00% → 0.00% | 0.00% → 25.00% |
| task2_4 | 96 (24/72) | 1.3125 → 0.6562 | 22.92% → 69.79% | 0.00% → 0.00% | 30.56% → 93.06% |
| task2_5 | 51 (17/34) | 1.5098 → 1.4118 | 21.57% → 49.02% | 0.00% → 0.00% | 32.35% → 73.53% |
| task3_1 | 24 (8/16) | 1.6667 → 1.5000 | 0.00% → 41.67% | 0.00% → 0.00% | 0.00% → 62.50% |
| task3_10 | 51 (17/34) | 1.5882 → 1.5686 | 17.65% → 37.25% | 0.00% → 0.00% | 26.47% → 55.88% |
| task3_2 | 16 (4/12) | 1.5000 → 1.0000 | 0.00% → 75.00% | 0.00% → 0.00% | 0.00% → 100.00% |
| task3_3 | 12 (4/8) | 1.6667 → 1.7500 | 0.00% → 25.00% | 0.00% → 0.00% | 0.00% → 37.50% |
| task3_4 | 30 (10/20) | 1.6333 → 1.4000 | 6.67% → 60.00% | 0.00% → 0.00% | 10.00% → 90.00% |
| task3_6 | 10 (5/5) | 2.0000 → 2.0000 | 0.00% → 20.00% | 0.00% → 0.00% | 0.00% → 40.00% |
| task3_7 | 10 (5/5) | 2.0000 → 2.0000 | 0.00% → 40.00% | 0.00% → 0.00% | 0.00% → 80.00% |
| task3_8 | 39 (13/26) | 1.4103 → 1.3846 | 51.28% → 61.54% | 0.00% → 0.00% | 76.92% → 92.31% |
| task4_1 | 16 (8/8) | 2.0000 → 1.4375 | 0.00% → 43.75% | 0.00% → 0.00% | 0.00% → 87.50% |
| task4_2 | 5 (2/3) | 1.6000 → 1.2000 | 20.00% → 60.00% | 0.00% → 0.00% | 33.33% → 100.00% |
| task4_3 | 12 (6/6) | 2.0000 → 1.1667 | 0.00% → 41.67% | 0.00% → 0.00% | 0.00% → 83.33% |
| task4_6 | 5 (1/4) | 1.4000 → 0.4000 | 0.00% → 80.00% | 0.00% → 0.00% | 0.00% → 100.00% |
| task4_7 | 11 (5/6) | 1.9091 → 1.1818 | 0.00% → 54.55% | 0.00% → 0.00% | 0.00% → 100.00% |
| task5_1 | 24 (8/16) | 1.6667 → 0.7083 | 0.00% → 54.17% | 0.00% → 0.00% | 0.00% → 81.25% |
| task5_2 | 15 (7/8) | 1.9333 → 0.7333 | 0.00% → 53.33% | 0.00% → 0.00% | 0.00% → 100.00% |
| task5_3 | 31 (15/16) | 1.9677 → 1.1613 | 0.00% → 41.94% | 0.00% → 0.00% | 0.00% → 81.25% |
| task5_4 | 8 (2/6) | 1.2500 → 1.0000 | 25.00% → 75.00% | 0.00% → 0.00% | 33.33% → 100.00% |
| task5_6 | 12 (6/6) | 2.0000 → 1.2500 | 0.00% → 50.00% | 0.00% → 0.00% | 0.00% → 100.00% |
| task5_7 | 11 (5/6) | 1.9091 → 1.3636 | 0.00% → 54.55% | 0.00% → 0.00% | 0.00% → 100.00% |
| task5_8 | 4 (1/3) | 1.0000 → 1.0000 | 50.00% → 75.00% | 0.00% → 0.00% | 66.67% → 100.00% |

</details>

<details>
<summary>text → images：34 个 full task 的双 head baseline，以及 28 个 cohort task 的 all_frames:target:8</summary>

**Baseline（full）**

| task | n（suc/fail） | progress MAE | success MAE | progress 严格 all/suc/fail | success 严格 all/suc/fail |
| --- | --- | --- | --- | --- | --- |
| task1_1 | 128 (32/96) | 1.5234 | 0.5781 | 28.91% / 96.88% / 6.25% | 71.09% / 100.00% / 61.46% |
| task1_2 | 45 (15/30) | 2.5333 | 2.3333 | 28.89% / 86.67% / 0.00% | 26.67% / 80.00% / 0.00% |
| task1_3 | 96 (24/72) | 1.1250 | 0.2917 | 37.50% / 100.00% / 16.67% | 81.25% / 100.00% / 75.00% |
| task2_1 | 64 (16/48) | 1.7188 | 0.5000 | 25.00% / 100.00% / 0.00% | 71.88% / 100.00% / 62.50% |
| task2_2 | 36 (12/24) | 1.4444 | 0.2222 | 33.33% / 100.00% / 0.00% | 83.33% / 100.00% / 75.00% |
| task2_3 | 16 (8/8) | 1.5625 | 1.0000 | 50.00% / 100.00% / 0.00% | 56.25% / 100.00% / 12.50% |
| task2_4 | 96 (24/72) | 1.8229 | 0.8750 | 21.88% / 87.50% / 0.00% | 58.33% / 87.50% / 48.61% |
| task2_5 | 54 (18/36) | 1.6111 | 0.7778 | 33.33% / 100.00% / 0.00% | 68.52% / 100.00% / 52.78% |
| task3_1 | 27 (9/18) | 2.7037 | 2.6667 | 29.63% / 88.89% / 0.00% | 25.93% / 77.78% / 0.00% |
| task3_10 | 54 (18/36) | 2.5741 | 2.3148 | 12.96% / 38.89% / 0.00% | 11.11% / 27.78% / 2.78% |
| task3_2 | 64 (16/48) | 2.8438 | 2.6719 | 17.19% / 68.75% / 0.00% | 10.94% / 43.75% / 0.00% |
| task3_3 | 12 (4/8) | 2.6667 | 2.5833 | 8.33% / 25.00% / 0.00% | 25.00% / 75.00% / 0.00% |
| task3_4 | 36 (12/24) | 2.4722 | 2.4444 | 13.89% / 41.67% / 0.00% | 19.44% / 33.33% / 12.50% |
| task3_5 | 64 (16/48) | 2.5625 | 2.2344 | 10.94% / 43.75% / 0.00% | 10.94% / 18.75% / 8.33% |
| task3_6 | 12 (6/6) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 41.67% / 83.33% / 0.00% |
| task3_7 | 12 (6/6) | 2.0000 | 1.6667 | 41.67% / 83.33% / 0.00% | 41.67% / 83.33% / 0.00% |
| task3_8 | 45 (15/30) | 2.2889 | 1.8667 | 4.44% / 13.33% / 0.00% | 17.78% / 6.67% / 23.33% |
| task3_9 | 80 (20/60) | 2.6125 | 2.1125 | 7.50% / 30.00% / 0.00% | 17.50% / 15.00% / 18.33% |
| task4_1 | 16 (8/8) | 2.0000 | 2.1250 | 50.00% / 100.00% / 0.00% | 31.25% / 62.50% / 0.00% |
| task4_2 | 12 (6/6) | 2.0000 | 1.8333 | 16.67% / 33.33% / 0.00% | 16.67% / 33.33% / 0.00% |
| task4_3 | 12 (6/6) | 2.0000 | 2.0833 | 33.33% / 66.67% / 0.00% | 41.67% / 83.33% / 0.00% |
| task4_4 | 12 (6/6) | 2.1667 | 2.1667 | 33.33% / 66.67% / 0.00% | 33.33% / 66.67% / 0.00% |
| task4_5 | 16 (8/8) | 2.0000 | 1.9375 | 37.50% / 75.00% / 0.00% | 31.25% / 62.50% / 0.00% |
| task4_6 | 8 (4/4) | 2.0000 | 2.0000 | 37.50% / 75.00% / 0.00% | 37.50% / 75.00% / 0.00% |
| task4_7 | 12 (6/6) | 2.0833 | 2.3333 | 41.67% / 83.33% / 0.00% | 33.33% / 66.67% / 0.00% |
| task5_1 | 32 (16/16) | 1.9375 | 1.8750 | 43.75% / 87.50% / 0.00% | 28.12% / 56.25% / 0.00% |
| task5_2 | 32 (16/16) | 2.0625 | 1.8750 | 25.00% / 50.00% / 0.00% | 9.38% / 18.75% / 0.00% |
| task5_3 | 32 (16/16) | 2.0000 | 2.0312 | 46.88% / 93.75% / 0.00% | 43.75% / 87.50% / 0.00% |
| task5_4 | 16 (8/8) | 1.8750 | 1.8125 | 37.50% / 75.00% / 0.00% | 37.50% / 62.50% / 12.50% |
| task5_5 | 16 (8/8) | 1.9375 | 1.7500 | 31.25% / 62.50% / 0.00% | 6.25% / 12.50% / 0.00% |
| task5_6 | 12 (6/6) | 2.0000 | 2.0833 | 50.00% / 100.00% / 0.00% | 41.67% / 83.33% / 0.00% |
| task5_7 | 12 (6/6) | 1.9167 | 2.0000 | 33.33% / 66.67% / 0.00% | 25.00% / 50.00% / 0.00% |
| task5_8 | 16 (8/8) | 1.9375 | 1.9375 | 18.75% / 37.50% / 0.00% | 31.25% / 37.50% / 25.00% |
| task5_9 | 16 (8/8) | 2.1250 | 1.9375 | 25.00% / 37.50% / 12.50% | 25.00% / 25.00% / 25.00% |

**Success steering（cohort）**

| task | n（suc/fail） | success MAE | 严格总准确率 | 严格 suc 准确率 | 严格 fail 准确率 |
| --- | --- | --- | --- | --- | --- |
| task1_1 | 127 (32/95) | 0.5827 → 0.0079 | 70.87% → 99.21% | 100.00% → 96.88% | 61.05% → 100.00% |
| task1_2 | 15 (4/11) | 2.4000 → 0.5333 | 26.67% → 80.00% | 100.00% → 50.00% | 0.00% → 90.91% |
| task1_3 | 95 (23/72) | 0.2947 → 0.0000 | 81.05% → 100.00% | 100.00% → 100.00% | 75.00% → 100.00% |
| task2_1 | 64 (16/48) | 0.5000 → 0.1094 | 71.88% → 96.88% | 100.00% → 93.75% | 62.50% → 97.92% |
| task2_2 | 36 (12/24) | 0.2222 → 0.3056 | 83.33% → 88.89% | 100.00% → 75.00% | 75.00% → 95.83% |
| task2_3 | 16 (8/8) | 1.0000 → 1.4375 | 56.25% → 50.00% | 100.00% → 62.50% | 12.50% → 37.50% |
| task2_4 | 96 (24/72) | 0.8750 → 0.1042 | 58.33% → 94.79% | 87.50% → 83.33% | 48.61% → 98.61% |
| task2_5 | 51 (17/34) | 0.7255 → 0.8627 | 70.59% → 72.55% | 100.00% → 41.18% | 55.88% → 88.24% |
| task3_1 | 24 (8/16) | 2.6667 → 1.6250 | 25.00% → 54.17% | 75.00% → 25.00% | 0.00% → 68.75% |
| task3_10 | 51 (17/34) | 2.3137 → 1.7255 | 11.76% → 50.98% | 29.41% → 17.65% | 2.94% → 67.65% |
| task3_2 | 16 (4/12) | 2.6250 → 1.0625 | 6.25% → 68.75% | 25.00% → 0.00% | 0.00% → 91.67% |
| task3_3 | 12 (4/8) | 2.5833 → 1.8333 | 25.00% → 50.00% | 75.00% → 50.00% | 0.00% → 50.00% |
| task3_4 | 30 (10/20) | 2.3667 → 1.6333 | 23.33% → 43.33% | 40.00% → 0.00% | 15.00% → 65.00% |
| task3_6 | 10 (5/5) | 2.0000 → 1.9000 | 40.00% → 50.00% | 80.00% → 40.00% | 0.00% → 60.00% |
| task3_7 | 10 (5/5) | 1.7000 → 2.1000 | 40.00% → 40.00% | 80.00% → 20.00% | 0.00% → 60.00% |
| task3_8 | 39 (13/26) | 1.9231 → 1.3846 | 12.82% → 64.10% | 7.69% → 0.00% | 15.38% → 96.15% |
| task4_1 | 16 (8/8) | 2.1250 → 0.1250 | 31.25% → 93.75% | 62.50% → 87.50% | 0.00% → 100.00% |
| task4_2 | 5 (2/3) | 1.8000 → 0.8000 | 20.00% → 80.00% | 50.00% → 50.00% | 0.00% → 100.00% |
| task4_3 | 12 (6/6) | 2.0833 → 0.4167 | 41.67% → 83.33% | 83.33% → 100.00% | 0.00% → 66.67% |
| task4_6 | 5 (1/4) | 3.0000 → 0.8000 | 20.00% → 80.00% | 100.00% → 100.00% | 0.00% → 75.00% |
| task4_7 | 11 (5/6) | 2.5455 → 0.4545 | 27.27% → 81.82% | 60.00% → 80.00% | 0.00% → 83.33% |
| task5_1 | 24 (8/16) | 2.3750 → 0.0833 | 16.67% → 91.67% | 50.00% → 87.50% | 0.00% → 93.75% |
| task5_2 | 15 (7/8) | 1.7333 → 0.0667 | 13.33% → 93.33% | 28.57% → 85.71% | 0.00% → 100.00% |
| task5_3 | 31 (15/16) | 2.0968 → 0.0968 | 41.94% → 96.77% | 86.67% → 100.00% | 0.00% → 93.75% |
| task5_4 | 8 (2/6) | 2.5000 → 1.0000 | 12.50% → 75.00% | 50.00% → 0.00% | 0.00% → 100.00% |
| task5_6 | 12 (6/6) | 2.0833 → 0.1667 | 41.67% → 91.67% | 83.33% → 83.33% | 0.00% → 100.00% |
| task5_7 | 11 (5/6) | 2.0000 → 0.1818 | 27.27% → 90.91% | 60.00% → 80.00% | 0.00% → 100.00% |
| task5_8 | 4 (1/3) | 1.7500 → 1.0000 | 25.00% → 75.00% | 100.00% → 0.00% | 0.00% → 100.00% |

</details>

<details>
<summary>交错输入：34 个 full task 的双 head baseline，以及 28 个 cohort task 的 all_frames:target:8</summary>

**Baseline（full）**

| task | n（suc/fail） | progress MAE | success MAE | progress 严格 all/suc/fail | success 严格 all/suc/fail |
| --- | --- | --- | --- | --- | --- |
| task1_1 | 128 (32/96) | 1.6875 | 1.0000 | 25.00% / 100.00% / 0.00% | 57.03% / 100.00% / 42.71% |
| task1_2 | 45 (15/30) | 2.6444 | 2.5556 | 28.89% / 86.67% / 0.00% | 26.67% / 80.00% / 0.00% |
| task1_3 | 96 (24/72) | 1.3229 | 0.6562 | 28.12% / 100.00% / 4.17% | 68.75% / 100.00% / 58.33% |
| task2_1 | 64 (16/48) | 1.8125 | 0.7188 | 25.00% / 100.00% / 0.00% | 59.38% / 100.00% / 45.83% |
| task2_2 | 36 (12/24) | 1.4167 | 0.5000 | 33.33% / 100.00% / 0.00% | 63.89% / 100.00% / 45.83% |
| task2_3 | 16 (8/8) | 1.6875 | 1.5000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task2_4 | 96 (24/72) | 2.0104 | 1.3333 | 25.00% / 100.00% / 0.00% | 39.58% / 87.50% / 23.61% |
| task2_5 | 54 (18/36) | 1.5741 | 1.1111 | 33.33% / 100.00% / 0.00% | 55.56% / 100.00% / 33.33% |
| task3_1 | 27 (9/18) | 2.6667 | 2.6667 | 33.33% / 100.00% / 0.00% | 33.33% / 100.00% / 0.00% |
| task3_10 | 54 (18/36) | 2.6667 | 2.4074 | 20.37% / 61.11% / 0.00% | 12.96% / 38.89% / 0.00% |
| task3_2 | 64 (16/48) | 2.9844 | 2.8125 | 25.00% / 100.00% / 0.00% | 17.19% / 68.75% / 0.00% |
| task3_3 | 12 (4/8) | 2.6667 | 2.6667 | 33.33% / 100.00% / 0.00% | 25.00% / 75.00% / 0.00% |
| task3_4 | 36 (12/24) | 2.5278 | 2.5278 | 25.00% / 75.00% / 0.00% | 19.44% / 41.67% / 8.33% |
| task3_5 | 64 (16/48) | 2.7812 | 2.5000 | 15.62% / 62.50% / 0.00% | 9.38% / 37.50% / 0.00% |
| task3_6 | 12 (6/6) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task3_7 | 12 (6/6) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task3_8 | 45 (15/30) | 2.4444 | 2.0889 | 15.56% / 46.67% / 0.00% | 15.56% / 20.00% / 13.33% |
| task3_9 | 80 (20/60) | 2.7125 | 2.4625 | 16.25% / 65.00% / 0.00% | 17.50% / 50.00% / 6.67% |
| task4_1 | 16 (8/8) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task4_2 | 12 (6/6) | 2.0833 | 1.9167 | 33.33% / 66.67% / 0.00% | 25.00% / 50.00% / 0.00% |
| task4_3 | 12 (6/6) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task4_4 | 12 (6/6) | 2.0000 | 2.0833 | 50.00% / 100.00% / 0.00% | 41.67% / 83.33% / 0.00% |
| task4_5 | 16 (8/8) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task4_6 | 8 (4/4) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task4_7 | 12 (6/6) | 2.0000 | 2.0833 | 50.00% / 100.00% / 0.00% | 41.67% / 83.33% / 0.00% |
| task5_1 | 32 (16/16) | 2.0000 | 1.9688 | 50.00% / 100.00% / 0.00% | 40.62% / 81.25% / 0.00% |
| task5_2 | 32 (16/16) | 1.9375 | 2.0312 | 50.00% / 100.00% / 0.00% | 25.00% / 50.00% / 0.00% |
| task5_3 | 32 (16/16) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 46.88% / 93.75% / 0.00% |
| task5_4 | 16 (8/8) | 2.0000 | 1.8750 | 43.75% / 87.50% / 0.00% | 43.75% / 75.00% / 12.50% |
| task5_5 | 16 (8/8) | 1.8125 | 1.8125 | 50.00% / 100.00% / 0.00% | 31.25% / 62.50% / 0.00% |
| task5_6 | 12 (6/6) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task5_7 | 12 (6/6) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 33.33% / 66.67% / 0.00% |
| task5_8 | 16 (8/8) | 2.0625 | 1.8750 | 18.75% / 37.50% / 0.00% | 18.75% / 37.50% / 0.00% |
| task5_9 | 16 (8/8) | 2.0625 | 1.9375 | 25.00% / 50.00% / 0.00% | 31.25% / 37.50% / 25.00% |

**Success steering（cohort）**

| task | n（suc/fail） | success MAE | 严格总准确率 | 严格 suc 准确率 | 严格 fail 准确率 |
| --- | --- | --- | --- | --- | --- |
| task1_1 | 127 (32/95) | 1.0000 → 0.0157 | 57.48% → 98.43% | 100.00% → 96.88% | 43.16% → 98.95% |
| task1_2 | 15 (4/11) | 2.6667 → 0.7333 | 26.67% → 66.67% | 100.00% → 50.00% | 0.00% → 72.73% |
| task1_3 | 95 (23/72) | 0.6632 → 0.0000 | 68.42% → 100.00% | 100.00% → 100.00% | 58.33% → 100.00% |
| task2_1 | 64 (16/48) | 0.7188 → 0.0625 | 59.38% → 96.88% | 100.00% → 93.75% | 45.83% → 97.92% |
| task2_2 | 36 (12/24) | 0.5000 → 0.2778 | 63.89% → 75.00% | 100.00% → 91.67% | 45.83% → 66.67% |
| task2_3 | 16 (8/8) | 1.5000 → 1.5625 | 50.00% → 56.25% | 100.00% → 75.00% | 0.00% → 37.50% |
| task2_4 | 96 (24/72) | 1.3333 → 0.1146 | 39.58% → 94.79% | 87.50% → 87.50% | 23.61% → 97.22% |
| task2_5 | 51 (17/34) | 1.0588 → 0.7647 | 56.86% → 70.59% | 100.00% → 52.94% | 35.29% → 79.41% |
| task3_1 | 24 (8/16) | 2.6667 → 1.7500 | 33.33% → 50.00% | 100.00% → 25.00% | 0.00% → 62.50% |
| task3_10 | 51 (17/34) | 2.4118 → 1.8824 | 13.73% → 41.18% | 41.18% → 17.65% | 0.00% → 52.94% |
| task3_2 | 16 (4/12) | 2.8125 → 1.4375 | 12.50% → 56.25% | 50.00% → 0.00% | 0.00% → 75.00% |
| task3_3 | 12 (4/8) | 2.6667 → 2.0000 | 25.00% → 50.00% | 75.00% → 50.00% | 0.00% → 50.00% |
| task3_4 | 30 (10/20) | 2.4667 → 1.6333 | 20.00% → 43.33% | 40.00% → 0.00% | 10.00% → 65.00% |
| task3_6 | 10 (5/5) | 2.0000 → 1.9000 | 50.00% → 50.00% | 100.00% → 40.00% | 0.00% → 60.00% |
| task3_7 | 10 (5/5) | 2.0000 → 2.0000 | 50.00% → 30.00% | 100.00% → 20.00% | 0.00% → 40.00% |
| task3_8 | 39 (13/26) | 2.1538 → 1.4872 | 12.82% → 53.85% | 23.08% → 0.00% | 7.69% → 80.77% |
| task4_1 | 16 (8/8) | 2.0000 → 0.3125 | 50.00% → 75.00% | 100.00% → 87.50% | 0.00% → 62.50% |
| task4_2 | 5 (2/3) | 2.2000 → 0.6000 | 20.00% → 80.00% | 50.00% → 50.00% | 0.00% → 100.00% |
| task4_3 | 12 (6/6) | 2.0000 → 0.5000 | 50.00% → 83.33% | 100.00% → 100.00% | 0.00% → 66.67% |
| task4_6 | 5 (1/4) | 3.2000 → 1.0000 | 20.00% → 60.00% | 100.00% → 100.00% | 0.00% → 50.00% |
| task4_7 | 11 (5/6) | 2.2727 → 0.4545 | 36.36% → 72.73% | 80.00% → 80.00% | 0.00% → 66.67% |
| task5_1 | 24 (8/16) | 2.5000 → 0.4167 | 33.33% → 75.00% | 100.00% → 100.00% | 0.00% → 62.50% |
| task5_2 | 15 (7/8) | 1.9333 → 0.0667 | 33.33% → 93.33% | 71.43% → 85.71% | 0.00% → 100.00% |
| task5_3 | 31 (15/16) | 2.0645 → 0.1290 | 45.16% → 96.77% | 93.33% → 100.00% | 0.00% → 93.75% |
| task5_4 | 8 (2/6) | 2.7500 → 1.1250 | 25.00% → 62.50% | 100.00% → 0.00% | 0.00% → 83.33% |
| task5_6 | 12 (6/6) | 2.0000 → 0.3333 | 50.00% → 75.00% | 100.00% → 83.33% | 0.00% → 66.67% |
| task5_7 | 11 (5/6) | 2.0909 → 0.1818 | 36.36% → 81.82% | 80.00% → 80.00% | 0.00% → 83.33% |
| task5_8 | 4 (1/3) | 2.2500 → 1.2500 | 25.00% → 50.00% | 100.00% → 0.00% | 0.00% → 66.67% |

</details>

<details>
<summary>官方输入：34 个 full task 的双 head baseline，以及 28 个 cohort task 的 last_frame:target:32</summary>

**Baseline（full）**

| task | n（suc/fail） | progress MAE | success MAE | progress 严格 all/suc/fail | success 严格 all/suc/fail |
| --- | --- | --- | --- | --- | --- |
| task1_1 | 128 (32/96) | 1.1641 | 0.3359 | 25.00% / 100.00% / 0.00% | 83.59% / 100.00% / 78.12% |
| task1_2 | 45 (15/30) | 2.6222 | 2.5111 | 33.33% / 100.00% / 0.00% | 33.33% / 100.00% / 0.00% |
| task1_3 | 96 (24/72) | 0.9062 | 0.1250 | 27.08% / 100.00% / 2.78% | 90.62% / 100.00% / 87.50% |
| task2_1 | 64 (16/48) | 1.7656 | 0.3438 | 25.00% / 100.00% / 0.00% | 68.75% / 100.00% / 58.33% |
| task2_2 | 36 (12/24) | 1.0556 | 0.1111 | 33.33% / 100.00% / 0.00% | 88.89% / 100.00% / 83.33% |
| task2_3 | 16 (8/8) | 1.5000 | 0.7500 | 50.00% / 100.00% / 0.00% | 56.25% / 100.00% / 12.50% |
| task2_4 | 96 (24/72) | 1.8750 | 0.8438 | 25.00% / 100.00% / 0.00% | 50.00% / 100.00% / 33.33% |
| task2_5 | 54 (18/36) | 1.2222 | 0.4630 | 33.33% / 100.00% / 0.00% | 72.22% / 100.00% / 58.33% |
| task3_1 | 27 (9/18) | 2.6667 | 2.7037 | 29.63% / 88.89% / 0.00% | 25.93% / 77.78% / 0.00% |
| task3_10 | 54 (18/36) | 2.6296 | 2.5370 | 31.48% / 94.44% / 0.00% | 20.37% / 61.11% / 0.00% |
| task3_2 | 64 (16/48) | 2.8906 | 2.4531 | 18.75% / 75.00% / 0.00% | 9.38% / 37.50% / 0.00% |
| task3_3 | 12 (4/8) | 2.6667 | 2.6667 | 33.33% / 100.00% / 0.00% | 33.33% / 100.00% / 0.00% |
| task3_4 | 36 (12/24) | 2.6667 | 2.6111 | 33.33% / 100.00% / 0.00% | 30.56% / 91.67% / 0.00% |
| task3_5 | 64 (16/48) | 3.0000 | 2.9375 | 23.44% / 93.75% / 0.00% | 20.31% / 81.25% / 0.00% |
| task3_6 | 12 (6/6) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task3_7 | 12 (6/6) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task3_8 | 45 (15/30) | 2.5333 | 2.2667 | 28.89% / 86.67% / 0.00% | 17.78% / 53.33% / 0.00% |
| task3_9 | 80 (20/60) | 2.9375 | 2.7875 | 23.75% / 95.00% / 0.00% | 21.25% / 75.00% / 3.33% |
| task4_1 | 16 (8/8) | 2.0000 | 2.0625 | 50.00% / 100.00% / 0.00% | 43.75% / 87.50% / 0.00% |
| task4_2 | 12 (6/6) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 41.67% / 83.33% / 0.00% |
| task4_3 | 12 (6/6) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task4_4 | 12 (6/6) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task4_5 | 16 (8/8) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task4_6 | 8 (4/4) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task4_7 | 12 (6/6) | 1.9167 | 2.0833 | 50.00% / 100.00% / 0.00% | 33.33% / 66.67% / 0.00% |
| task5_1 | 32 (16/16) | 2.0312 | 2.0000 | 46.88% / 93.75% / 0.00% | 31.25% / 62.50% / 0.00% |
| task5_2 | 32 (16/16) | 1.9375 | 1.8438 | 28.12% / 56.25% / 0.00% | 12.50% / 12.50% / 12.50% |
| task5_3 | 32 (16/16) | 2.0000 | 1.9375 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task5_4 | 16 (8/8) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 43.75% / 87.50% / 0.00% |
| task5_5 | 16 (8/8) | 2.0000 | 1.9375 | 50.00% / 100.00% / 0.00% | 37.50% / 75.00% / 0.00% |
| task5_6 | 12 (6/6) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 50.00% / 100.00% / 0.00% |
| task5_7 | 12 (6/6) | 2.0000 | 2.0000 | 50.00% / 100.00% / 0.00% | 41.67% / 83.33% / 0.00% |
| task5_8 | 16 (8/8) | 2.0000 | 1.8125 | 50.00% / 100.00% / 0.00% | 43.75% / 87.50% / 0.00% |
| task5_9 | 16 (8/8) | 2.0000 | 1.9375 | 37.50% / 75.00% / 0.00% | 37.50% / 62.50% / 12.50% |

**Success steering（cohort）**

| task | n（suc/fail） | success MAE | 严格总准确率 | 严格 suc 准确率 | 严格 fail 准确率 |
| --- | --- | --- | --- | --- | --- |
| task1_1 | 127 (32/95) | 0.3386 → 0.0000 | 83.46% → 100.00% | 100.00% → 100.00% | 77.89% → 100.00% |
| task1_2 | 15 (4/11) | 2.8667 → 0.8667 | 26.67% → 53.33% | 100.00% → 50.00% | 0.00% → 54.55% |
| task1_3 | 95 (23/72) | 0.1263 → 0.0000 | 90.53% → 100.00% | 100.00% → 100.00% | 87.50% → 100.00% |
| task2_1 | 64 (16/48) | 0.3438 → 0.0000 | 68.75% → 100.00% | 100.00% → 100.00% | 58.33% → 100.00% |
| task2_2 | 36 (12/24) | 0.1111 → 0.2222 | 88.89% → 86.11% | 100.00% → 83.33% | 83.33% → 87.50% |
| task2_3 | 16 (8/8) | 0.7500 → 1.0000 | 56.25% → 56.25% | 100.00% → 75.00% | 12.50% → 37.50% |
| task2_4 | 96 (24/72) | 0.8438 → 0.0000 | 50.00% → 100.00% | 100.00% → 100.00% | 33.33% → 100.00% |
| task2_5 | 51 (17/34) | 0.3922 → 0.2549 | 74.51% → 86.27% | 100.00% → 100.00% | 61.76% → 79.41% |
| task3_1 | 24 (8/16) | 2.7083 → 1.7917 | 25.00% → 45.83% | 75.00% → 37.50% | 0.00% → 50.00% |
| task3_10 | 51 (17/34) | 2.5294 → 1.5882 | 21.57% → 43.14% | 64.71% → 17.65% | 0.00% → 55.88% |
| task3_2 | 16 (4/12) | 2.4375 → 1.3750 | 6.25% → 56.25% | 25.00% → 0.00% | 0.00% → 75.00% |
| task3_3 | 12 (4/8) | 2.6667 → 2.2500 | 33.33% → 33.33% | 100.00% → 50.00% | 0.00% → 25.00% |
| task3_4 | 30 (10/20) | 2.5667 → 1.6667 | 33.33% → 50.00% | 100.00% → 0.00% | 0.00% → 75.00% |
| task3_6 | 10 (5/5) | 2.0000 → 2.7000 | 50.00% → 20.00% | 100.00% → 40.00% | 0.00% → 0.00% |
| task3_7 | 10 (5/5) | 2.0000 → 2.1000 | 50.00% → 40.00% | 100.00% → 20.00% | 0.00% → 60.00% |
| task3_8 | 39 (13/26) | 2.2051 → 1.4615 | 20.51% → 53.85% | 61.54% → 0.00% | 0.00% → 80.77% |
| task4_1 | 16 (8/8) | 2.0625 → 0.2500 | 43.75% → 75.00% | 87.50% → 75.00% | 0.00% → 75.00% |
| task4_2 | 5 (2/3) | 2.4000 → 0.6000 | 20.00% → 60.00% | 50.00% → 0.00% | 0.00% → 100.00% |
| task4_3 | 12 (6/6) | 2.0000 → 0.0000 | 50.00% → 100.00% | 100.00% → 100.00% | 0.00% → 100.00% |
| task4_6 | 5 (1/4) | 3.2000 → 0.8000 | 20.00% → 60.00% | 100.00% → 100.00% | 0.00% → 50.00% |
| task4_7 | 11 (5/6) | 2.1818 → 0.2727 | 36.36% → 90.91% | 80.00% → 100.00% | 0.00% → 83.33% |
| task5_1 | 24 (8/16) | 2.4583 → 0.7500 | 25.00% → 58.33% | 75.00% → 12.50% | 0.00% → 81.25% |
| task5_2 | 15 (7/8) | 1.4667 → 0.6667 | 20.00% → 73.33% | 14.29% → 42.86% | 25.00% → 100.00% |
| task5_3 | 31 (15/16) | 2.0000 → 0.1613 | 48.39% → 90.32% | 100.00% → 100.00% | 0.00% → 81.25% |
| task5_4 | 8 (2/6) | 2.8750 → 1.0000 | 25.00% → 75.00% | 100.00% → 0.00% | 0.00% → 100.00% |
| task5_6 | 12 (6/6) | 2.0000 → 0.0000 | 50.00% → 100.00% | 100.00% → 100.00% | 0.00% → 100.00% |
| task5_7 | 11 (5/6) | 2.0909 → 0.2727 | 45.45% → 81.82% | 100.00% → 80.00% | 0.00% → 83.33% |
| task5_8 | 4 (1/3) | 2.5000 → 1.0000 | 25.00% → 75.00% | 100.00% → 0.00% | 0.00% → 100.00% |

</details>

### 12.5 数据复核与解释边界

新增导出共 13,644 行总体/逐任务指标，覆盖六种输入、每种 baseline+18 干预、两个 head 和所有适用样本集；每个配置另存逐任务 JSON 与共同有效控制指标。逐任务指标从最终预测文件按 example_id 取最新记录复算，排除 pilot；468 个既有 population/head 汇总的样本数、MAE、两套端点正确数与 analysis_v1 一致，逐任务正确数可加总回总体。此项验证属于统计复算，未执行新推理。

新增分析按 11/11 项统计谬误检查复核：总体与 task/类别方向均披露；不从 task 均值推断逐例改善；保留 grounding 选择与同域限制；不据预测分数筛样；报告基率和始终失败反例；最低点明确为事后选择；无效控制公开并匹配共同 ID；success 与原检验 family 分开；.5 指标标为新增固定诊断；不推断闭环成功率；不据 head 结构解释训练机制因果。解释等级仍为 CAUTION，success 的拟合目标更合适与其泛化能力已获证实是两个不同判断。

建议后续以 success head 的二分类和平衡准确率、双类召回率及同视频配对作为主要判断指标，同时保留 progress 作为过程信号。在独立视频组上固定官方最后一帧 k32 与 text → images 全帧 k8 后，再检验类别取舍与任务退化。此建议不改变本轮历史主指标。

- [全部逐任务 CSV](../results/mydata_bench/experiments_v2_addbase/success_head_v1/metrics_by_task.csv)
- [补充索引、评分定义与来源指纹](../results/mydata_bench/experiments_v2_addbase/success_head_v1/index.json)
- [复算脚本](addbase_eval/summarize_success_head.py)（运行 `python -m mydata_bench.addbase_eval.summarize_success_head --output-name success_head_v2` 可新增一版，保留现有输出）

- [video → text：全部条件、样本集和 task JSON](../results/mydata_bench/experiments_v2_addbase/success_head_v1/meter_video_text.json)
- [text → video：全部条件、样本集和 task JSON](../results/mydata_bench/experiments_v2_addbase/success_head_v1/meter_text_video.json)
- [images → text：全部条件、样本集和 task JSON](../results/mydata_bench/experiments_v2_addbase/success_head_v1/meter_image_text.json)
- [text → images：全部条件、样本集和 task JSON](../results/mydata_bench/experiments_v2_addbase/success_head_v1/meter_text_image.json)
- [交错输入：全部条件、样本集和 task JSON](../results/mydata_bench/experiments_v2_addbase/success_head_v1/meter_interleaved.json)
- [官方输入：全部条件、样本集和 task JSON](../results/mydata_bench/experiments_v2_addbase/success_head_v1/meter_official.json)
