# Basic method 全量实验就绪性审查

## Material Passport

- 检查日期：2026-09-13。
- 范围：本仓库的 `exp_plan_basic_method.md`、`exp_use_basic_method.md`、`mydata_bench/basic_method`、20 份配置及其依赖。未参考其他本地仓库。
- 状态：**主推理流程可运行，完整交付尚未就绪**。这是执行和报告契约检查，不是全量性能复现。
- 当前实现指纹：`7637ce14c629b7ece79cbf0a32f899c7edadd6a5a33157ab86fd6523e9b27026`，与已有 `verification/validation_summary.json` 一致。
- 本次仅新增审计产物；未修改实验实现、配置、计划、使用说明或既有预测。未启动完整矩阵。

## 结论

按使用说明，prepare、preflight、推理/排名/干预和正常完成后的续跑能执行；但**现在直接跑完整矩阵，再执行 score，不能完整满足计划要求**。最明确的缺口是 Robometer success head 未纳入新 scorer。另需区分记录完成与预测有效，并修正报告的 pairwise 单位说明。建议先补齐报告与验收口径，再冻结代码开始全量运行。

## 发现的问题

### 1. 必须补齐：Robometer success head 只保存，没有统计

计划明确要求 success head 纳入统计。`basic_method/runtime.py:249` 会保存 `success_probability` 和 `per_frame_success`，但是 `basic_method/score.py:52` 直接对原始行调用通用 `summary()`；该函数只读取 `progress`。整个新 scorer 没有 success head 分支，模型 summary、总 summary、CSV、逐 task、pairwise 和共同控制子集均只含 progress head。

已用独立合成数据调用实际 scorer 复现：progress head 完全正确、success head 完全相反时，报告只得到 MAE=0，success head 应为 MAE=4 的结果完全缺失。证据：`verification/readiness_20260913/score_contract_reproduction.json`。合成数据仅用于检查报告契约，不属于实验结果。

建议：为 Robometer 两个 head 分别输出 full/holdout、各 task、预测分布、pairwise、SAS/对照和最佳配置；明确 head 名称，不能择优混合。原始输出已保存 success head，推理接口本身不需要因此重写。

### 2. 实测限制：completion 不等于全部预测有效，续跑不重试解析失败

按照使用说明的 pilot 方式实际运行 Qwen interleaved：评测 2 条、ranking 2 条、19 个条件，共 38 条预测。结果是 **37 ok / 1 parse_error**：

- 样本：`suc/ljx_lfz_task_1_1/2`。
- 条件：`last_frame:wrong_region:32`。
- 原文：`5`，共 2 个生成 token；这次失败不是 max_new_tokens 截断。
- 原因：当前统一协议要求 `ANSWER: <1-5>`，因此严格解析器拒绝裸整数。

程序仍以退出码 0 完成并写出 `completion.json`。再次运行同一命令、隐藏 GPU，返回 `Already complete`。这是 `run.py` 按已保存 ID 续跑的现有语义。`score.py:38` 的 complete 标记同样只检查 ID 是否齐全；合成测试证明有无效预测仍可得到 `complete=true`。

现有 scorer 会把无效输出计入 invalid，accuracy 使用固定分母，MAE 给出有效子集及全量上下界；**这不是把解析失败伪造成正确预测**。但使用说明中“全量有效条件输出”的说法过强，也不能把“完整矩阵结束”当作“每条件 1213 条有效预测”。

建议：单独输出 `records_complete`、`predictions_valid_complete` 和各条件状态统计；明确解析失败的验收规则。若决定接受裸整数，应在全量运行前统一冻结解析规则，不能只对表现不好的条件临时放宽。普通续跑不会自动修复已有 parse_error。

### 3. 报告单位：主表的 pairwise Δ 是归一化 progress 差

`score.py:89` 将 `continuous_mean_delta` 放入未注明单位的 `pairwise Δ` 列。RoboReward/Qwen 保存 `progress=(reward-1)/4`，所以 suc=5、fail=1 时主表显示 **1.0**，不是原生 reward 差 **4**。

详细 JSON 的 `ordinal_difference_counts` 已正确统计 `<0/0/1/2/3/4`，不是全部 pairwise 计算都错了。应为离散模型额外报告原生 reward 差，并把归一化差的列名/单位写清楚，避免总 summary 被按计划中的 1–5 分差误读。

### 4. 设计边界：只搜索固定 bias=6 下的配置

20 份配置均固定 ±6，没有 bias 扫描。可以选出固定强度下的最佳输入、scope、k；不能据此声称 ±6 是最佳 bias。计划的“最佳 bias”若要求比较不同强度，还需在开跑前定义候选值及独立配置。若本轮只研究 basic SAS，则在报告中明确固定强度即可。

## 已通过的检查

| 项目 | 证据 |
| --- | --- |
| 全量输入、配置及不可覆盖约束 | 重复执行 prepare 成功；20 份配置均通过 validate_config/validate_inputs，所有冻结输入哈希一致 |
| 样本规模 | 1213 条评测、407 个视频；release eligible 1116；独立 ranking 36，release eligible 33；排除 ranking 视频组的 holdout 1089 |
| 矩阵 | 5 模型×4 输入=20 个 baseline；20×2 范围×3 k×3 干预=360 个 SAS/对照；合计 460,940 条评测条件记录，不含 ranking 和 SOLE 中间步骤 |
| 精确框与 baseline 回退 | 所需源帧缺任一框即整条对应条件回退；不借邻帧框；缺文件/身份错误不冒充 grounding 缺失；baseline 构造不读取 bbox |
| GRM 范围 | forward；last 只作用 after high；all 仅 reference start、before high、after high，腕相机和空白 goal 不干预 |
| SOLE | 七步开始前检查全部所需框；条件独立递归；ranking 使用对应未干预历史；官方输入使用既有修正协议 |
| 原生视频 | 使用实际采样索引和两帧 tubelet；目标为源框 token 集的并集；验证了奇数帧 padding 情况 |
| 控制 | wrong region 同范围、等 token 数、不重叠；不可构造单独记录 control_unavailable；共同控制统计使用共同有效且真正施加干预的样本 |
| 测试 | 本次 41 tests passed，另有 17 subtests passed；未将单元测试当作全量性能证据 |
| CLI | SOLE 两样本 processor-only preflight 成功；Qwen 两样本完整 19 条件执行及无 GPU 续跑成功，存在上文明确列出的 1 条格式失败 |
| 资源 | 检查时 GPU 0–3 为 A100 80GB，均空闲；结果所在卷约 1.8TB 可用；最大 Qwen native 输入压力检查通过 |

矩阵数量按计划最后明确要求的四种输入和 target/wrong/low-rank 三类条件计算。计划中的“约 15/180”不是本次实际矩阵数量。GRM official 与 interleaved 输入等价，重复结果不能当独立复现。

## 原生视频覆盖和压力检查

根据冻结视频元数据调用真实 checkpoint `sample_frames`，覆盖结果如下；另实际解码处理了 RoboReward 两个样本和 Qwen 三个样本（含奇数帧和最大输入）进行核对。

| 模型/群体 | last_frame 可干预 | all_frames 可干预 |
| --- | ---: | ---: |
| RoboReward 评测 | 1074/1213 | 531/1213 |
| RoboReward ranking | 31/36 | 29/36 |
| Qwen 评测 | 1115/1213 | 396/1213 |
| Qwen ranking | 33/36 | 27/36 |

因此没有发现 native ranking 因有效样本少于 2 条而必然中断的问题。Qwen all-frame 只有约 32.65% 样本实际 SAS，其余使用 baseline，这符合计划，但解释全量增益时必须同时报告 applied-only 和回退占比。

当前数据 Qwen native 实际采样为 15–52 帧；最大视觉输入约 7800 token。实际对 `suc/ljx_lfz_task_1_3/22` 运行 8189-token、52 帧的 baseline、attention mass 采集和 k64 干预，均成功，PyTorch 峰值 allocated **26.38 GiB**、reserved **27.13 GiB**。此测试的 heads 来自该样本，只用于执行和显存压力检查，不是独立 ranking 的实验结果，也不用于评价 SAS 效果。

## 开跑前的顺序

1. 补齐 success head 报告；明确完成/有效两类标记、pairwise 单位及固定 bias 的结论范围。
2. 对最终代码补跑报告回归和小规模 CLI 验证，使用新验证后缀；现有标准全量目录尚无正式运行身份文件。
3. 冻结实现后，按 `exp_use_basic_method.md` 执行 prepare → schedule → score。全量结束后同时验收 380 个条件的记录数、有效数、回退数、control_unavailable、parse_error，而不是只查看进程退出码或 completion。

必须在长实验前处理这些修改：`common.py:124` 的实现指纹包含 scorer 和 tests。运行中修改这些 Python 文件也会导致续跑身份不兼容；当前没有开始正式全量矩阵，适合先补齐再冻结。

## 本次新增证据

- `results/mydata_bench/experiments_v2_basic_method/verification/readiness_20260913/readiness_summary.json`
- 同目录：`native_sampling_coverage.json`、`native_processor_spotcheck.json`、`qwen_largest_input_stress.json`、`score_contract_reproduction.json`。
- `results/mydata_bench/experiments_v2_basic_method/sole_official_readiness_20260913/`：processor preflight。
- `results/mydata_bench/experiments_v2_basic_method/qwen_interleaved_readiness_pilot_20260913/`：两样本 19 条件及实际失败记录。
