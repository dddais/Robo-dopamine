# 五个模型的评估实现复核（2026-09-13）

**不能把“主路径通过实现验证”写成“五个模型所有实验代码及所有历史结果都正确”。** 本次检查 A 仓库 `mydata_bench` 的 baseline、普通 attention ranking/steering 和评分口径，补齐发现的入口与解析漏洞。可用结果须限定到配置、输入协议、样本集合及完整条件；SOLE official 的旧指标仍作废，修正后尚无全量指标。

| 模型 | 当前实现判断 | 历史结果使用边界 |
| --- | --- | --- |
| GRM | forward 主路径通过真实权重检查；official incremental 有单元回归及整段执行检查 | 官方全量 vLLM baseline、HF attention 内 baseline 分别有效，不能互相替代；旧 local-hop 不是整段 progress；缺控制样本时用共同样本集合 |
| RoboReward | 原生视频、独立八帧、交错八帧的主路径通过真实权重检查；修正了共享解析器边界漏洞 | 未发现已成功的主实验输出触发新修正；保留原指标，但不同输入顺序、视频采样、图像协议须分别解释 |
| Qwen3-VL | 同上；Qwen 是通用 VLM，reward prompt 属于本项目评估任务 | 不完整条件不能按完整 846 条 cohort 比较；本次没有重跑全量性能 |
| Robometer | 沿用已完成的官方输入、权重/head、数值及 steering 核查 | A official success head 的 baseline 和完整同 cohort steering 可以使用；progress head 与 success head 分开报告 |
| SOLE-R1 | 保留单一修正实现，official 输入和七步独立反馈通过既有真实权重检查；本次补齐探索入口 | 旧 official baseline/ranking/steering 不可使用；修正后需要重跑，当前没有可报告的全量新指标 |

## 本次发现并修正的代码问题

1. `auto_research_addbase/runtime.py` 覆盖了父类 `prepare/collect`，遗漏反馈列表长度与八帧约束，ranking observations 也没写 `sole_protocol_version`。修正后的 SOLE ranking 首次写出后，恢复运行会被新版缓存校验拒绝。现已补齐检查和协议标记，并在每次 prepare 校验协议，覆盖 worker 复用模型切换配置的情况。新增测试通过生产 `rank()` 首次运行和恢复运行，验证使用第六步的精确百分比反馈。
2. `roboreward_eval/runner.py::parse_native_score` 原来使用首个正则匹配：`ANSWER: 1.5` 被读成 1，冲突的 `ANSWER: 5 ... ANSWER: 1` 被读成 5。RoboReward、Qwen 及探索运行时共用它。现在只接受单个、明确的 1–5 整数答案，仍允许解释文字和常见标点；不合格输出抛错，不能成为有效低分。
3. `attention_eval/masking.py` 原来遇到 `attention_mask=None` 静默跳过，bool mask 也可能错误转换偏置符号。当前 GRM/Qwen/RoboReward 强制 eager，实际检查使用的是正确的四维浮点 additive mask，历史成功记录也没有“hook 标为开启但无任何应用”的情况。现增加显式拒绝不支持的 mask、负 token 位置和正负区域重叠，避免未来更换后端后静默产生错误结果。这是未在历史主运行触发的保护性修正。

探索入口仍叫 `configs/v2_crossmodel/auto_20260911/sole_official.json`，已指向同一修正协议；标准入口仍是 `configs/v2_crossmodel_addbase/sole_official.yaml`，没有新建第二套 SOLE 实现。剩余旧探索输出已从 active experiments 移到：

`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911/retired/sole_official_20260913/`

这是可恢复的历史归档，不是可运行的旧实现，也不能用于当前结论。探索 SOLE 的默认 worker 复用标准入口生成的 ranking，须先用修正标准入口生成有效 ranking。

## 验证证据

真实权重检查脚本：[verify_core_execution.py](verify_core_execution.py)。结果只写入独立的 [audit_20260913](../results/mydata_bench/audit_20260913/)；没有覆盖历史预测或重新选择最优 k。

- GRM forward：`fail/ljx_lfz_task_1_1/1`、`suc/ljx_lfz_task_1_1/1`。直接执行 `examples/inference.py` 中官方类的请求构造，用捕获的官方 prompt/images 构造 HF tensors，与生产路径逐 tensor 相等；直接 greedy 解码文本相等；36×32 ranking 有限；真正安装 bias=0 hook 时输出不变；bias=6 hook 应用成功；退出 hook 后 baseline 恢复。结果见 `grm_core_execution.json`。
- GRM official incremental：同一失败视频的完整相邻 hop 计划，执行 baseline → 每个 hop steering → baseline；独立重算正/负增量递推，与每步保存的累计值严格一致；最后一次 baseline 的所有逐步结果与首次相同。见 `grm_incremental_execution.json`。
- RoboReward/Qwen：各两样本 × 原生视频、独立八帧、交错八帧三种协议。生产 Qwen baseline 输入方法与 attention 内 baseline 的 tensors、greedy 文本相等；RoboReward 另执行其专用 baseline runner 的原生视频和独立图像方法，也相等。36×32 ranking、零偏置、非零偏置、baseline 恢复和 all-frame bbox 映射均检查。见 `qwen_core_execution.json`、`roboreward_baseline_and_core_execution.json`。原生视频比较显式共用 attention 的八帧上限；没有把完整 baseline 的默认采样冒充同条件输入。
- 非零 steering 的 smoke 使用固定的一个 head 验证执行，不是最优 head/k 的性能实验；不能把这些样本的分数当作新的研究指标。
- 在线获取 RoboReward、Qwen 官方 Hugging Face model cards，均与本地文件逐字节一致；RoboReward prompt 与官方卡片一致。见 `upstream_model_cards.json`。
- 扫描主实验 35 个目录、325,363 条原始记录（含 merged/shard 的重复记录，**不是独立样本数**）。已成功输出中没有非有限 progress，没有新解析规则拒绝的离散答案；282,990 条标记 hook active 的记录均有实际 applied calls。266 条非 ok 原始记录的存在与已有总结中的控制缺失/解析失败一致。见 `historical_core_rows.json`。此扫描不等于全量模型复现或所有统计量重算。
- Robometer/SOLE 的既有官方核对、两样本七步验证、Robometer 原 batch size=4 精确复现，以及 Robometer 全条件重新评分，详见 [addbase 专项报告](addbase_eval/REVIEW_20260913.md)。

最终回归：**209 passed，另有 17 subtests passed**。覆盖 `tests/test_mydata_{grm_incremental,attention_resume,ranking,ranking_grounding_guard}.py`、`mydata_bench/tests`、addbase execution contracts、SOLE protocol，以及探索方法的现有测试。`rewardbench` 是另一个实现目录，其测试通过不能直接算作本次 `mydata_bench` 生产路径的证据；上述新增解析/mask 测试明确导入 `mydata_bench`。这些测试加小样本执行检查仍不等于各探索方法已经完整审计。

特别核查了两项容易误判的实现：

- HF fast image processor 的 `min_pixels/max_pixels` 赋值在本环境确实生效。对 640×480 图像，GRM 网格从 `[1,30,40]` 改为 `[1,14,20]`；Qwen 从 `[1,30,40]` 改为 `[1,12,16]`，与显式传参一致。RoboReward 本来就使用较小的 checkpoint 默认图像预算。
- 原生视频 `_prepare_native` 会要求采样最后索引等于原视频 `total_num_frames-1`，否则报错；因此 last bbox 对末帧的映射不是静默套用到更早采样帧。末个视觉 span 仍是 temporal tubelet，不能描述为未混合的单独末帧。

## 与 B 仓库、官方协议的关系

B：`/home/dais/workspace/Robo-Dopamine-addbase`。GRM 官方示例、prompt/递推、raw runner、HF attention runtime，以及 Qwen baseline/attention runtime 在本次检查前与 A 一致；RoboReward runner 也在本次解析器修正前一致。现在 A 的共享 mask 和离散 parser 多了上述检查。SHA256 记录见 `a_b_source_comparison.json`。两仓库相同只能证明没有分叉，不能替代对官方实现和数值执行的验证；本次未修改 B。

GRM full baseline 使用 vLLM，外部 AutoProcessor 只生成 prompt；传给 vLLM 的是原始 PIL 图片，没有传入 `mm_processor_kwargs`。因此外部设置的 min/max 不控制 vLLM 的内部图像预处理，内部使用 checkpoint 默认预算。官方 `examples/inference.py` 也如此。HF attention 则真正以配置的 12,544–76,800 像素预算处理图片。这与 temperature=0.1/top_p=0.9/top_k=50/max_tokens=1024 对 HF greedy/max_new_tokens=16 的差异一起，构成两种 baseline 协议的区别。这里不为“统一”而修改已有官方示例复现路径；attention 效果必须与同一 HF 运行协议内的 baseline 比较。

RoboReward 的独立图像/交错图像是显式输入改造，Qwen 使用 RoboReward rubric 是任务适配。它们可以作为受控实验结果，但不能称作原论文完整官方 benchmark 的复现。

## 结果使用边界

- 保留原标签、五档 MAE 和端点准确率定义；没有换成二分类 accuracy 来提高数字。steering 与 baseline 必须同 cohort、同输入、同解码，控制比较再取共同成功样本。
- Robometer success head 原数值不变：全量 1213 条 baseline MAE **1.5664**、准确率 **47.57%**；846 条 cohort baseline **1.2258 / 56.03%**；同 cohort official/最后帧/k32 **0.5201 / 80.97%**。
- Qwen image-sequence 实验 11/12 的 k64 不完整；15/16 只有 baseline 和 target-k8 完整。GRM official incremental wrong-region 各缺 8 条 suc；13 official 的 k64 target/low-rank 只有 844 条。具体边界沿用 [crossmodel 总结](exp_plan_crossmodel_summary.md) 和 [GRM 总结](exp_plan_GRM_summary.md)，不能将这些条件扩写成完整 846 条结果。
- `auto_research_addbase` 的大量方法变体（logit contrast、learned gates、head delta 等）没有在本次逐方法做真实权重和完整统计审计。共享层测试通过不能为全部变体背书。
- 本次没有全量重新推理，也没有做多随机种子、跨任务独立复现。实现正确、历史记录完整、steering 在独立数据有效，是三件需要分别证明的事。
