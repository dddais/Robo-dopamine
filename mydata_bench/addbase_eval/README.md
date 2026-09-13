# Robometer / SOLE-R1 新增 baseline

仅新增实现，供仓库根目录 `exp_plan_addbase.md` 使用。Robometer 入口是
`mydata_bench.meter_eval`；SOLE-R1 入口依计划命名为 `mydata_bench.top_eval`，并非 TOPReward。

2026-09-13 完整复核见 [REVIEW_20260913.md](REVIEW_20260913.md)：35 项 CPU 检查、
官方源码与实际 processor 对照、两个模型的真实权重小样本核验、历史 197,292 条输出审计。
修正版 SOLE 尚未重跑全量实验。Robometer 历史 official 实际 batch size 为 4（YAML 默认 1）；
逐位复现应以保存的 `run_config.json`、批次分组和环境为准。

## 运行

在 Robo-Dopamine 根目录、robo-dopamine conda 环境中执行：

```bash
python -m mydata_bench.addbase_eval.prepare
CUDA_VISIBLE_DEVICES=0 python -m mydata_bench.meter_eval \
  --config mydata_bench/configs/v2_crossmodel_addbase/meter_interleaved.yaml --batch-size 16
CUDA_VISIBLE_DEVICES=1 python -m mydata_bench.top_eval \
  --config mydata_bench/configs/v2_crossmodel_addbase/sole_official.yaml
python -m unittest mydata_bench.addbase_eval.test_contracts
CUDA_VISIBLE_DEVICES=0 python -m mydata_bench.addbase_eval.verify_reference
python -m mydata_bench.addbase_eval.score --output-name analysis_v1
```

## SOLE 官方输入修正（2026-09-13）

SOLE 只保留一个配置 `sole_official.yaml`，其内容已替换为修正后的实现，输出目录为
`sole_official/`。原独立的 `sole_official_v2.yaml` 已移除；默认矩阵和配置生成器均使用
这个唯一入口。内部 `sole_protocol_version: sole_official_v2` 仅用于识别正确缓存，
不代表保留另一套模型或旧推理代码。旧配置和无版本标记的预测不再允许推理或评分。
旧 official / pilot 原始结果已从运行目录移至
`results/mydata_bench/experiments_v2_addbase/retired/sole_official_20260913/`，不参与当前运行。
本次未重跑全量模型，历史 SOLE 指标不是修正版指标。Robometer 的输入和评分口径不变。

修正内容：

- 递推直接使用解析后的百分比字符串，例如 `57`；不再经 `0.57 * 100` 产生
  `56.99999999999999`。保留有效小数原文，越界值先裁剪至 [-100,100] 再反馈，
  首步仍为 `0`，各条件使用自身上一绝对进度，不累加。
- 官方拼图先按 RewardGen 的 factor-28/PIL 路径处理，再交给 checkpoint processor
  执行原生 resize；移除手工第二次 PIL resize 和对此输入的 `do_resize=False`。
- 预测、步骤、ranking 观察和 head 排名均带 `sole_protocol_version`，拒绝复用
  未标记或不兼容的旧缓存。清理旧结果后，baseline、ranking、steering 必须完整生成。
- 保持 greedy/512-token 的实验解码设置，显式关闭 sampling 参数。这仍不是官方
  随机解码结果的复现；普通五种输入仍是八帧单次预测，只有 official 做七步递推。
- 保持原干预定义：`last_frame` 为每一步的当前帧区域，`all_frames` 为每一步三个
  时刻的内容区域，两者都干预七步。矩形相交的网格边界可能含黑边或邻帧像素。

CPU 回归检查（仅读取本地 processor 文件，不加载模型权重）：

```bash
python -m unittest mydata_bench.top_eval.test_protocol mydata_bench.addbase_eval.test_contracts mydata_bench.addbase_eval.test_execution_contracts
```

检查覆盖官方文本/token/pixel tensor 的直接一致性、左侧 padding、精确百分比反馈、
各条件独立递推、格式失败隔离、选头前驱及历史缓存拒绝。未安装本地 SOLE processor
时对应两项检查会跳过，可用 `SOLE_PROCESSOR_PATH` 指定文件目录。

修正版实验完成后单独统计：

```bash
python -m mydata_bench.addbase_eval.score \
  --configs mydata_bench/configs/v2_crossmodel_addbase/sole_official.yaml \
  --output-name analysis_sole_official
```

`--configs` 定义本次完整比较矩阵，因此上述分析每个 population 的 Holm family 为
6 个 target 条件；不能与历史 72 条件校正混称。未提供该参数时仍读取历史冻结矩阵。

## 续跑与历史统计

`--phase baseline/rank/steer/all` 控制执行阶段。`--limit N --output-suffix _pilot`
只用于独立 pilot 目录。正式实验不设 limit。`--retry-runtime-errors` 仅对已记录的运行错误
新增重试记录，不覆盖旧行；解析错误保留原始输出，不能自动当作失败任务或 reward=1。

配置与新汇总使用排他创建；如同名文件已有不同内容，程序报错。重做统计须使用新的
`--output-name`。预测文件和阶段文件只追加，以每个 example_id 最新一行为评分记录。

## 官方协议与共同对照

- Robometer 严格载入全部 738 个权重张量，包括 checkpoint 的历史 similarity head；进度
  来自实际训练的 prog_token → 10-bin progress MLP → softmax 期望。success probability
  另外保存，不与 progress 混用。基础 Qwen3-VL-4B 目录只提供 tokenizer/processor。
- SOLE 使用 checkpoint 自带 processor；official 从官方 RewardGen 源码直接取纯 prompt 和
  拼图函数，按首/前/当前图像递推。上一步预测是绝对进度；首帧规定为 0，不累计相加。
- 共同对照固定八个时刻及正面视角。official 原生分辨率和官方 SOLE 上下文作为额外条件，
  不与共同对照混称纯顺序对照。所有 SOLE 配置用 greedy，故不是复现官方随机采样分布。

## check.md 实现核对

| 检查项 | 本实现及证据 |
| --- | --- |
| bias 广播所有 query | `attention.py` 的 key bias 为 B×H×1×K，叠加原 causal/padding mask；数值测试覆盖 prefill、decode、未来 key、未选 head |
| bbox 与实际视觉 key 对齐 | 处理后的 image/video token 连续段与 grid_thw 逐项核验；每行保存源帧、跟踪帧、bbox、token positions |
| video temporal span | 明确一对源帧对应一个 tubelet，取两个 bbox 的并集；do_sample_frames=False 避免重复隐式采样 |
| 排除早期层 | 排除零基 L0–L7；低排名对照同样不使用这些层 |
| ranking 指标 | 未干预 mean raw mass；同时记录总 visual mass，不依据标签挑 head |
| ranking query | Robometer 最后 prog_token；SOLE 最后 prompt token；不混称 decode token |
| wrong-region | 同时域等 token 数、无交集、按网格距离选远处；可行区域不足仅标记该对照不可用 |
| 时域范围 | last：末 image / 最后 tubelet / 拼图当前帧；all：所有输入帧或拼图三个时刻；负 bias 只在相应有效视觉域内 |
| baseline 顺序和 prompt | 五种顺序及额外 official 全覆盖，逐样本保存完整序列化 prompt 与 token 哈希 |
| GRM 八图与多视角 | 不修改任何旧 GRM 实现；新模型共同对照只用正面视角；SOLE official 用官方 external-only 分支 |
| incremental | SOLE 每个条件从 0 独立递推，上一步值取该条件自身输出；每步原文、输入、曲线全保存 |
| 输出与标签对应 | labels_for_scoring_only.json 与模型输入分离；以 example_id 连接；同视频配对另核验 source_suc_id 和视频哈希 |

## 证据文件

实验输出：`results/mydata_bench/experiments_v2_addbase/`。每个配置包含：

- `run_config.json`、`loading_audit.json`、`events.jsonl`；
- `predictions/*.jsonl`：全量 baseline、2 范围 × 3 个 k × 3 种区域/head 条件；
- `ranking_observations.jsonl`、`ranking_last_frame.json`、`ranking_all_frames.json`；
- SOLE official 的 `steps/*.jsonl`：逐步推理及自身预测递推记录；
- 独立统计目录：五档 MAE、两套阈值准确率、task/suc/fail 分布、同视频配对、视频组 bootstrap、Holm 校正。

完整有效样本数与固定期望样本数分别记录。MAE 另外给出将无效输出误差视为 [0,4]
得到的全体上下界；准确率同时列有效输出分母及固定分母。不得把解析失败删掉后假称全量完成。

ranking 原始发现集与 cohort 有交集，因此额外排除所有 ranking 视频组，得到 730 条留出
样本。该集仍属于相同任务域，不是独立外部数据集；head 编号跨模型相交不等价于功能相同。

空间映射采用“矩形相交的所有视觉单元”。在 SOLE 官方拼图中，边界单元可以同时含当前帧、
相邻帧或黑边像素；last_frame 不能解释为严格的单时刻像素隔离。确定性示例的可视化与几何
核对见 [geometry_audit_v1.md](../../results/mydata_bench/experiments_v2_addbase/research/geometry_audit_v1.md)。
`python -B -m mydata_bench.addbase_eval.progress` 可只读查看当前调度和最新无效输出类型。

最终统计同时保存 MAE 奖励档分布（`ordinal_prediction_distributions`）与等宽进度分布
（原 `prediction_distributions`）。全 population 的 task 分布 CSV 用 `binning` 明确区分二者；
端点准确率仍按两套 low/high 阈值单独计算。详见
[reporting_addendum_v1.md](../../results/mydata_bench/experiments_v2_addbase/research/reporting_addendum_v1.md)。
