# mydata_bench 迁移与实验设计记录

## 原 rewardbench 的实验逻辑

原代码把评测分成四层：数据层只给模型 instruction 和媒体；raw baseline 按模型协议推理；grounding 产生目标 bbox/mask 并接受人工审核；attention 实验先独立发现 heads，再运行 baseline、target、wrong-region、low-rank 等条件。reward 只在推理结束后加入 metrics。迁移保持了这个分层，原 rewardbench 目录没有修改。

## 新数据结构

data/ljx_lfz_task/new/metadata.jsonl 共 755 条：

- suc 169 条：真实成功轨迹，映射 reward=5。
- fail 586 条：复用某条 suc 的真实轨迹，但 instruction 与实际对象不匹配，映射 reward=1。
- 每条包含 front、left wrist、right wrist 三个 MP4。
- fail 通过 source_suc_id 指向 suc；一条 suc 可以对应多个 fail。

同视频实验因此构造 586 个一对多 pair，每个 fail 有独立 pair ID。三视角内容 hash 用于确认 fail 和 suc 是同一轨迹。

`ranking_data.jsonl` 另列 30 条成功轨迹，使用单独的 grounding 与 ranking
artifact。最终审计发现这 30 个 ID 全部也在 336 条 attention cohort 中；
所以它是独立的输入清单和运行链路，但不是与评测 cohort 样本互斥的发现集。
当前结果不能声称完全消除了 head-selection leakage。若要做严格的样本外验证，
应重新构造按 `video_sha256` 与评测 cohort 互斥的 ranking 数据，并使用新输出目录重跑。

## 三视角与 token span

GRM 和 Qwen forward 的八个槽固定为：

1. reference_start = front first
2. reference_end = blank
3. before_cam_high = front first
4. before_cam_left_wrist = left first
5. before_cam_right_wrist = right first
6. after_cam_high = front last
7. after_cam_left_wrist = left last
8. after_cam_right_wrist = right last

processor 运行后，代码根据 image_grid_thw 和连续 image-token spans 再次绑定这八个语义槽。默认干预目标是 after_cam_high；若它没有对应 front terminal image，运行会报错。

Qwen/RoboReward native video 会校验 video_grid_thw、temporal spans 和 processor 返回的 source frame indices，并要求最后一个 temporal span 包含视频末帧。诊断中记录 target_source_frame_indices，防止把末帧 bbox 施加到早期时间片。

## 空间关系 grounding

以下关系先由确定性 parser 作为完整语义单元识别，不交给通用介词 splitter：

- to the left of -> left_of
- to the right of -> right_of
- closest to -> closest_to
- farthest from -> farthest_from

SAM3 可以开放词表检测 cup、yellow block、purple cup 等实体，但不能可靠地单独推理“离 A 最近的 cup”。当前流程为：

1. 首帧分别检测目标类别和 reference object。
2. 去重候选 bbox，排除与 reference 高 IoU 的同一实例。
3. left/right 用 bbox 中心 x 坐标；closest/farthest 用中心欧氏距离。
4. 选中 bbox 作为归一化 xywh box prompt 输入 SAM3。
5. 后续帧只 tracking 该 obj_id，不再重新按关系词定位。

所以 SAM3 单独不能可靠完成关系定位；SAM3 实体提议加显式几何消歧可以完成，但仍需要人工审核。

## Tracking 与审核

每个 instruction 使用 example-ID hash 作为 artifact key，因而同视频 suc/fail 和一条 suc 的多个 fail 不会覆盖产物。每个样本产生：

- track.json：逐帧 index、bbox、score、obj_id。
- tracking_preview.mp4：全视频 bbox overlay。
- tracking_contact_sheet.jpg：六个时间点的时序图。
- 首末帧 mask；中间帧 full-resolution mask 不保留，避免内存膨胀。

Web reviewer 展示首帧、末帧、contact sheet 和可播放视频。审核媒体路径必须位于 grounding run directory 内。

## 跨模型 attention intervention

GRM、Qwen 和 RoboReward 共用同一个 negative scope 解析函数：

| negative_scope | 正向 bias | 负向 bias |
|---|---|---|
| other_spans | 选中 bbox tokens | 其它 image/time spans；默认 |
| all_visual | 选中 bbox tokens | 除选中 tokens 外所有视觉 tokens |
| target_span | 选中 bbox tokens | 目标 span 内其余 tokens |
| none | 选中 bbox tokens | 无 |

other_spans 对应“目标帧 bbox 加正 bias、其它所有图片或时间片加负 bias”；目标帧内 bbox 外 token 保持零 bias。

本轮三模型统一 `steering_query_scope=last_prompt`、top-k 8/32/64 和 bias 6。
last-frame/GRM 使用 `negative_scope=target_span`，native all-frame 使用
`negative_scope=all_visual`。三者都是 additive pre-softmax bias；Qwen 和
RoboReward 直接共用 Qwen runtime，但不同 processor 的视觉网格仍不相同。

## 解释边界

- raw baseline 覆盖完整 755 条，不受 grounding coverage 影响。
- 本轮按实验计划跳过人工审核，attention 结论只适用于 SAM3 自动 grounding
  双端点成功的 exploratory cohort，不能表述成 audited/formal causal 结果。
- ranking 使用单独的 30 条清单和 grounding 产物，但当前 30 条全部与最终
  attention cohort 重叠；它满足本轮 `exp_plan.md` 指定的数据源，却不是严格样本外发现集。
- fail 的错误目标可能仍在场景中，也可能不存在；后者可 grounding 失败，但必须保留在 raw baseline。
- 遮挡、出画和错误首帧 bbox 仍可能使 tracking 失败，必须查看 tracking 时序产物。
- negative scope、query scope、top-k 或 bias 改变时必须使用新的 output directory。

## 已完成验证

- 真实数据 755 = 169 suc + 586 fail。
- reward、三视角、一对多 pair。
- 四类关系 parser 与几何选择。
- 四种 negative scope 的精确 token 集。
- reviewer 媒体路径边界。
- mock SAM3 box prompt、逐帧输出和 session close。
- mydata_bench 不再导入旧 rewardbench 包。

## 2026-08-06 baseline 实际结果

6 个 baseline 均完成 755/755 条推理，状态全部为 `ok`，并生成
`metrics.json`、`metrics.md` 和 `completion.json`。

| 实验 | exact accuracy | MAE | fail/reward-1 预测为 1 |
|---|---:|---:|---:|
| RoboReward text→video | 0.1576 | 2.2715 | 0.0000 |
| RoboReward video→text | 0.4861 | 1.1258 | 0.4505 |
| Qwen text→video | 0.2291 | 2.1801 | 0.0444 |
| Qwen video→text | 0.3007 | 1.9470 | 0.1587 |
| GRM forward | 0.0053 | 1.9921 | 0.0000 |
| GRM incremental | 0.5801 | 0.9881 | 0.7474 |

逐条 `input_diagnostics.content_order` 证明四个 native-video 实验实际采用了
配置指定的顺序；GRM 逐条记录也分别标记 `forward` 和 `incremental`。
当前结果显示输入顺序和 GRM 评测模式都会显著改变反事实 fail 的判别，
因此这两项不能在不同模型之间被当成等价 protocol。

## 2026-08-06 grounding、ranking 与 completion 审计

最终 grounding 合并文件有 2495 行；按 `(example_id, first/last)` 取最新记录后
是 1510 个端点：680 `ok`、696 `no_detection`、134 `invalid`。共有 336 条样本
首末端点都为 `ok`，冻结到 `results/mydata_bench/cohorts/auto_grounded/`。
逐条检查对应 `track.json` 的 `terminal_frame_index`、终帧 bbox 和最新 last 记录，
terminal error 为 0。该 336 条 cohort 未经人工审核，仍只属于 exploratory 集合。

ranking grounding 的 30 条样本、60 个端点全部 `ok`。8 个实验分别生成自己的
ranking artifact，每项都覆盖 30 条输入和 1088 个 eligible heads（36 层 × 32 heads，
跳过前 2 层）。四个 native processor alignment manifest 均为 `all_valid=true`；
四个 GRM ranking 均记录 `n_discovery_samples=30`。

| 实验组 | cohort | 每条条件数 | 当前指纹下的 ok 行数 |
|---|---:|---:|---:|
| RoboReward last/all frames | 各 336 | 10 | 各 3360 |
| Qwen last/all frames | 各 336 | 10 | 各 3360 |
| GRM forward after/before+after | 各 336 | 13 | 各 4368 |
| GRM incremental after/before+after | 各 336 | 13 | 各 4368 |

每条验收记录的 `grounding_fingerprint` 和 `ranking_fingerprint` 都与当前输入链一致。
所有非 baseline 条件均满足：hook active、`query_scope=last_prompt`、正负 token 非空、
`selected_negative_disjoint=true`、只在 prefill 的最后 prompt query row 施加 bias，
decode/new text key 保持零 bias。native last-frame 的目标 span 均为 terminal `video_t3`；
all-frame 的 `video_t0...video_t3` 均使用与各 span 代表帧最近的 tracking bbox。
GRM after 与 before+after 分别只绑定 `after_cam_high` 和
`before_cam_high + after_cam_high`。

wrong-target 控制优先在每个选中 span 内构造等大小远端区域。RoboReward all-frame
有 2/336、Qwen all-frame 有 5/336 因某个粗网格平面无法构造等大小远端区域，
显式退化为 `other_visual_tokens_fallback`；其 cardinality 与互斥性仍通过，但这些
控制不再保证逐 span 时间匹配，解释 spatial-specificity 时需要保留这一限制。

## Attention steering 最终效果

native 模型的主效应为 candidate-target 相对 baseline 的平均 progress 变化；
CI 是按 source-video cluster、task 分层的 bootstrap，`p` 是双侧 cluster sign-flip。
`ΔMAE<0` 表示误差下降，`ΔAcc>0` 表示 exact accuracy 上升。

| 实验 | top-k | Δprogress [95% CI] | p | ΔMAE | ΔAcc |
|---|---:|---:|---:|---:|---:|
| RoboReward last | 8 | +0.000000 [-0.008000, +0.016000] | 1.0000 | -0.011905 | +0.002976 |
| RoboReward last | 32 | -0.014881 [-0.032000, +0.010667] | 0.7442 | -0.026786 | +0.002976 |
| RoboReward last | 64 | -0.005952 [-0.024000, +0.016000] | 0.8800 | -0.023810 | +0.005952 |
| RoboReward all | 8 | -0.008929 [-0.032000, +0.016000] | 1.0000 | -0.014881 | +0.002976 |
| RoboReward all | 32 | -0.020833 [-0.045333, +0.006000] | 0.3086 | -0.026786 | +0.002976 |
| RoboReward all | 64 | -0.002976 [-0.030667, +0.016000] | 0.8066 | -0.008929 | +0.002976 |
| Qwen last | 8 | -0.011905 [-0.057333, +0.021333] | 0.5174 | -0.017857 | +0.005952 |
| Qwen last | 32 | -0.008929 [-0.056000, +0.024000] | 0.6394 | -0.020833 | +0.002976 |
| Qwen last | 64 | -0.008929 [-0.042667, +0.026667] | 0.7610 | -0.026786 | +0.002976 |
| Qwen all | 8 | -0.017857 [-0.066000, +0.006000] | 0.2245 | -0.011905 | -0.002976 |
| Qwen all | 32 | -0.026786 [-0.076000, -0.001333] | 0.1229 | -0.014881 | +0.000000 |
| Qwen all | 64 | -0.017857 [-0.049333, +0.005333] | 0.2263 | -0.011905 | +0.000000 |

native 的 12 个预设 sign-flip 检验均 `p>0.05`。Qwen all/top-32 的 bootstrap
区间不跨 0，但 sign-flip `p=0.1229`，不能按预设检验宣称显著。输出为 1–5 离散分数，
所以大量样本不变；所有 Δaccuracy 的绝对值均不超过 0.006。

GRM 的 `target_shift` 是 target steering − baseline；`spatial_specificity` 是
target − wrong-region；`head_specificity` 是 target-head − low-rank-head。

| 实验 | top-k | target shift [95% CI] | spatial specificity [95% CI] | head specificity [95% CI] |
|---|---:|---:|---:|---:|
| GRM forward after | 8 | +0.000231 [-0.001193, +0.001689] | -0.000173 [-0.001275, +0.000997] | -0.000127 [-0.001068, +0.000888] |
| GRM forward after | 32 | -0.000908 [-0.002283, +0.000302] | -0.001634 [-0.003180, -0.000219] | -0.001237 [-0.002793, +0.000108] |
| GRM forward after | 64 | +0.000826 [-0.001733, +0.004639] | +0.000805 [-0.001760, +0.004606] | +0.000043 [-0.002713, +0.003949] |
| GRM forward before+after | 8 | +0.000140 [-0.000844, +0.001242] | -0.001521 [-0.002862, -0.000327] | +0.000504 [-0.000743, +0.001925] |
| GRM forward before+after | 32 | -0.000841 [-0.002246, +0.000573] | -0.001429 [-0.002811, -0.000069] | -0.001207 [-0.002693, +0.000179] |
| GRM forward before+after | 64 | -0.001568 [-0.002925, -0.000461] | -0.003027 [-0.004769, -0.001583] | -0.001169 [-0.002367, -0.000079] |
| GRM incremental after | 8 | -0.001158 [-0.005781, +0.003409] | +0.001376 [-0.006659, +0.012319] | +0.003491 [-0.003138, +0.013459] |
| GRM incremental after | 32 | -0.003183 [-0.013426, +0.003882] | +0.001431 [-0.003684, +0.006820] | -0.001899 [-0.013243, +0.007133] |
| GRM incremental after | 64 | -0.003631 [-0.009873, +0.001243] | -0.002695 [-0.010481, +0.004795] | -0.003784 [-0.011127, +0.002527] |
| GRM incremental before+after | 8 | -0.002799 [-0.010144, +0.003451] | -0.001983 [-0.008956, +0.004013] | -0.003376 [-0.009742, +0.001602] |
| GRM incremental before+after | 32 | +0.002615 [-0.002919, +0.008082] | +0.006416 [-0.001173, +0.016432] | +0.001713 [-0.002416, +0.006627] |
| GRM incremental before+after | 64 | +0.000542 [-0.006549, +0.007506] | -0.001399 [-0.009434, +0.006649] | +0.002858 [-0.003505, +0.010125] |

预设 top-8 检验中，四个 GRM 的 `target_shift` 均不显著。只有 GRM forward
before+after 的 top-8 spatial specificity 在 Holm 校正后为 `p=0.0422`，方向为负；
它表示 target 区域不优于 wrong-region，不支持预期的 target-specific steering。
部分 top-32/64 exploratory CI 不跨 0，但没有为所有 top-k 提供预注册式多重校正检验，
不能单独提升为确认性结论。四个 GRM 输出的
`exploratory_target_head_specific_pattern` 和
`target_head_specific_causal_effect_supported` 均为 `false`。

综合而言，bias=6、`last_prompt`、top-8/32/64 下 attention steering 对当前 336 条
自动 grounding cohort 的平均作用很小，没有稳定提升反事实区分能力。这个结论还受
未人工审核 grounding、ranking/评测样本重叠和少量 wrong-control fallback 三项限制。
## 2026-08-10 v2 最终实验审计

> 本节对应 `data/mydata_v2/new`、`mydata_bench/configs/v2/` 和
> `results/mydata_bench/experiments_v2/`。前文的 755/336/30、`last_prompt` 以及
> `results/mydata_bench/experiments/` 均为 v1 历史口径，不能与本节数值混用。

v2 完整集为 1213 条（407 suc、806 fail）。自动 grounding 后的 attention cohort
为 846 条（268 suc、578 fail），仍未经过人工审核。ranking 来源清单有 36 条，
grounding 后实际可用 34 条；ranking 和 steering 已改为自动使用当前可用子集，36
只作为来源清单的数量防错，不要求 36/36。ranking 排除前 8 层，在 28 层 × 32 heads
= 896 个 eligible heads 中按 raw attention mass 排名。v2 steering 使用
`steering_query_scope=all`、bias=6、top-8/32/64；所以它专门回答“把 key-column bias
广播到所有 query 行”后的效果，不能与 v1 `last_prompt` 结果直接当作同一 intervention。

`exp_plan.md` 的最终矩阵是 6 个 baseline 和 10 个 attention，共 16 项：

- baseline_01--06 每项 1213 条，全部 `status=ok`，均有 `metrics.json`。
- native attention_06--09、14--15 每项 846 × 10 = 8460 条，全部 `status=ok`，
  均有 `steering_metrics.json`。
- GRM attention_10--13 每项 846 × 13 = 10998 条，全部 `status=ok`，
  均有 `attention_metrics.json`。
- 16 个实验目录均有 `exp_record.md`。生成脚本按 append-only 主键取最后一条记录，
  并与正式 metrics 交叉校验；每份记录包含 MAE、总体/suc/fail/task 准确率、预测
  label 分布和同视频 suc−fail pairwise 分档。GRM 同时记录 0.125/0.875 与
  0.2/0.8 两套阈值。

### v2 baseline 结果

| 实验 | MAE/离散 MAE | 总体准确率 |
|---|---:|---:|
| RoboReward text→video | 2.1187 | 23.08% |
| RoboReward video→text | 1.1072 | 60.35% |
| Qwen text→video | 1.6966 | 31.49% |
| Qwen video→text | 1.6059 | 35.28% |
| GRM forward | 1.9802 | 5.28% / 10.47% |
| GRM incremental | 1.3899 | 40.81% / 54.74% |

GRM 准确率两列依次对应 0.125/0.875 和 0.2/0.8 阈值。离散模型中，
RoboReward 的 video→text 明显优于 text→video；GRM incremental 也明显优于
forward，说明输入协议仍是影响反事实判别的主要变量之一。

### v2 all-query attention 结果

native 模型下表给出同一 846 条 cohort 上 baseline 与 target steering 的 MAE：

| 实验 | baseline | top-8 | top-32 | top-64 |
|---|---:|---:|---:|---:|
| RoboReward text→video，last frame | 2.1206 | 1.5213 | 1.3487 | 1.5071 |
| RoboReward text→video，all frames | 2.1206 | 1.4574 | 1.2092 | 1.3168 |
| Qwen text→video，last frame | 1.4586 | 1.2849 | 1.1915 | 1.0319 |
| Qwen text→video，all frames | 1.4586 | 1.1927 | 1.3499 | 1.1608 |
| RoboReward video→text，last frame | 0.8333 | 0.8617 | 0.8097 | 1.0686 |
| RoboReward video→text，all frames | 0.8333 | 0.7920 | 0.8440 | 1.3570 |

GRM 的 target shift 定义为 `target steering progress - baseline progress`：

| 实验 | top-8 | top-32 | top-64 |
|---|---:|---:|---:|
| forward，after | -0.188157 | -0.253421 | -0.292904 |
| forward，before+after | -0.269017 | -0.217402 | -0.414103 |
| incremental，after | -0.397309 | -0.626759 | -0.708266 |
| incremental，before+after | -0.188024 | -0.093476 | -0.488743 |

### 2026-08-10：GRM incremental 口径修正

复核 `examples/inference.py` 和 `test_my_data_suc.py` 后确认，既有 v2 incremental
baseline/attention 只输入了 `terminal-20 → terminal`，并把该局部 hop 的 score 直接当作
整段 progress；官方定义实际是 `0→20, 20→40, ..., last_sample→terminal`，随后按剩余
进度/既有进度递推累计。因此旧 `baseline_06`、`attention_12/13` 只可解释为 local-hop
诊断，不能再与 forward 作 episode-progress 对比。

修正实现共享 `official_incremental_indices` 和 `accumulate_incremental_progress`。raw
baseline 对所有相邻 hop 做三视角推理；attention ranking 对所有 hop 的 mass 取均值；
attention steering 则在每个 hop 读取对应 tracking bbox、重新映射 before/after
img-token key 列，并对每个 intervention condition 独立累计。新结果写入带 `_official`
的三个目录，保留旧结果不覆盖。官方递推的未裁剪值单独留档，统计用 `progress` 在最终
一步裁剪到 `[0,1]`。

四个 GRM attention 实验的 hook audit 均记录 `bbox_mass_increase_rate=1.0`，证明
所选 bbox key columns 的 attention mass 在 intervention 后按实现预期增加；这只验证
干预机制生效，不等价于 reward 判别改善。native 结果也显示 all-query intervention
的大小和方向依赖模型、输入顺序、frame scope 与 top-k，不能把三个模型的 steering
当作效果等价的操作。

### ranking 重合度与解释边界

`ranking_overlap.md/json` 列出 10 个 attention 实验各自的 top-8，并统计 top-8/32/64
交集与 Jaccard。代表性结果为：RoboReward 06 vs 07 共享 7/29/54，Qwen 08 vs 09
共享 7/29/53，RoboReward 14 vs 15 共享 5/23/44，GRM 10 vs 11 共享 7/28/58，
GRM 12 vs 13 共享 6/29/57；RoboReward 06 vs Qwen 08 共享 5/22/47。协议变化后
top heads 高度相关但并不相同，因此每个配置使用自己的 ranking artifact 是必要的。

本轮 16 项运行和统计报告已经完成，但结论的 formal gate 仍是
`exploratory_unaudited_auto_grounding`。未经人工审核的 bbox/tracking、34 条 ranking
子集的代表性，以及多组 top-k/scope 比较都会限制因果解释。可以报告当前 cohort 上的
描述性差异和 intervention diagnostics，不能表述成已经人工审核或正式确认的普遍因果效应。
