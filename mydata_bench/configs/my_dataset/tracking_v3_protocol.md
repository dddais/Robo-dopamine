# LJX/LFZ tracking-reviewed attention v3 协议

## 1. 证据版本与结论边界

v2 的代码、requests、tracks、人工审核和实验输出是不可原地修改的历史证据。
v3 是针对实例身份、全程漂移和原生视频 temporal-patch token 对齐问题的更正重跑，
所有产物必须写入独立的 `*_v3` 目录。v2 的人工审核不能转移为 v3 审核。

本轮仍属于 human-reviewed exploratory robustness experiment，不是新的预注册验证。
grounding、tracking、attention ranking 和 steering 阶段不得读取 reward/scoring
labels；labels 只允许在全部推理结果冻结后用于 scoring。

## 2. 首帧实例确认与全视频 tracking

- SAM3 image proposer 只负责首帧 target/reference 候选，不负责决定困难关系的实例。
- simple、object identity、attribute/color 可保留算法默认和有限备选。
- ordinal、left/right、closest/farthest，以及
  `requires_instance_review=true` 的样本，必须设置
  `pretrack_identity_selection=true`。这些样本在人工确认首帧具体实例之前不生成
  自动 candidate tracks。
- 审核者可以复制一个 target candidate bbox 或在首帧手画 bbox；该框只生成
  `needs_retrack`，不能直接成为 eligible。
- manual anchor 必须用 official SAM3 instance tracker 从 frame 0 重新传播；禁止将
  首帧 bbox 复制到 terminal，禁止逐帧重新检测或重新选择同类实例。
- tracker 输出必须覆盖完整的 `0..terminal` 帧序列，并保持同一 locked obj_id。
  obj_id 连续只是结构条件，不是语义身份正确的充分条件。

每条轨迹的审核帧包含：

1. 两个原生视频 processor 的全部 sampled source frames；
2. GRM 端点；
3. 首帧和终帧；
4. 默认 16 个确定性的全视频均匀采样帧；
5. 漂移诊断触发的帧。

短视频保留全部帧。低 tracker score、不可见、相邻面积突变、相对 anchor 面积异常和
中心跳跃只触发醒目的人工复核警告，不自动替代人的语义判断。正确首帧但中途换实例、
漂移到机械臂或目标消失后落到相邻同类物体，都必须标为 skipped 或重新画框、retrack。

## 3. 两阶段人工审核

第一阶段：

- 查看首帧 target/reference 与全部自动 tracks；
- 关系/顺序样本必须人工确认具体实例并保存 manual anchor；
- 自动 track 只有在首帧身份和全程轨迹都正确时才可 eligible；
- 其余样本记录为 needs_retrack 或 skipped。

第二阶段：

- 将第一阶段全部 needs_retrack 批量写入
  `grounding_tracking_reviewed_v3/manual_anchors.jsonl`；
- 统一生成 `grounding_tracking_v3/manual_tracks.jsonl`；
- 用相同 reviewer ID 和审核目录重启页面；
- 逐条查看均匀帧、processor 帧、警告帧和三模型 terminal，之后才能
  `accept_manual_track`。

一旦任何 manual track 被批准，manual_tracks.jsonl 的整文件 SHA 即冻结，禁止继续
append、重试或重画。若发现错误，建立新的版本化 tracking/review 目录，不原地修补。

## 4. 原生 Qwen/RoboReward 视频 token 语义

Qwen3-VL video processor 以 `temporal_patch_size` 个连续 sampled frames 组成一个
temporal patch；末尾不足时只重复最后一个 sampled frame。当前官方 processor 为每个
temporal patch 插入 timestamp/vision delimiters，当前模型的位置编码也按这些独立 run
拆分 `video_grid_thw`。因此兼容代码只接受一个 patch 一个 token run；多 patch 的单
连续总 run 即使 token 总数相同也不兼容当前模型，必须 fail-closed。任何 run 数、每
run token 数、`video_grid_thw` 或 temporal patch 不一致同样失败闭合。

Qwen/RoboReward 的目标不是“孤立的最后一帧”。目标 span 是最后一个 temporal patch；
目标 token 是该 patch 中所有可见 source frames 的 tracked bbox 映射到合并后视觉
网格的 token 并集。manifest 必须冻结：

- processor sampled frame indices；
- `video_grid_thw`；
- `processor_temporal_patch_size`；
- 每个 sampled frame 的 image SHA、visible 状态和 tracked bbox；
- `target_token_grounding_scope=terminal_temporal_patch_tracked_bbox_union`。

GRM 的 target 仍是 `after_cam_high` 单图内的 terminal bbox。

## 5. Ranking、steering 与 visual scope

主矩阵固定：

- ranking score：excess mass；
- skip early layers：8；
- ranking prefix：N={5,10,20}；
- steering heads：K={8,32,64}；
- additive pre-softmax bias：6；
- query scope：`all`；
- 每个样本 28 个 paired conditions：1 个 baseline、9 个
  candidate-target、9 个 candidate-wrong-region、9 个 low-rank-target。

`candidate-wrong-region` 必须在同一 target span 内使用与 target 完全相同的 token
footprint/count，且与 target 不相交；不存在合法平移时该样本失败闭合。
`low-rank-target` 必须与 candidate heads 不重叠，并严格匹配 candidate 的逐层 head
数量。

可选 visual scope：

- `target_slot_only`：GRM 只抑制 after_cam_high 内非目标 token；原生视频只抑制最后
  temporal patch 内非目标 token。
- `all_visual`：GRM 抑制全部八个图像槽的非目标 token；原生视频抑制整个视频的
  非目标视觉 token。

attention bias 施加在选中 head 的所有合法 query 行与目标/对照视觉 key 列；原始
causal/padding mask 仍保留。跨模型相同 bias 和 K 不代表相同 intervention dose，
主要结论必须来自各模型内部 paired contrasts，不能直接用模型间 raw shift 排序依赖
强弱。

若运行 `all_visual` 消融，必须同时修改
`intervention_visual_scope`、`variant_id` 和 `output_dir`，不得复用主矩阵目录。
ranking scope 默认保持 `target_slot_only`；若也改变 ranking scope，同样视为新的
实验变体。

## 6. Provenance、resume 与 cache 的最小心智模型

- 同一 fingerprint + 同一 output directory：只 resume 缺失的工作。
- 输入、配置、代码、模型 checkpoint、关键运行库版本或 scope fingerprint 改变：使用新 output
  directory。
- track cache 只用于避免对完全相同的 video+bbox+tracker+quality contract+required
  review frames 重跑 SAM3。
- `from_audit` 只是免去手工复制 SHA；程序仍重新计算配置路径上的真实文件 SHA，并与
  audit 比较。

这些机制不改变数据和模型逻辑，只用于阻止新旧产物在相同 example ID 下被静默混合。

## 7. 进入下游实验的硬门

- reviews 必须覆盖全部 requests，不得残留 needs_retrack；
- review audit 必须 `passed=true`，并绑定 requests、自动 tracks、manifest、
  reviews，以及实际被接受时的 manual_tracks；
- attention manifest 只能接收 eligible；评估只使用完整 eligible groups；
- ranking S20 不得静默用第 21 名以后替代缺失样本；
- RoboReward text/video 两种 content order 只有在逐例 sampled frames、
  `video_grid_thw`、视频身份和 metadata 均一致的冻结 contract 下才能共用 bbox；
- 真实模型运行前应通过静态检查、单元测试和小规模 runtime token/mask trace；
- 代码修复不能追溯性地修复旧结果，v3 的 tracking、审核、attention、ranking 和
  matrix 必须重新执行。
