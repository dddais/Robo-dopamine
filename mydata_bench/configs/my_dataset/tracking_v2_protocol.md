# LJX/LFZ 首帧实例锁定与全视频 tracking v2 人工审核协议

## 1. 结论边界

本轮是看过旧自动 grounding/steering 结果之后启动的
human-reviewed exploratory robustness rerun，不能表述为 confirmatory、formal
或预注册验证。旧的 terminal-only 自动框不能继续充当人工真值；本轮只接受：

1. 首帧上由 SAM3 image proposer 给出的候选；
2. 审核者确认的首帧实例；
3. official SAM3 的 SAM2-style instance tracker 从该视觉框锁定并连续传播出的
   同一 obj_id；
4. 经严格 audit 通过的 terminal bbox。

tracking、审核和 attention manifest 构建阶段不得读取 reward/scoring labels。

## 2. 冻结输入与独立输出

- tracking 配置：grounding_tracking_ljx_lfz_cf_v2.yaml；
- 自动 run-dir：artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_v2/；
- 人工审核目录：artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_reviewed_v2/；
- reviewed attention：artifacts/my_dataset/ljx_lfz_cf_v1/attention_tracking_reviewed_v2/；
- reviewed ranking cohort：
  artifacts/my_dataset/ljx_lfz_cf_v1/ranking_cohort_tracking_reviewed_v2/；
- reviewed matrix：
  outputs/my_dataset/ljx_lfz_cf_v1/reviewed_tracking_matrix_v2/。

禁止回写或覆盖 grounding_auto_v2_*、grounding_reviewed_v1、
attention_reviewed_v1、ranking_cohort_reviewed_v1 和 reviewed_matrix_v1。

requests.jsonl、tracks.jsonl、manifest.json 的 SHA 只有实际运行后才能得到。
在它们生成且审核完成之前，不得在下游 YAML/文档中填写猜测 SHA。下游配置采用
两阶段冻结：先生成并 audit，再把 audit 报告中的真实 64 位 SHA 回填到
tracking-reviewed attention 配置，最后才允许 attention-prepare。

### RoboReward 两种媒体顺序共用 bbox 的硬门

四个 reviewed matrix 同时包含 RoboReward `text_then_video` 与
`video_then_text`。两者只有在 processor 的几何/采帧契约逐例完全一致时，才能共用
同一套首帧传播和 terminal bbox。tracking 配置必须提供
`roboreward_content_order_runs`，且键和值固定为：

- `text_then_video`：`roboreward_8b_native_front`；
- `video_then_text`：`roboreward_8b_model_card_native_front`。

`ground-track-prepare` 必须在读取任何标签前，对全部共同 example ID 核对两条 run 的
成功记录与内容顺序，并比较 source video 身份、processor sampled frame indices、
`video_grid_thw`、decoded frame count、宽高、fps 和 terminal frame 绑定。任一缺失、
重复、内容顺序错误或逐例不一致都必须 fail-closed；不得仍用一条 run 的 terminal bbox
代表另一条。若契约不一致，应为两种内容顺序分别建立版本化 tracking artifact。
配置落盘前的只读核对覆盖 755 条，frames/grid/video mismatch 均为 0；这只是预检证据，
不能替代 prepare 的运行时门禁，也不表示 tracking 已经执行。

## 3. 首帧 proposal 规则

- target_phrase 与 reference_object 必须分开查询；destination 不查询；
- target options 只能来自 target query，reference candidate 永远不能混入；
- simple/object-identity/color 使用最高分合法 target 作为 algorithmic default；
- ordinal、left/right、closest/farthest 必须在首帧用 bbox 中心几何解析；
- reference 缺失或多实例歧义时不得把 top-score proxy 标成 algorithmic default；
  页面可把 options[0] 绿色显示为审核预选，但提交仍是人工 select_alternative；
- 有 algorithmic default 时保留 default 加最多两个 alternatives；无 default
  时保留绿色审核预选加最多两个其他 target，即最多三个 options；
- relation task 中，与唯一 reference 的 IoU 达
  reference_exclusion_iou=0.50 的 target detection 必须排除，并在 proposal
  中记录排除原因；
- 所有候选先做 query-local NMS；非法、越界、非有限 bbox/score 或空 mask
  失败闭合。

## 4. 全视频传播规则

- 必须用 `build_sam3_video_model` 得到官方模型，按官方
  `sam3_for_sam2_video_task_example.ipynb` 取 `model.tracker`，并绑定
  `model.detector.backbone`；不得使用 dense semantic
  `handle_request(add_prompt)` 代替单实例追踪；
- tracker 只接收首帧 normalized xyxy 视觉框，通过
  `init_state → add_new_points_or_box → propagate_in_video` 传播；不能
  text-only tracking，不能逐帧重新检测或重新选最高分实例；
- 首帧框固定分配一个 client obj_id，此后每帧只接受完全相同的 ID；anchor mask
  必须非空并与 proposal bbox 达到配置的 IoU 门槛；
- 帧重复、越界、缺帧、locked ID 改变、输出 tuple/mask shape 异常或 terminal
  不可见均记为 invalid；
- tracker 保留 obj_id 但 mask 暂时为空表示遮挡/离开视野：普通帧和 sampled
  keyframe 允许记录为 `visible=false, bbox=null`，同时必须冻结全零 mask；
  后续重新出现时仍使用同一 obj_id。三模型 terminal 必须
  `visible=true`，不得沿用旧框或伪造 bbox；
- 即使发生异常也必须关闭传播 generator 并释放 caller-owned inference state；
  run 结束必须退出官方 tracker 的 bf16 context 并清理模型；
- RR/Qwen processor sampled frames 的 union、首帧和三模型 terminal 是可审核
  keyframes；连续性检查覆盖从 0 到 terminal 的每一帧；
- manual bbox 也必须从首帧重新传播。禁止把首帧框直接复制为 terminal 框。

缓存键冻结为 source video SHA、首帧 index、首帧 image SHA、anchor bbox/mask SHA
和 tracker fingerprint；缓存只复用完整、SHA 校验通过的成功 track。

## 5. Provenance 要求

Image proposer provenance 必须包含 Transformers 版本、阈值/NMS、grounder 源码
SHA、模型 config SHA，以及实际 safetensors/sharded weight 文件的逐文件 SHA。
Video tracker provenance 必须包含 checkpoint SHA、official source path、sam3/
下全部普通 Python 文件与 BPE asset 的 source-tree fingerprint。上述哈希均为
run 级计算一次后复制，不能每个 example 重算 3.4GB 权重。

## 6. 两轮人工审核

页面只允许三种状态，且状态与来源严格对应：

- eligible：decision.source 只能是 accept_default、select_alternative 或
  accept_manual_track；三者都必须绑定完整、有效且已冻结的 terminal track；
- needs_retrack：decision.source 固定为 manual_first_bbox，只保存冻结首帧上的
  bbox，models 必须为空；它不是 eligible，也不代表 terminal 已确认；
- skipped：disposition.code 只能是 reviewer_skip，models 必须为空；禁止 reason、
  note、wrong_region 或其他自由文本/旧 disposition code。

同一审核目录只能使用一个固定、非空 reviewer ID。审核时不得打开 labels、旧
steering 分数或 head ranking。首轮应先审完全部样本；所有 needs_retrack 的最新
决定统一物化到 grounding_tracking_reviewed_v2/manual_anchors.jsonl。不得边审核
边零散传播并批准，以免 manual artifact 的整文件身份不断变化。

manual retrack 必须批量完成并处理完失败重试，然后把最终 manual_tracks.jsonl 写在
自动 run-dir 下。第二次以相同 reviewer/output-dir 重启 Web Store 后，所有
needs_retrack 会重新进入 pending；审核者必须查看 keyframes 和三模型 terminal，
再选择 accept_manual_track 或 reviewer_skip。若某条人工传播缺失/无效，应在批准
任何 manual track 之前完成重画和整批重跑。

manual_tracks.jsonl 采用整文件 SHA 冻结，而不是逐行 SHA：第一条
accept_manual_track 保存后，manual artifact 即冻结，Web 与服务端都禁止再新增或
修改 needs_retrack；同一文件也不得 append、重试或重写。后续每条
accept_manual_track 都必须绑定完全相同的整文件 SHA。若冻结后发现必须重画或重跑，
应新建版本化的 tracking/review 目录重新审核，不能原地修补。若最终没有任何
accept_manual_track，即使磁盘上存在 manual_tracks.jsonl，audit 中的
manual_tracking_artifact_sha256 也必须是 null；只要接受过一条，则必须是该冻结
文件的真实 SHA。

## 7. Audit、attention 与矩阵硬门

- 最终 reviews.jsonl 必须覆盖全部 requests，不得残留 needs_retrack；
- ground-audit 必须同时绑定自动 tracking artifact；若任何审核选择了 manual
  track，还必须绑定 manual tracking artifact；
- audit 必须验证 request/track/manifest/review/manual SHA、fingerprint、reviewer、
  selected candidate、terminal image/index/SHA 和 bbox；
- attention 只接收 audit 中 eligible 的实例；成组评估只使用组内全部样本均
  eligible 的 complete groups；
- 固定 S20 source 不允许静默用第 21 名以后替换；若其中有真正 skipped，
  当前 cohort 协议应硬失败并另立版本；
- 四个 N×K matrix 仍冻结 excess-mass、skip8、N={5,10,20}、K={8,32,64}、
  bias=6、all-query；只改变经审核的 tracked target bbox；
- 与旧结果比较时必须把旧 records 限制到完全相同 complete-group IDs，并执行
  shared baseline 逐例 parity gate。

本轮未执行 wrong-target、low-rank、layer-matched-random 等 controls，不得声称
这些 control 已由 tracking v2 覆盖。
