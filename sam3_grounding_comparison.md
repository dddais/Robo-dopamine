# SAM3 grounding 对比：原仓库离线实验与当前在线 Monitor

核对日期：2026-09-12。原仓库代码版本 `0b3489e`，交付仓库代码版本 `e5e526b`。本文以代码和实际 YAML 为准，结合已有运行产物说明效果；本次没有重跑两套模型的同视频准确率或性能对照实验。

比较入口是原仓库 [mydata_bench/exp_use.md](../../Robo-Dopamine/mydata_bench/exp_use.md)，重点是其中的自动 grounding、tracking 和 attention cohort。当前在线实现见 [sam3_tracking.md](sam3_tracking.md)。这里的 grounding 指“确定任务目标对应的图像区域”；区域正确、始终是同一个实例、有足够多的有效帧、结果足够新，是不同的要求。

## 1. 判断与建议

**原仓库已经使用“首帧 SAM3 检测＋视频跟踪”，当前 Monitor 并不是首次把 tracking 加入这套方法。两者的主要区别是视频后端、首帧目标选择、失跟处理，以及离线全视频与在线稀疏快照的运行条件。**

| 想解决的问题 | 当前判断 | 判断依据与限制 |
| --- | --- | --- |
| 真机上持续给 GRM 提供及时、与图像对应的 bbox | 当前 Monitor 的结构更适合 | 独立采图/跟踪、有限历史、只取最新完整快照、HTTP 和图像哈希校验；不需要先有完整视频 |
| 首帧选择“在某物体左边／离某物体最近的物体” | 原仓库的显式关系选择更完整 | 先检测目标与参照物，再按首帧二维几何筛选；当前 Monitor 没有这一层。前提是解析正确、候选和参照物检测正确 |
| “宁愿没有框，也不主动换目标” | 当前 `locked` 的应用层约束更严格 | 不做文本找回；原仓库指定 ID 缺失时有选其他 ID 的退路。两者都不能保证模型本身不漂移 |
| 遮挡或暂时失跟后尽量恢复 bbox | 在线可选 `hybrid`；离线保留原视频 predictor 作为对照 | `hybrid` 明确支持重检测；原版支持整段传播及末帧缺失后的补提示重跑。哪套对同一实例恢复更准，尚无配对实验证据 |
| 最终 GRM 打分更准确、机器人任务完成率更高 | 目前不能宣布哪套胜出 | 还没有固定视频、首帧目标、采样频率、GRM 和 head profile 的对照；有框率不能代替目标正确率 |

对于目前“两支笔、拿起其中一支”的任务，建议保留在线结构，以 `locked + continuity_iou: 0.0` 作为身份优先的对照，再与 `hybrid` 比较。最值得从原仓库迁移的是**首帧关系解析和几何选实例**，而不是直接把整段视频 predictor 调用搬到实时循环里。迁移尚未实施。

## 2. 原仓库实际上怎么做

实现入口为 [grounding/pipeline.py](../../Robo-Dopamine/mydata_bench/grounding/pipeline.py) 的 `run_grounding()`，SAM3 封装为 [grounding/sam3.py](../../Robo-Dopamine/mydata_bench/grounding/sam3.py) 的 `SAM3Grounder`。

### 2.1 指令解析和首帧检测

1. `InstructionParser` 使用配置中的 Qwen3-4B 指令模型生成目标结构，包含目标短语、类别、属性、参照物和关系；解析失败时有启发式兜底。
2. `build_queries()` 生成检测词。一般目标可同时使用完整短语和类别词，部件目标还可能增加所属物体的查询。每个词单独调用 Transformers `Sam3Model + Sam3Processor`。
3. 图像模型进行完整分割推理，使用 `post_process_instance_segmentation()`，获取候选框、分数和 mask。现有 grounding YAML 使用检测阈值 `0.30`、mask 阈值 `0.50`、`top_n: 20`。
4. 非关系目标从合法候选中取最高分；分数相同才按查询优先级选。它没有当前普通在线检测客户端的“前两名分差小于等于 0.05 就不选框”规则。
5. 关系目标额外检测参照物，调用 `select_relational_candidate()` 选择初始实例。

关系选择的具体逻辑见 [grounding/base.py](../../Robo-Dopamine/mydata_bench/grounding/base.py)：

| 关系 | 实际规则 |
| --- | --- |
| `left_of` / `right_of` | 比较目标与参照物框中心的横坐标；在满足方向的目标中取最高分 |
| `closest_to` / `farthest_from` | 按目标与参照物框中心的二维像素距离取最近／最远 |
| 参照物选择 | 取最高分参照物；排除与其高度重叠的目标候选，防止把参照物自己选作目标 |

这层把“识别候选物体”和“按关系挑选实例”分开，关系含义只在初始场景中求解。它使用相机图像的二维几何，不是机器人坐标系中的三维左右或距离。

### 2.2 官方 SAM3 video predictor 传播

首帧框选定后，原版调用官方 `sam3` 包的 `build_sam3_video_predictor()`，加载 `sam3.pt`。这与首帧使用的 Transformers 图像模型是两条加载路径。

调用顺序是：

1. `start_session(resource_path=video_path)`，提供完整本地视频。
2. 在 anchor 帧添加归一化 `xywh` 框，提示是 `text: "visual"`，而不是再次提交原始 `left pen` 短语。
3. 保存初始输出的 `obj_id`，使用 `propagate_in_video(..., propagation_direction="forward")` 遍历视频。
4. 如果初始实例存在，但没有末帧输出，则在初始帧为同一 `obj_id` 增加一个正点：优先取初始 mask 的质心，缺失时用框中心；随后再向前传播。
5. 保存每个有输出帧的 bbox、分数和 ID，最后关闭会话。

第 4 步是重要差异：**它可以在看到整段视频没有跟到终点后，回到首帧修改提示，再重新处理这段视频。** 这属于离线修正，不能作为实时 Monitor 当时已经拿到的结果。

本环境实际导入的官方包位于 [cap-x 中的 SAM3](../../cap-x/capx/third_party/sam3/sam3/model_builder.py)。其默认视频构建包含 detector、tracker、检测关联、轨迹存活和定期重新加入检测提示的逻辑；配置中有 `hotstart_delay=15`、`recondition_every_nth_frame=16`。完整传播会缓冲输出，补点后还可走部分 tracker 传播。因此，原版不等同于“纯视频 tracker 每帧向前算一次”。这些是本机安装版本的行为，不能假定所有官方版本相同。

### 2.3 保存结果与进入 GRM

原版主要输出 `track.json`、首末帧 grounding JSONL、端点 mask，以及可选的视频框预览。`tracking_preview: false` 只关闭预览，仍执行跟踪。其他相机可以提供 GRM 图像，但这条 grounding 主路径跟踪的是主视频目标，不是在三个视角上做跨相机身份融合。

`freeze_auto_grounded_cohort()` 选择最新记录中**首帧和末帧均为 `status: ok`** 的样本，供后续 attention 实验使用。这个筛选不证明中间每帧都跟到，也不证明框始终属于正确物理实例。

GRM 读取保存的 bbox，将区域映射成视觉 token，再对选定 heads 加 attention bias。官方 incremental 实验优先按各 hop 的视频帧号取框，但该帧无框时会取最近有框帧的 bbox；native video 实验另有时间片 bbox 的 `last / union / intersection` 规则。它们都不能与在线“每次取最新三视角”混为同一种采样协议，具体的跨帧取框影响见第 4.3 节。

原仓库还保留 `tracking: false` 的分支：独立检测首末端点，再用置信度、query 和颜色外观联合选框。这不是 `exp_use.md` 描述的默认 SAM3 tracking 主流程。

## 3. 当前 Monitor 怎么做

### 3.1 目标词与检测路径

当前目标词优先取 UI/请求中的 `target_queries`，其次使用 `task_queries` 映射，最后才由 [grm_runtime/common.py](../grm_runtime/common.py) 的简单英文规则从 instruction 中提取。在线链路没有加载原仓库的 Qwen 解析器，也没有参照物几何选择层。

图像检测仍使用 Transformers `Sam3Model`，但 [sam3_runtime/detector.py](../sam3_runtime/detector.py) 已做以下调整：

- 检测服务常驻，精度由 YAML 显式指定；fast/tracker 配置使用 BF16，另保留 FP32 配置。
- `bbox_only: true` 跳过图像检测模型的 mask decoder，只做 box 后处理，减少不需要的 mask 计算和搬运。
- 对高度重叠的候选做去重，减少同一物体被多个 query 重复返回。它不等于在多个真实物体中解析“左边”。
- 不把 bbox 画进 GRM 输入图。UI 的框是显示叠加，模型输入仍是原始或已配置预处理的图像，bbox 用于 attention token 选择。

**检测的 bbox-only 优化不作用于视频 tracker。** 视频 tracker 仍然预测 mask，并由当前 mask 的包围矩形产生 bbox；删掉这部分 mask 会改变跟踪本身。

### 3.2 流式视频 tracker 与三种使用方式

[sam3_runtime/tracker.py](../sam3_runtime/tracker.py) 使用 Transformers `Sam3TrackerVideoModel + Sam3TrackerVideoProcessor`，从同一个模型目录加载 tracker 权重，不调用原仓库的官方整段视频 predictor。

首次检测框以原图坐标提示 `obj_ids=1`；后续输入逐张新图片和显式帧序号。每个任务、每个启用的相机独立维护状态，保留提示帧和有限近期历史，当前 `memory_frames: 32`。帧号是收到的跟踪更新序号，并不是相机完整视频中的原始帧号。

| 使用方式 | 正常工作 | 缺失或失跟后 |
| --- | --- | --- |
| 不启用 Monitor tracking | 按 GRM 轮次做图像文本检测；相同输入可命中缓存 | 下一轮重新检测；多个高分候选分差不超过 0.05 时返回歧义空框 |
| `locked` | 首帧取最高分候选，之后只传播这个视频状态 | 一次空 mask、低分、更新间隔过长或推理异常后，本任务持续空框；不文本找回 |
| `hybrid` | 视频传播，默认每约 5 秒文本检测一次；周期候选需匹配当前帧的跟踪框 | 失跟后在当前帧重新检测，取最高分候选；无候选则空框并在后续更新重试 |

`hybrid` 周期检测没有匹配候选时保留健康的视频轨迹；不会仅因文本结果不一致而清除它。失跟后的重新绑定不要求匹配旧位置，因此允许选中另一实例。它也没有原版“回到首帧补点、重跑此前视频”的能力。

两种 tracking 模式都禁止旧任务的 continuation 请求在会话过期、关闭或 SAM3 重启后重新创建会话；这些情况需要新任务。

### 3.3 当前配置与旧配置必须分清

本次核对时，[sam3_tracker.yaml](../configs/sam3_tracker.yaml) 已经是：

```yaml
tracking:
  mode: locked
  min_score: 0.1
  continuity_iou: 0.0
  max_gap_s: 2.0
  memory_frames: 32
```

因此**当前这份 YAML 不会再仅因相邻框不重叠而触发 `discontinuous_bbox`**。历史配置是 `continuity_iou: 0.1`；旧 YAML 若省略 `continuity_iou`，类在 `locked` 下仍回退到 `match_iou`，默认 `0.1`。运行进程以启动时读取的配置为准，可通过 SAM3 `/health` 的 `tracking_config` 核对。

不要混淆这些参数：

| 参数 | 实际判断对象 |
| --- | --- |
| 检测 `threshold` | 文本检测候选分数 |
| 跟踪 `min_score` | 视频模型的目标存在分数；当前上游还会根据 `object_score_logits <= 0` 清空 mask，外层阈值再低也无法恢复已清空的 mask |
| `continuity_iou` | 相邻有效视频框的硬拒绝阈值，`0` 关闭 |
| `match_iou` | `hybrid` 周期检测框与**当前帧** tracker 框的匹配阈值 |
| `max_gap_s` | 两次 SAM3 更新间隔，超限释放旧视频状态 |
| Monitor `max_frame_age_s` | GRM 读取时，完整快照从该轮开始请求取图算起的年龄；当前为 `2.0` 秒 |

### 3.4 在线调度与状态影响

[monitor_runtime/tracking.py](../monitor_runtime/tracking.py) 独立于 GRM 运行取图和跟踪，只保留最新完成的一组“三视角＋对应定位”。GRM 固定该组文件用于本轮评分；forward/incremental 与双分支共享这组 AFTER 输入。默认仅对 `after_cam_high` 做定位干预，三视角输入不意味着三个视角都已做身份融合。

这种结构避免排队补算旧帧，但实际进入 tracker 的帧率受相机缓存、取图、SSH/HTTP、编码和推理限制；`poll_interval_s: 0.1` 不保证 10 FPS。离线逐帧视频与在线每秒少量快照的运动幅度不同，不能仅凭在线更容易跟丢就认定 Transformers tracker 更差。

没有 bbox 时，默认 `on_missing_bbox: baseline` 继续 GRM 评分但不加目标 bias；它不能解决画面本身超龄。`Latest tracked frame is stale` 会跳过该轮评分并发布顶层错误。当前 [Simple Loop](../../dualsystem-agentic/src/dualsystem_agentic/simple_loop.py) 一旦轮询读到该错误，就会进入停止/恢复；若新的成功评分在轮询前清除了错误，则这次日志不一定造成状态转移。原离线 grounding 不存在这种机器人状态转移或实时超龄门槛。

## 4. 对目标身份与空间关系的具体影响

### 4.1 原版并不严格保证同一 ID

原版 `_one_track_output()` 的选择顺序是：先找指定 `obj_id`；找不到时，如果还有其他输出，则退回最高 `out_probs` 的候选，而不是返回缺失。

本次直接调用该函数验证：请求 `obj_id=1`，构造只有 `obj_id=9`、分数 `0.99` 的输出，函数返回了 ID 9。这个检查证明代码**允许换 ID**，不代表已统计出真实视频换 ID 的频率。

此外，某一帧没有输出只会被跳过，后续传播仍继续；没有当前 `locked` 的“一次失跟永久空框”策略。因此原版可能有更高的末帧有框率，但其中也可能混有恢复到其他实例的情况。原版 `out_probs` 与当前直接计算的 `object_presence` 也不是可以直接套用同一个 `min_score` 的评分接口。

当前 `locked` 消除了应用层重新挑选的退路，但视频模型仍可能在相似外观、遮挡和重叠时漂移。关闭 `continuity_iou` 可减少正常位移被拒绝，同时也会放过模型的大幅跳框。这是覆盖率与身份约束之间的取舍。

### 4.2 “left pen”与“pen to the left of the cup”不是同一条解析路径

对于 `Pick up the pen to the left of the cup and place it in the box.`：

- 原版可解析出目标 `pen`、参照物 `cup`、关系 `left_of`，先检测两类物体，再在首帧按位置选择。
- 当前 Monitor 若没有显式 `target_queries`，简单解析会在 `to` 处分割，得到 `pen`，丢掉关系。即使显式传入完整关系短语，也只是避免截断；仍由 SAM3 理解整句，没有几何验证。

上述 query 差异已通过两个仓库的解析函数直接验证，未运行 Qwen 或 SAM3。实际 UI 显式传入目标词时，应以请求中的词为准。

对于用户当前的 `left pen` / `leftmost pen`：原版没有专门的“在全部笔中取最左框”算子；它主要依赖目标解析和 SAM3 的短语检测。原版确定性的关系语法也只覆盖特定句式，例如 `pick ... and place ...`，不能保证所有 `and put ...` 的表达走同一逻辑。因此，不能表述为“换回原版就已经解决左边那支笔”。

更合适的后续方案是：在任务初始帧显式解析 `leftmost/rightmost` 或“相对于参照物”的关系，选定实例后沿用视频状态。若允许恢复，应额外研究与最初实例的外观/提示匹配，而不是在变化后的场景上直接重新解释 `left pen`。这些能力当前尚未接入。

### 4.3 有框不一定意味着框来自评分图像的同一帧

原版 `attention_eval/runtime.py` 的 `incremental_steps()` 在某个 hop 帧没有 tracking bbox 时，会取时间上最近的有框帧，但图像仍按原定 hop 帧号提取，并分别记录 `before/after_frame_index` 和 `before/after_bbox_frame_index`。最近有框帧也可能位于请求帧之后。这是离线代码允许的补缺路径，不是每一个样本都会触发。

这种处理能增加可评分的 hop 数量，但快速运动时，跨帧框可能不再覆盖当前图像中的目标。不能仅检查文件存在或 `status: ok`，就断言该 hop 的图像和 bbox 完全同帧；做效果统计时应检查两组帧号是否相等。

当前在线 tracker 返回结果与实际输入图片的 SHA、尺寸、query、session 对齐，GRM 再验证冻结文件；缺失时不借用其他帧的 bbox。它在这个接口约束上更严格，但代价是可能更多地返回空框或因画面超龄而暂停评分。哈希一致只证明输入对应，仍不能证明模型框中了正确物体。

## 5. 现有证据能说明什么

### 5.1 原版的数量是自动可用样本覆盖率，不是跟踪准确率

[exp_use.md](../../Robo-Dopamine/mydata_bench/exp_use.md) 记录：

| 历史实验口径 | 全集 | 自动 grounding cohort | 可计算的覆盖率 |
| --- | ---: | ---: | ---: |
| v1 历史记录 | 755 | 336 | 44.50% |
| v2 记录 | 1213 | 846 | 69.74% |

这些是文档中的历史结果，本次没有重新生成 cohort。两批数据不同，比例不能用来证明算法迭代提升了多少。`status: ok`、`tracking terminal audit error: 0`、下游实验全部完成，都不等于“全程正确实例跟踪率为 100%”。文档明确这些 cohort 未经人工审核；而且它们先排除了 grounding 不可用的样本，不能直接与在线全部任务比较。

### 5.2 在线的历史失跟有应用层误判证据

2026-09-11 排查的最近 10 个在线会话中，9 个失跟，原因均为 `discontinuous_bbox`。对其中一个会话的 54 张已保存评分图做真实 SAM3 回放，笔被拿起时视频模型仍框中手里的笔，分数 `0.97265625`，但相邻框 IoU 为 `0`，会被当时的硬阈值拒绝。

同一回放保留 32 与 256 帧历史，输出框和分数一致。它支持“这次误判来自硬阈值，未发现该序列中历史裁剪造成差异”，不支持“所有视频中两种历史长度效果相同”。输入是 GRM 快照、以首个已发布框初始化，并非运行中完整 tracker 流的精确复现。

新 `hybrid` 另有真实权重离线检查，走通了初始化、周期匹配、插入空白图后的失跟、场景恢复后的重检测。它验证控制流程，不是遮挡/多实例场景的准确率评测。

### 5.3 加速测量没有直接比较官方 predictor 与当前 tracker

已有 A100 80 GB、3 张 1280×720 图片、目标 `pen` 的离线测量：

| 测量路径 | 平均耗时约 | 可得结论 |
| --- | ---: | --- |
| 当前检测实现，FP32，保留 mask forward | 398 ms | 视觉编码占主要时间 |
| 当前检测实现，FP32，跳过 mask forward | 391 ms | 单独跳过 mask head 的收益较小 |
| 当前检测实现，BF16，跳过 mask forward | 92 ms | 该设备、输入和负载下，低精度是主要加速来源 |
| BF16 视频 tracker，40 帧重复回放中除首帧外 | 约 77 ms | 纯视频更新较轻，但还不含在线取图与传输 |

这里第一行仍使用当前检测封装和 box 后处理，**不是原仓库完整分割后处理＋官方 video predictor 的耗时**。原官方视频实现包含 BF16 autocast 路径，也不能把原版整条流程都算作 FP32，再把差距全归因于精度。

同批检测对照中，两条 FP32 路径的框和分数完全一致；BF16 最大坐标差约 4.52 像素、最大分数差约 0.00565。它们只是这几张图的数值差异，不是目标正确率指标或任意场景误差上界。

这些数字不含 SSH、HTTP、GRM、Loop 和浏览器等待。当前 `hybrid` 重检测当轮还需要检测及重新初始化，不能用 77 ms 代表它每一轮的固定耗时。

本工作区的原始证据保存在以下运行产物中，`results/` 不随 Git 同步：

- `results/benchmarks/sam3_20260909.json`
- `results/diagnostics/sam3_tracker_policy_replay_20260911.json`
- `results/diagnostics/sam3_hybrid_smoke_20260911.json`

## 6. 如何比较“哪个更有效”

最小有意义的对照应该分成两组，避免把首帧选错与后续跟丢混在一起。

| 对照 | 固定什么 | 改变什么 | 主要指标 |
| --- | --- | --- | --- |
| 首帧 grounding | 同一批图、目标语义、检测权重与精度 | 完整短语最高分；类别候选＋显式关系几何 | 首次选中正确实例的比例、无框率、耗时 |
| 视频跟踪 | 同一段视频、同一个正确初始框、相同输入帧序列 | 官方 predictor；当前 `locked`；当前 `hybrid` | 正确实例 bbox 覆盖率、框 IoU、换实例次数、连续缺失时长、恢复到原实例的比例 |
| 在线系统 | 同一回放源、GRM、head profile、评分模式和设备负载 | 定位方式与采样调度 | 图像年龄 p50/p95、有效评分率、超龄次数、误停止次数、端到端延时 |

视频跟踪组还应分别使用完整视频帧率和实际在线采样序列。官方的末帧补点重跑要单独统计为离线修正，不能把修正后的早期结果计作当时可用的在线结果。所有方法应同时报告全集结果和共同可用子集，不能仅保留各自成功样本后比较。

若进一步比较 GRM 效果，还需固定 head ranking、输入模式和缺框处理。例如 `exp_use.md` 的 v2 实验排除前 8 层、比较 top-8/32/64；当前 [在线 profile](../assets/steering/grm_8b_profile.json) 是 raw-mean-12 三任务聚合、排除前 2 层，在线默认 top-8。即使两边都使用 bias=6，也不是只换了 SAM3 的对照。

现阶段实际使用上，**保留当前在线框架；身份优先用 `locked`，可接受重选而希望恢复定位时用 `hybrid`；关系复杂时优先补首帧几何选择。** 原仓库的官方 predictor 和补提示传播适合作为离线对照，但现有证据不足以声称它或当前 tracker 在目标跟踪准确率上全面更优。

## 7. 代码核对索引

| 位置 | 本文核对的内容 |
| --- | --- |
| [原 exp_use.md](../../Robo-Dopamine/mydata_bench/exp_use.md) | 第 1、2 节 grounding；第 8 节 v1 结果；第 11 节 v2；第 12 节官方 incremental |
| [原 parser.py](../../Robo-Dopamine/mydata_bench/grounding/parser.py) | Qwen/启发式解析、关系语法、`build_queries()` |
| [原 base.py](../../Robo-Dopamine/mydata_bench/grounding/base.py) | 最高分选择、`select_relational_candidate()`、非 tracking 的端点联合选择 |
| [原 sam3.py](../../Robo-Dopamine/mydata_bench/grounding/sam3.py) | 图像分割、官方视频 predictor、ID 回退、末帧缺失补点 |
| [原 pipeline.py](../../Robo-Dopamine/mydata_bench/grounding/pipeline.py) / [cohorts.py](../../Robo-Dopamine/mydata_bench/cohorts.py) | 主视角传播、端点记录和自动 cohort 规则 |
| [原 attention runtime.py](../../Robo-Dopamine/mydata_bench/attention_eval/runtime.py) | 保存框与图像帧、GRM token 的对应 |
| [本机官方 model_builder.py](../../cap-x/capx/third_party/sam3/sam3/model_builder.py) / [video inference](../../cap-x/capx/third_party/sam3/sam3/model/sam3_video_inference.py) | 官方 predictor 的 detector/tracker 组合、输出缓冲和补点后的传播 |
| [当前 detector.py](../sam3_runtime/detector.py) / [tracker.py](../sam3_runtime/tracker.py) | BF16、bbox-only、流式状态、两种身份策略和诊断 |
| [当前 grounding.py](../grm_runtime/grounding.py) / [common.py](../grm_runtime/common.py) | HTTP 对齐校验、普通检测歧义判断、在线目标词解析 |
| [当前 tracking.py](../monitor_runtime/tracking.py) / [grm_backend.py](../monitor_runtime/grm_backend.py) | 最新完整快照、超龄错误、双分支输入和发布 |

本次只编写比较文档并验证少量纯函数行为，没有修改模型、跟踪逻辑、机器人控制或运行服务。
