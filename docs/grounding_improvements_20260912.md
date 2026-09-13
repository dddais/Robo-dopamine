# SAM3 离线 grounding 改进与验证（2026-09-12）

本文针对 `mydata_bench`，依据 `mydata_bench/exp_use.md`、
`sam3_grounding_comparison.md`、本机历史 JSONL 和真实 SAM3 权重验证。
没有改动在线 Monitor，也没有重跑已有 reward/attention 实验。

## 1. 历史失败来自哪里

按 `(example_id, frame)` 取 JSONL 最后一条记录，v2 共 1213 条视频指令样本：

| 首帧 / 末帧 | 样本数 | 含义 |
| --- | ---: | --- |
| ok / ok | 846 | 自动端点可用，不等于全程正确 |
| no_detection / no_detection | 315 | 首帧没有选出目标，尚未进入有效传播 |
| ok / no_detection | 52 | 首帧检测成功，但未保留有效末帧 |

因此，367 条不可用样本中，315 条首先需要改进检测/语义选择，
只修改 tracking 无法覆盖主要失败来源。旧 v1 的 134 个 invalid 端点
对应 67 条样本的 `Broken pipe`，属于运行错误，不应计作检测模型漏检。

具体缺陷：

- 关系启发式只保留末尾名词，`salmon sushi` / `shrimp sushi` 退化成 `sushi`。
- `second/fourth ... from the left` 没有序数算子，候选按置信度选，
  即使返回框，也未必是指令中的第几个物体。
- 完整短语与裸类别、部件与所属物体混合竞分，容易丢失颜色、亚型和部件身份。
- 部分目标较小；数据集中的 `cup` 在图像里呈小碗外形，虾寿司呈虾形玩具外观，
  原始文本提示不适配模型的识别词汇。
- 视频输出指定 ID 缺失时会选其他高分 ID；补点使用 mask 质心，
  弯曲/环形 mask 的质心可能落在背景；两次传播结果被合并，可能保留旧框。
- 只看双端点无法发现中间缺失；incremental 曾允许对当前图像借用附近帧的 bbox。

## 2. 实现改变

### 语义和首帧选择

新增 `subject_phrase` 和 `ordinal_index`，兼容旧 TargetSpec JSONL。
解析支持 `and put`、单独的 grasp/lift/touch、最左/最右、从左/右/上/下数第 N 个，
以及原有左右、最近、最远关系。复合名称和颜色保留在 SAM3 查询中。
旧解析文件中的明确序数/关系短语也会在读取时恢复几何操作。

几何选择先去重，然后按框中心排序，或相对于参照物筛选。候选不足、
几何位置无法区分、参照物候选分差过小时明确返回原因。
不再用裸类别/整个父物体作为默认语义退路。普通目标的近分多候选也会标记歧义。

### 检测

新配置保持检测阈值 **0.30**、mask 阈值 **0.50**，增加全图加 2×2 重叠裁剪检测。
裁剪框和 mask 映射回原图，跨视图/查询候选去重后再执行 top-N。
落在裁剪内部边界的截断框被过滤。小型 LRU 缓存按图像内容和查询复用结果，
适合本数据集同视频的不同指令。

同义/外观名称通过 YAML 的 `query_aliases` 显式配置，不是所有数据集通用的等价规则：

| 指令名称 | 新配置的附加检测词 |
| --- | --- |
| shrimp sushi | shrimp、prawn |
| salmon sushi | salmon nigiri |
| cup | small bowl |

有颜色等前缀时保留前缀，如 `red cup` → `red small bowl`。
每个候选保存实际 query、semantic_query、裁剪坐标，供检查误检。
若迁移到真正区分杯子/碗或虾/虾寿司的场景，应移除对应映射。

### 视频跟踪

继续使用本机官方 SAM3 predictor，按以下顺序执行：

1. 首帧 visual box 提示，初始 ID 按与选定检测框的 IoU 绑定，不按最高置信度绑定。
2. 完成正常传播。指定 ID 不在某帧输出中时记为缺失，绝不选其他 ID。
3. 新配置 `tracking_prompt: mask_refine` 在缓存就绪后，以检测 mask 内部距离变换的
   最大值点修正实例并重新传播。默认 `visual_box` 模式仍只在有缺失时尝试修正。
   如果初始 visual 提示未绑定实例，先建立缓存，再用检测 mask 在首帧建立新实例，
   必须通过初始框 IoU 校验；不会在视频后段任意挑一个物体。
4. 修正结果完整替换前一遍结果。中间帧的全分辨率 mask 当场释放，只保留端点 mask。
5. 记录 frame_coverage、missing_frame_count、longest_missing_run、obj_id、refined 等。

没有引入相邻框 IoU 的位移硬阈值。`tracking_anchor_min_iou` 比较的是
**同一初始帧**的检测与提示框，不会因物体被拿起后移动而拒绝轨迹。
稳定的模型 ID 仍不能排除模型内部漂移。

本机官方实现不支持在新会话里直接加点，必须先正常传播；真实验证已覆盖这个约束。

### 结果与评测一致性

- 运行指纹包括实现源码、模型/config、目标结构、任务和视频 hash；变化会使旧缓存失效。
- 重试一个视频时同时更新两个端点，每次轨迹和 mask 使用独立路径，旧 JSONL 引用不变。
- tracking 异常保留有效首帧，末帧标记 invalid，并单独记录 tracking_error。
- 自动 cohort 排除由两次不同尝试拼接出的端点，可选 `--min-tracking-coverage`。
- 新 track.json 标记 `bbox_frame_policy: exact`。官方 incremental 路径缺少某个
  hop 的同帧 bbox 时明确失败；旧 track.json 保持历史补缺语义以便复现。
- 新增 `grounding diagnose`，报告最新端点状态、全程覆盖和与旧结果的共同样本变化。

## 3. 真实权重检查和结论边界

A100 上使用现有 SAM3 权重、FP32 图像检测，查询探针保持同图和 0.30 阈值：

| 同一图像上的查询对照 | 候选数 |
| --- | --- |
| shrimp sushi → shrimp | 0 → 4 |
| cup → small bowl | 0 → 4 |
| sushi → salmon sushi → salmon nigiri | 0 → 1 → 2 |

也尝试了 `hokkigai nigiri`：虽然有 3 个高分框，但图上是三种不同寿司，
存在亚型混淆，因此**没有将该词加入正式配置**。surf clam sushi 的稳定识别仍未解决。

选取 4 条已有视频做完整首帧检测和全视频传播，其中 3 条旧首帧不可用、1 条旧可用：

| example_id | 旧双端点 | 新双端点 | 新轨迹有框帧比例 |
| --- | --- | --- | ---: |
| fail/ljx_lfz_task_2_1/1 | 可用 | 可用 | 100.00% |
| fail/ljx_lfz_task_3_2/15 | 不可用 | 可用 | 59.03% |
| fail/ljx_lfz_task_3_5/1 | 不可用 | 可用 | 71.57% |
| fail/ljx_lfz_task_4_4/1 | 不可用 | 可用 | 75.32% |

六时刻 contact sheet 检查支持首帧选中了预期序数/关系实例，末帧也恢复到对应物体；
三个视频中机器人遮挡期间仍有较长缺失，不能把 4/4 端点可用写成 4/4 全程准确。
本批是用于发现和验证问题的定向样例，不是随机测试集，也没有逐帧人工 GT。
随后已完成全部 1213 条重跑：991 条双端点可用（81.70%），335 条全帧有框，
运行异常和产物一致性错误均为 0。与旧版配对，恢复 265 条、排除 120 条，净增 145 条。
完整结果见 [grounding_full_run_20260912.md](grounding_full_run_20260912.md)。
这些统计反映自动可用性，尚不能说明人工验证的目标正确率。

产物目录：`results/diagnostics/grounding_improvement_20260912/`。
`query_probe.json` 保存查询对照；`smoke/sam3/grounding.jsonl` 保存端点；
对应 track.json、预览 MP4 和 contact sheet 位于记录的 provenance 路径。
`parser_smoke/targets.jsonl` 单独验证 Qwen3-4B 的四条真实解析；视频 smoke
使用确定性解析，空间部分与 Qwen 的确定性覆盖逻辑一致。

## 4. 运行新实验

从仓库根目录运行。新目录与历史 v2 隔离：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n rewardbench-sam3 \
  python mydata_bench/run_grounding.py parse \
  --config mydata_bench/configs/mydata_grounding_v2_improved.yaml

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n rewardbench-sam3 \
  python mydata_bench/run_grounding.py run --backend sam3 --retry-failed \
  --config mydata_bench/configs/mydata_grounding_v2_improved.yaml

python3 mydata_bench/run_grounding.py diagnose \
  --run-dir results/mydata_bench/grounding_v2_improved/sam3 \
  --baseline-run results/mydata_bench/grounding_v2/sam3
```

仍可添加 `--shard-id 0 --num-shards 3`，分别运行三个分片。
同一运行目录的分片数保持一致。等所有进程结束后再读取合并文件或冻结 cohort。

只要求双端点的 cohort 可沿用旧命令；要做全帧 attention 或精确 incremental，
可以先要求整段有框：

```bash
python3 mydata_bench/prepare_reward_cohort.py \
  --dataset-root /home/dais/workspace/data/mydata_v2/new --split all \
  --grounding-run results/mydata_bench/grounding_v2_improved/sam3 \
  --min-tracking-coverage 1.0 \
  --output-dir results/mydata_bench/cohorts/auto_grounded_v2_improved_full_frames
```

这是覆盖完整性筛选，仍不是人工正确率认证。严格筛选会排除完全遮挡的轨迹，
统计时应同时报告全集、双端点子集和全帧子集。

ranking 使用 `mydata_ranking_grounding_v2_improved.yaml`，步骤相同。
如果将新 grounding 用于 attention，需修改新的实验配置指向新的 grounding/cohort，
重新生成 ranking，并使用新的实验输出目录。

验证命令：

```bash
python3 -m pytest tests/test_mydata_*.py mydata_bench/tests \
  tests/test_reward_cohorts.py tests/test_dopamine_eval_grounding.py -q
python3 -m compileall -q mydata_bench
git diff --check
```
