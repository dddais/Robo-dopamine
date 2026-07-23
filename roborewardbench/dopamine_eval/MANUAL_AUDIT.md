# Grounding 人工审核操作指南

本指南说明 `roborewardbench/dopamine_eval` 产生的 LLM + GroundingDINO 结果如何人工确认，
以及 attention-mask 实验实际读取哪些审核文件。命令均从 Robo-Dopamine 仓库根目录执行。

## 1. 为什么必须人工审核

以下自动字段都只是 proxy，不能证明框中了 instruction 的目标物体：

- GroundingDINO `score` / `accepted`；
- 首末帧 bbox 的 IoU、中心距离和 `pair_consistency`；
- `steering_ready` 自动门控；
- 两个 LLM 对 target phrase 的一致性。

例如 detector 可以在首末帧都稳定框住同一个错误物体。因此正式 target-bbox intervention
只读取 `manual_label=correct` 且 fingerprint 仍与当前 selected boxes 一致的记录。

## 2. 审核时需要看的文件

以默认目录为例：

```text
roborewardbench/dopamine_eval/outputs/counterfactual_reward1/
```

输入/参考文件：

| 文件 | 作用 |
|---|---|
| `audit_sample.jsonl` | 冻结的待审核样本清单；每行给出 `example_id`、instruction、target parse、状态和单样本图名 |
| `audit_sheets/*.jpg` | 多条样本拼成的快速浏览页 |
| `visualizations/<visualization_file>` | 单条样本的 before/after 大图，最适合做最终判断 |
| `grounding_results.jsonl` | selected bbox、候选框、分数、query、pair consistency 的完整机器记录 |
| `frame_manifest.jsonl` | before/after 原始 PNG 路径与视频帧索引 |
| `frames/` | 无框的 endpoint PNG；框遮挡细节时可对照 |
| 数据集原始 `.mp4` | 端点不足以判断同类实例、关系或遮挡时查看完整视频 |

图中绿色粗框是实际写入 `before.selected` / `after.selected`、随后会用于 attention mask 的框；
橙色细框只是其它 detector candidates。审核结论必须针对绿色框，不能因为某个橙色框正确
就把当前 selected 结果标成 correct。

## 3. 每条样本怎么判断

对 `audit_sample.jsonl` 的每一行执行：

1. 先读 `task`，确认 instruction 直接操作的目标实体；放置目标/参照物通常不是被抓取、推动
   或移动的实体。
2. 检查 `selected_parse.target_phrase` 是否提取了这个实体。parse 本身错误时，即使 detector
   框与错误 phrase 一致，也应标 `incorrect`。
3. 打开 `visualizations/<visualization_file>`，分别检查 BEFORE 和 AFTER 的绿色框。
4. `correct`：两端绿色框都对应 instruction 的同一目标实体；物体部件、颜色/大小属性和同类
   实例也必须正确。
5. `incorrect`：任一端框错物体、框到机器人/背景、首末目标身份切换、只框到参照物，或
   target parse 错误。
6. `uncertain`：仅凭现有图/视频仍无法可靠判断。不要用 detector score 猜测。
7. 遇到同色同类实例、关系词、物体被遮挡或 endpoint 语义不足时，必须查看原视频，并把
   `review_basis` 写成 `full_video`。

推荐的 failure category（代码允许自由字符串，但同批审核应统一词表）：

```text
wrong_target_parse
wrong_object
reference_object_confusion
same_category_instance_confusion
object_part_confusion
robot_or_gripper_confusion
background_scene_box
endpoint_identity_switch
empty_or_severely_misaligned_box
```

## 4. 填写哪个文件

人工只编辑：

```text
manual_audit_annotations.jsonl
```

它必须与 `audit_sample.jsonl` 一一对应：不能少、不能多、不能重复 `example_id`。每行格式：

```json
{"example_id":"subset/example.mp4","manual_label":"correct","failure_category":null,"reason":"首末帧绿色框均覆盖 instruction 指定的目标物体。","review_basis":"endpoint_visualization"}
```

错误样本：

```json
{"example_id":"subset/example.mp4","manual_label":"incorrect","failure_category":"endpoint_identity_switch","reason":"首帧框住目标块，末帧切换到另一物体。","review_basis":"full_video"}
```

合法 `manual_label` 只有：

```text
correct | incorrect | uncertain
```

规则：

- 所有行的 `reason` 必填，写可复核的视觉依据；
- `incorrect` 的 `failure_category` 必填且非空；
- `correct`/`uncertain` 的 `failure_category` 可以为 `null`；
- `review_basis` 可省略，默认 `endpoint_visualization`，但看过完整视频时应显式写
  `full_video`。

## 5. 校验并生成正式审核产物

填完所有行后运行：

```bash
python -m roborewardbench.dopamine_eval.audit \
  --output-dir roborewardbench/dopamine_eval/outputs/counterfactual_reward1
```

该命令会拒绝以下情况：缺失/多余/重复 ID、非法 label、空 reason、incorrect 没有 failure
category、audit sample 已落后于当前 grounding，以及人工标签对应的 selected boxes 已变化。

成功后生成/更新：

| 文件 | 用途 |
|---|---|
| `manual_audit.jsonl` | sample 与人工标签合并后的权威逐条记录；attention split 实际读取它 |
| `manual_audit.csv` | 便于人工查看的平铺版本 |
| `manual_audit_summary.json` | correct/incorrect/uncertain 与 ready/non-ready 汇总 |
| `audit_report.md` | 人工审核方法、错误样本和解释边界 |
| `audit_grounding_fingerprints.jsonl` | 把每个人工结论冻结到当时的 task/parse/selected before/after boxes |
| `summary.json` / `summary.md` | 回写人工审计摘要 |

第一次成功运行会创建 fingerprint 文件。以后 grounding query、selected bbox、task/parse、
状态等任何关键字段变化，审核命令都会失败，防止旧标签静默套到新框上。只有明确准备重新
审核全部样本时，才应重新生成 sample/annotations/fingerprint；不能为了绕过 stale 检查只
删除 fingerprint 然后沿用旧结论。

## 6. attention-mask 实验到底使用哪些审核结果

`roborewardbench/attention_mask/dataset.py` 会重新计算当前 grounding fingerprint，并要求与
`manual_audit.jsonl` 完全一致：

- `manual_correct_ready`：`manual_label=correct` 且 `steering_ready=true`；当前 19 条，用作
  evaluation；
- `manual_correct` 中非 ready：当前 10 条，用作 discovery；
- `auto_ready`：只用于探索，不能替代人工审核。

当前 `audit_sample.jsonl` 是分层目的性样本，审计比例不能无偏外推到 228 条全量。如果要把
剩余自动 ready 样本加入正式实验，必须先把它们加入新的冻结 audit sample、逐条审核并生成
新 fingerprint；不能直接把 `steering_ready=true` 当作 `correct`。

## 7. 建议的质量控制

当前代码能保证标签完整性和框版本一致性，但不能消除人的主观误差。用于论文主结果时建议：

- 两位审核者独立、盲于 reward 和模型 intervention 结果进行标注；
- 记录 reviewer ID/版本，先计算一致率，再对分歧做 adjudication；
- 审核期间不看 reward、GRM score 或“哪个条件更好”，避免确认偏差；
- 保存使用过的 audit sample、annotations、fingerprints 和审核规范版本。
