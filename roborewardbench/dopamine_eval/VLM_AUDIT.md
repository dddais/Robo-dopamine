# Grounding VLM 审查流水线

`vlm_audit.py` 用本地 Qwen3-VL 对冻结的 `audit_sample.jsonl` 审查 GroundingDINO 的**绿色 selected bbox**。它是对 [`MANUAL_AUDIT.md`](MANUAL_AUDIT.md) 判据的自动化实现，输出标签格式兼容人工 annotations，但默认不会覆盖人工审核产物。

## 审查证据与判据

每条记录提供给 VLM 的信息只有：

- instruction 和 `selected_parse`；
- BEFORE / AFTER 原图上绘制的绿色 selected bbox；
- 两个 selected bbox 的放大 crop；
- 默认额外提供原视频均匀抽取的 8 帧联系表。

不会提供 reward、GRM score、`gpt5_mini_check`、GroundingDINO score、`steering_ready` 或其他 detector candidates。模型仍须依据与人工规范相同的规则判断 target parse、目标属性/部件、同类实例，以及首末帧身份是否一致。

视频联系表的 `review_basis=video_keyframes` 明确表示均匀抽帧；它不等价于人工逐帧观看。因此模型无法可靠判定时会输出 `uncertain`，不应用置信度或 detector proxy 强行改判。

## 运行

从 Robo-Dopamine 仓库根目录运行。推荐 8B 作为正式审核器；4B 适合连通性检查或资源受限实验。

```bash
conda run -n robo-dopamine python -m roborewardbench.dopamine_eval.vlm_audit \
  --output-dir roborewardbench/dopamine_eval/outputs/counterfactual_reward1 \
  --model-path /home/dais/workspace/model/Qwen3-VL-8B-Instruct \
  --device cuda:0 --review-mode video
```

最小连通性测试只写可恢复的 raw cache，不会生成不完整的审核报告：

```bash
conda run -n robo-dopamine python -m roborewardbench.dopamine_eval.vlm_audit \
  --output-dir roborewardbench/dopamine_eval/outputs/counterfactual_reward1 \
  --model-path /home/dais/workspace/model/Qwen3-VL-4B-Instruct \
  --max-samples 1
```

流水线按条落盘，重跑时只有 grounding fingerprint、模型文件清单、提示词/生成参数或审查模式改变的条目才会重新推理。`--review-mode endpoints` 可以关闭视频联系表。

## 产物

- `vlm_audit_raw.jsonl`：原始模型响应、缓存签名和显式错误；
- `vlm_audit_annotations.jsonl`：与人工 annotation schema 兼容的机器标签；
- `vlm_audit.jsonl` / `.csv`：与 frozen sample 合并后的权威 VLM 记录；
- `vlm_audit_summary.json` 和 `vlm_audit_report.md`：结果、错误和解释边界。

推理或 JSON 校验失败不会丢弃样本，而是写为 `uncertain`、置信度 0，并在 raw record 中保留错误。这样不会把失败静默当成 `correct`。

## 接入 attention-mask 前的校准与提升

`attention_mask` 只读取 `manual_audit.jsonl`，因此默认不会把 VLM 结论当作人工标签。应先在独立、盲审的人类校准集上报告 VLM–人工一致率，尤其检查 `correct` 的 precision、错误类别混淆和 `uncertain` 率；通过预先定义的门槛后，才显式提升：

```bash
conda run -n robo-dopamine python -m roborewardbench.dopamine_eval.vlm_audit \
  --output-dir roborewardbench/dopamine_eval/outputs/counterfactual_reward1 \
  --model-path /home/dais/workspace/model/Qwen3-VL-8B-Instruct \
  --promote
```

`--promote` 会写入 `manual_audit_annotations.jsonl` 并运行现有的 fingerprint 校验/聚合器，生成下游读取的 `manual_audit.*`。该动作是显式的，因为产物名称为兼容旧接口，内容仍应在实验记录中标为 `audit_source=qwen3_vl`，不能声称为人工审查。
