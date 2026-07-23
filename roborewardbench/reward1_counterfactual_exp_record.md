# Robo-Dopamine 在反事实 reward=1 子集上的实验记录

实验日期：2026-07-22  
数据集：`/home/dais/workspace/data/RoboRewardBench_counterfactual_reward1`  
模型：`Robo-Dopamine-GRM-2.0-8B-Preview`

## 1. 结论

228 条反事实 `reward=1` 样本全部推理成功。固定分箱后的 micro 精确准确率为
**28.07%（64/228）**，21-subset macro 精确准确率为 **28.16%**。模型把另外
164/228（71.93%）条本应为“无目标相关进展”的任务预测成 2–5 分，说明这一切片对
instruction-conditioned grounding 较难。

准确率需要和 MAE 一起看。这个数据集的真值全是 1，因此这里的精确准确率实际上就是
`reward=1` 的召回率，不能代表模型在完整 1–5 分类上的总体准确率。

## 2. 评测协议

- checkpoint 指纹：
  `3b4f8d67ee0191bd0985527fc32148676e57c7f756459530f825721422b4fd19`
- metadata：228 条，SHA-256
  `ad2b369db530aba5f93b749c681e96cca8ce905ffd7f2ac36d1681b418938d58`
- 228/228 个视频与 metadata 精确对应，21 个 OXE subset。
- `forward` 端点协议：使用真实首帧和真实终帧。
- RoboRewardBench 只有单视角，因此把同一视角复制到三个相机槽位。
- goal image 使用不泄漏目标状态的中性灰占位图。
- batch size 8、temperature 0、seed 0。
- 连续输出 `p∈[0,1]` 使用固定阈值
  `0.125, 0.375, 0.625, 0.875` 映射到 1–5；没有使用 test 标签调阈值。
- 置信区间使用 10,000 次、seed=0 的 subset 内分层 bootstrap。

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n robo-dopamine \
  python -m roborewardbench.run_benchmark \
  --dataset-root /home/dais/workspace/data/RoboRewardBench_counterfactual_reward1 \
  --split test \
  --model /home/dais/workspace/Robo-Dopamine/pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview \
  --mode forward \
  --frame-sampling 1fps \
  --batch-size 8 \
  --temperature 0 \
  --seed 0 \
  --bootstrap-samples 10000 \
  --output-dir outputs/roborewardbench/rd8b_forward_counterfactual_reward1
```

## 3. 完整指标

| 指标 | 21-subset Macro | 95% CI | Micro |
|---|---:|---:|---:|
| 固定分箱 MAE | **1.2194** | [1.1235, 1.3149] | 1.2500 |
| 精确准确率 | **28.16%** | [22.98%, 33.28%] | 28.07% |
| ±1 容差准确率 | **67.56%** | [63.44%, 71.67%] | 68.42% |
| 连续序数 MAE `|1+4p-y|` | 1.2913 | [1.2126, 1.3720] | 1.3109 |
| 区间序数 MAE | 0.8126 | [0.7360, 0.8912] | 0.8344 |

其他有效性检查：

| 检查 | 结果 |
|---|---:|
| 原始/去重记录 | 228 / 228 |
| 有效/无效预测 | 228 / 0 |
| invalid rate | 0% |
| 与子集 metadata 的 ID、task、reward、subset 精确匹配 | 是 |
| `official_comparable` | `false`（预期；这里只评估完整 benchmark 的一个切片） |

## 4. 预测分布

由于所有真值都是 1，下面一行就是该切片的有效混淆矩阵：

| 真实 reward | 预测 1 | 预测 2 | 预测 3 | 预测 4 | 预测 5 |
|---:|---:|---:|---:|---:|---:|
| 1（228 条） | 64（28.07%） | 92（40.35%） | 38（16.67%） | 19（8.33%） | 15（6.58%） |

- 平均离散预测为 2.25，平均有符号误差为 +1.25。
- 预测为 2–5 的比例是 71.93%，可解释为“把 no-success 判断成存在进展”的错误率。
- 预测为 3–5 的比例是 31.58%；预测为 5 的比例是 6.58%。
- 平均连续 progress 为 0.3277，中位数为 0.25，四分位数为 0.111/0.50。

`underprediction_rate=0` 是标签下界为 1 的必然结果，不代表模型没有低估倾向。

## 5. 为什么同时报告准确率和 MAE

- **精确准确率**直接回答“模型是否输出正确的 1 分”。
- **±1 容差准确率**允许输出相邻的 2 分；在本数据中等价于预测 1 或 2。
- **MAE**保留序数距离：把 1 错判成 2 只罚 1，把 1 错判成 5 要罚 4。普通准确率会把
  两者视为同样错误。
- **预测分布和有符号误差**揭示模型是系统性高估还是低估。
- 连续序数和区间序数 MAE 用于检查结论是否主要由离散阈值造成。

这个单标签切片不能可靠计算 balanced accuracy、macro-F1、Spearman 相关、Cohen's kappa
或 AUROC：真实标签没有方差，也没有 2–5 类负例。若要报告这些指标，应在完整 1–5
test split 上计算。

## 6. 重复运行稳定性

这 228 条样本也存在于此前的 2,831 条完整 benchmark 运行中。对两次原始输出逐条比较：

- 214/228 条连续输出完全相同；14 条不同。
- 223/228 条固定分箱标签相同；5 条跨过离散阈值。
- 完整运行切片的 micro 准确率为 27.63%，本次独立运行是 28.07%，相差 0.44 个百分点。
- 完整运行切片的 Macro MAE 为 1.2204，本次是 1.2194，相差 0.0010。

因此 temperature=0、seed=0 在当前 vLLM/GPU 批处理下并不保证逐 token 的 bitwise
determinism，但聚合指标非常稳定。原始输出必须随实验结果保存；对细小模型差异做正式
比较时，建议多次运行或报告置信区间。

## 7. 结果文件

- `outputs/roborewardbench/rd8b_forward_counterfactual_reward1/predictions.jsonl`
- `outputs/roborewardbench/rd8b_forward_counterfactual_reward1/metrics.json`
- `outputs/roborewardbench/rd8b_forward_counterfactual_reward1/run_manifest.json`
- `outputs/roborewardbench/rd8b_forward_counterfactual_reward1/reproducibility/`

本次直接运行文件 SHA-256：

- predictions：`7a09487182a80234387267bc16d5c40d0c5286b12774957f3b4a3f36c38b4eb1`
- metrics：`b26619e96878216b06a49cb6f866034fe1fc847b1ad3216324c3fce0e0044859`
- manifest：`22fa4f502db228528688acc171622ecf28042785e9fcb7ffe956a4a265e45120`
