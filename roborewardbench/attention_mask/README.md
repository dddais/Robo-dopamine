# RoboRewardBench target-bbox attention-mask 实验

本目录实现“先用冻结的 grounding bbox 发现 heads，再在独立 evaluation 样本上施加
attention-logit intervention”的 RoboRewardBench 实验。统一入口已经改名为：

```text
run_targetbbox_roborewardbench.py
```

它与标准 Robo-Dopamine benchmark 评测是两条不同流程：

- 标准评测：`python -m roborewardbench.run_benchmark`，不需要 grounding，也不修改 attention；
- target-bbox 评测：`python run_targetbbox_roborewardbench.py ...`，需要人工确认的 bbox，并报告
  baseline 与 intervention 的配对变化。

总导航见 [`../README.md`](../README.md)，grounding 人工审核的逐步说明见
[`../dopamine_eval/MANUAL_AUDIT.md`](../dopamine_eval/MANUAL_AUDIT.md)，当前 hook/token 的
逐项静态与实际 Qwen runtime 审计见 [`MASK_AUDIT.md`](MASK_AUDIT.md)。

## 1. 数据划分与适用边界

当前 grounding 人工审计中：

| 范围 | 数量 | 用途 |
|---|---:|---|
| 人工 `correct` 且自动 `steering_ready` | 19 | intervention evaluation |
| 人工 `correct`、自动非 ready | 10 | head discovery |
| 全部人工 `correct` | 29 | 上述两组之和 |
| 自动 `steering_ready` | 143 | 只能作为待审核候选，不能自动当作正确 bbox |
| grounding 全量 | 228 | 不能直接用于正式 intervention 结论 |

默认 `prepare` 会冻结这个天然不重叠的 10/19 split。它是 RoboRewardBench test 内经过
目的性人工审核的子集，不是 2,831 条完整 test 的官方得分。head discovery 不能看 19 条
evaluation，reward 标签也不会进入 prompt、grounding、head ranking 或生成；只有 `metrics`
阶段按 `example_id` 事后连接 reward。

### 1.1 外部固定 heads + 自动 bbox 的迁移实验

探索性跨数据实验可以跳过本数据集内的 head discovery，直接冻结另一数据集产生的完整
ranking JSON。此时使用 `--fixed-head-ranking`，并通过 `--selection-mode` 选择自动 bbox：

- `auto_detected`：before/after 都存在 selected bbox，不要求置信度或人工标签；
- `auto_ready`：进一步要求自动 `steering_ready=true`；
- 两者都不把自动 bbox 解释为人工确认的正确目标。

reward=1 当前 grounding 中，`auto_detected` 覆盖 227/228，`auto_ready` 覆盖 143/228。
缺少任一 endpoint bbox 的样本不会用全图或虚构 bbox 代替。

固定 carrot ranking、after-role 的准备与运行示例：

```bash
RANKING=results/attention/3data_3instruction_20260709/rank_carrot_forward_last_prompt/head_ranking.json
OUT=results/attention/roborewardbench_reward1_fixed_carrot_auto

python run_targetbbox_roborewardbench.py prepare \
  --grounding-dir roborewardbench/dopamine_eval/outputs/counterfactual_reward1 \
  --dataset-root /home/dais/workspace/data/RoboRewardBench_counterfactual_reward1 \
  --fixed-head-ranking "$RANKING" \
  --selection-mode auto_detected \
  --target-role after \
  --output-root "$OUT"

python run_targetbbox_roborewardbench.py experiment \
  --grounding-dir roborewardbench/dopamine_eval/outputs/counterfactual_reward1 \
  --dataset-root /home/dais/workspace/data/RoboRewardBench_counterfactual_reward1 \
  --fixed-head-ranking "$RANKING" \
  --selection-mode auto_detected \
  --target-role after \
  --top-ks 8,64 \
  --biases 0,2,4 \
  --gpus 2,3 \
  --num-shards 2 \
  --output-root "$OUT"

python run_targetbbox_roborewardbench.py metrics \
  --dataset-root /home/dais/workspace/data/RoboRewardBench_counterfactual_reward1 \
  --fixed-head-ranking "$RANKING" \
  --selection-mode auto_detected \
  --target-role after \
  --num-shards 2 \
  --output-root "$OUT"
```

若要严格复现旧 `run_targetbbox_success_experiments.py` 的 intervention
范围，将三条命令中的 `--target-role after` 改为：

```bash
--target-role after_high
```

此时只在 `after_cam_high` 内对目标 bbox tokens 加 `+bias`、对同一张图的
其余视觉 tokens 加 `-bias`；另外七张图和文本 tokens 均不修改。请使用新的
`--output-root`，不要与已经完成的 `target_role=after` 结果混用。

外部 ranking 的 SHA-256、路径和 head ranking 会进入 run provenance。运行器跳过的只是
“ranking 来自当前 split discovery”的链接检查；模型层数、query-head 数和所有实际 head
下标仍会校验。此模式的结论是“固定 head set + 自动 grounding 的端到端迁移效果”，不能
替代人工 bbox 验证后的 grounding-specific 因果结论。

## 2. 代码结构

```text
attention_mask/
├── dataset.py          grounding/audit join、fingerprint、10/19 split
├── masking.py          bbox→视觉 token、wrong region、per-head mask hook
├── modeling.py         Qwen3-VL eager attention、八图 GRM 输入、score 生成
├── rank_heads.py       discovery-only、bbox 面积校正后的 head ranking
├── run_experiment.py   baseline + 4 类 intervention、shard、断点续跑
├── metrics.py          paired effects、controls、bootstrap、事后 ordinal 指标
├── curve.py            从已保存结果绘制 bias dose-response（CPU）
└── visualize.py        baseline/candidate endpoint attention heatmap（GPU）

../../run_targetbbox_roborewardbench.py
                        prepare/rank/experiment/metrics/curve/video 统一入口
```

## 3. attention mask 到底施加在哪里

默认 `target_role=both` 时，RoboRewardBench 单视角 before/after 图各复制到 GRM 的三个
camera span。grounding 的 before bbox 映射到三个 before span，after bbox 映射到三个 after
span。reference start/end、文本 token 和其它非目标 role 均不修改。

在 discovery 排名选出的每个 query head 上，`boost_suppress` 对 attention logits 加：

```text
target bbox keys                         +bias
同一 target role 的其它 endpoint image keys   -bias
reference images / text / future decode keys   0
```

这里是 soft attention-logit bias，不是把 token 删除的二值 mask。hook 注册在：

```text
model.model.language_model.layers[layer].self_attn
```

Qwen3-VL eager attention 收到的 causal mask 为 `[batch, 1|num_heads, query, key]`。代码构造
`[1, num_query_heads, 1, key]` 的 bias：只写选中的 query-head 行，然后沿 batch 和所有 query
行广播。默认在 prefill 和后续 decode 都生效；decode 的 key 序列变长时，只在右侧补 0，
所以原 prompt 中的绝对视觉 token 位置不发生偏移，新增文本 token 不被干预。`--decode-only`
会跳过 `query_length>1` 的 prefill，只在自回归 decode 生效。

已加入的防错检查/测试覆盖：

- 八张图的 span 标签必须完整且唯一，span 长度必须等于 merged vision grid；
- bbox 使用绝对 LM token index，并检查 before/after/both 的精确 span 集合；
- `target ∪ other_image` 必须恰好等于所选 endpoint 图像 token，且不包含 reference；
- bbox 同时复制到三个 before/after camera slots；
- wrong region 与 bbox 不重叠且 token 数相同；
- 只有指定 layer/head 发生 `±bias`；batch/query 广播不串 head；
- prefill、短 key、等长 key 和扩展 decode key 的位置保持正确，扩展 token bias 为 0；
- `bias=0` 是 no-op；正式已有结果中 125/125 zero-bias 对照也逐值完全相同。

`candidate_wrong` 在目标超过一半 grid、无法构造同尺寸完全不重叠区域时会明确缺失，并由
metrics 报告较小的 paired n，不会偷偷缩小控制框。

## 4. 一条命令运行完整流程

从仓库根目录运行。建议 smoke 和正式实验使用不同输出目录：

```bash
cd /home/dais/workspace/Robo-Dopamine

python run_targetbbox_roborewardbench.py all \
  --smoke \
  --python-bin /home/dais/miniconda3/envs/robo-dopamine/bin/python \
  --gpus 2 \
  --bootstrap-samples 100 \
  --output-root results/attention/roborewardbench_targetbbox_smoke
```

`--smoke` 只跑 1 条 discovery 和 1 条 evaluation，不能用于结论；它仍会生成 metrics、curve
和 heatmap 视频，适合检查端到端依赖。

正式运行示例：

```bash
python run_targetbbox_roborewardbench.py all \
  --python-bin /home/dais/miniconda3/envs/robo-dopamine/bin/python \
  --gpus 2,3 \
  --num-shards 2 \
  --top-ks 8,64 \
  --biases 0,2,4,6 \
  --ranking-score excess_mass \
  --bootstrap-samples 10000 \
  --video-top-k 64 \
  --video-bias 4 \
  --output-root results/attention/roborewardbench_counterfactual_reward1_excess
```

可逐阶段执行：

```bash
python run_targetbbox_roborewardbench.py prepare
python run_targetbbox_roborewardbench.py rank --gpus 2
python run_targetbbox_roborewardbench.py experiment --gpus 2,3 --num-shards 2
python run_targetbbox_roborewardbench.py metrics --num-shards 2
python run_targetbbox_roborewardbench.py curve --num-shards 2
python run_targetbbox_roborewardbench.py video --gpus 2 --video-top-k 64 --video-bias 4
```

昂贵的 ranking/experiment 有 provenance 和 resume 检查。旧目录中的模型、split、代码或参数
指纹不匹配时不会和新结果静默混写。

## 5. curve 的含义与产物

`run_targetbbox_success_experiments.py` 的 curve 是一个成功轨迹内的时间/进度曲线；而
RoboRewardBench 的标准 forward 协议每个视频只产生一个首末端点分数。因此这里不会把
endpoint 人为伪装成时间轨迹，`curve` 表示 intervention dose-response：

```text
x: attention-logit bias
y1: 各 condition 的 mean raw score
y2: 相对同一样本 baseline 的 mean paired shift
```

输出：

```text
artifacts/curves/
├── curve.csv
├── curve.json
├── score_curve.png
└── paired_shift_curve.png
```

error bar 是对 observed examples 直接 bootstrap 的 95% CI。`curve` 只读取已保存 JSONL，
不加载模型，可以单独在 CPU 重建。

## 6. heatmap 视频的含义与产物

`visualize.py` 对冻结 evaluation 样本分别做 baseline 和 `candidate_target` forward，提取最后
一个非特殊 prompt query 对图像 keys 的 post-softmax attention。空间图对正式 intervention
使用的同一组 top-k heads（可以跨 layer）做平均；baseline/candidate 在每个样本和 endpoint
使用共同的 raw-attention 色标。绿色框是 frozen target bbox。

正式 mask 同时修改 role 内全部三份 camera span；视频只绘制 `cam_high` 那一份，manifest
会记录这个区别。heatmap 按 merged visual-token cell 用 nearest-neighbor 绘制，不进行可能
造成视觉偏移的 cubic 插值。绿色框仍是像素 bbox；当输入只产生很粗的 token grid（例如
4×4）时，与小 bbox 相交的 token cell 会明显大于绿色框，这是当前视觉 token 分辨率与
`intersection` 映射的真实结果，不是 bbox 坐标偏移。输出：

```text
artifacts/attention_video/
├── baseline_attention.mp4
├── candidate_target_attention.mp4
├── baseline_vs_candidate_attention.mp4
├── attention_video_manifest.json
└── frames/*.png
```

这些 MP4 按 evaluation 样本依次展示 before/after endpoint，不是原视频的逐帧 attention
tracking。当前 frozen grounding 只有首末帧 bbox；若要真实 temporal tracking，必须另外对
中间帧运行/审核 detector 或 tracker，不能把 endpoint bbox 无依据复制到所有帧。

当使用 `--decode-only` 时，这里绘制的 prefill-query heatmap按定义应与 baseline 相同；真正
干预从生成第一个 decode step 才开始，manifest 会标出这一点。

## 7. 默认输出树

```text
results/attention/<run>/
├── split_manifest.json
├── head_discovery/
│   └── head_ranking.json
├── shards/shard_000/
│   ├── run_manifest.json
│   ├── results.jsonl
│   └── completion.json
├── metrics.json
├── metrics.md
├── artifacts/
│   ├── curves/
│   └── attention_video/
└── logs/
    ├── prepare.log
    ├── rank.log
    ├── experiment_shard_*.log
    ├── metrics.log
    ├── curve.log
    └── video.log
```

## 8. 指标与结论标准

主要 estimand 是同一样本内的连续分数变化：

- `candidate_target - baseline` signed/absolute shift；
- candidate-target 与 same-head wrong-region 的差分（空间特异性）；
- candidate-target 与 low-ranked-head target 的差分（head 特异性）；
- all-head stress control、invalid parse rate、paired coverage；
- observed-example bootstrap 95% CI。

固定分箱 exact accuracy、within-one accuracy、fixed-bin MAE、continuous ordinal MAE 和
interval ordinal MAE 是 generation 完成后才计算的补充指标。准确率依赖离散阈值，不能
替代连续 paired effect。只有 target 效应同时稳定超过 wrong-region 和 low-ranked-head
对照，才足以支持 grounding/head 特异性的因果表述。
