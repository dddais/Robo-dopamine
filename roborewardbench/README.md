# Robo-Dopamine 的 RoboRewardBench 测评

本目录包含两条必须区分的评测流程：

| 流程 | 入口 | 是否需要 grounding | 主要问题 |
|---|---|:---:|---|
| 标准 Robo-Dopamine benchmark | `python -m roborewardbench.run_benchmark` | 否 | 原始 GRM 在完整 RoboRewardBench 上预测得怎样 |
| target-bbox attention intervention | `python run_targetbbox_roborewardbench.py` | 是，且正式样本必须人工审核 | 选定 heads 对目标 bbox 的 attention-logit 干预是否因果改变输出 |

不要把第二条 19-example 人工审计子集的结果写成完整 benchmark 得分，也不要为了运行标准
评测先做 grounding。所有命令都从仓库根目录执行：

```bash
cd /home/dais/workspace/Robo-Dopamine
conda activate robo-dopamine
```

目录职责：

```text
roborewardbench/
├── run_benchmark.py, score.py, metrics.py, data.py
│   └── 标准 benchmark 适配、推理和计分
├── dopamine_eval/
│   └── instruction target parse + GroundingDINO + 人工审核
└── attention_mask/
    └── frozen bbox head discovery、intervention、curve 和 heatmap
```

详细入口：

- 本 README 第 1–7 节：标准 benchmark；
- [`attention_mask/README.md`](attention_mask/README.md)：target-bbox attention 实验；
- [`dopamine_eval/README.md`](dopamine_eval/README.md)：grounding pipeline；
- [`dopamine_eval/MANUAL_AUDIT.md`](dopamine_eval/MANUAL_AUDIT.md)：人工逐条审核哪些文件、如何填写 label。

当前机器上的本地资源为：

- 模型：`pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview/`
- 数据：`/home/dais/workspace/data/RoboRewardBench/`
- test metadata：`/home/dais/workspace/data/RoboRewardBench/test/metadata.jsonl`

如需访问 Hugging Face，可按需设置代理：

```bash
export http_proxy=http://127.0.0.1:7898
export https_proxy=http://127.0.0.1:7898
```

本地模型和已下载数据的评测过程不依赖网络。

## 1. 连续值与 1–5 标签如何比较

RoboRewardBench 要求模型直接预测离散的 `{1,2,3,4,5}`，而 Robo-Dopamine 输出
连续进度 `p∈[0,1]`。官方没有规定连续值的离散化方式，因此本实现同时报告四组结果：

| 输出字段 | 定义 | 用途 |
|---|---|---|
| `benchmark_compatible_fixed_bin` | 用 `0.125, 0.375, 0.625, 0.875` 将 `p` 映射到 1–5，再计算 23 个 subset 等权的 Macro MAE | 与榜单离散 MAE 最接近的主结果，但离散化规则属于本适配器 |
| `continuous_ordinal` | 直接计算 `|1+4p-y|`，不舍入 | 避免离散边界跳变 |
| `interval_ordinal` | `1+4p` 落在真实标签对应的量化区间内时误差为 0 | 对阈值更稳健的补充指标 |
| `validation_calibrated` | 只在 validation 上拟合单调 isotonic 映射，冻结后用于 test | 分析模型刻度与标签刻度的系统偏差 |

计分结果还会在 `discrete_classification` 中报告固定分箱后的精确准确率、±1 容差
准确率、预测分数分布、混淆矩阵以及高估/低估率。准确率适合解释单一标签切片，但完整
benchmark 仍以 MAE 为主，因为 MAE 会区分“错 1 级”和“错 4 级”。

建议把 `benchmark_compatible_fixed_bin.macro_mae` 作为主要可比结果，同时报告
`continuous_ordinal`、`interval_ordinal` 和 `invalid_rate`。不要用 test 标签选择阈值或拟合
校准器。`official_comparable=true` 只有在 2,831 条预测全部有效，并且 ID、task、reward、
subset 与指定 test metadata 精确一致时才会出现。

## 2. 数据下载与四类数据统计

尚未下载数据时可运行：

```bash
python -m roborewardbench.download_data \
  --output /home/dais/workspace/data/RoboRewardBench \
  --splits test
```

统计 test 中 RoboArena 自然 rollout、OXE 时间截断、OXE 反事实任务和 OXE 原始成功
样本：

```bash
python -m roborewardbench.data_stats \
  --metadata /home/dais/workspace/data/RoboRewardBench/test/metadata.jsonl \
  --output outputs/roborewardbench/test_data_stats.json
```

分类依据与本次数据的详细统计见 [exp_record.md](exp_record.md)。

## 3. 真实模型 smoke test

`forward` 模式只比较真实首帧和真实终帧。RoboRewardBench 是单视角视频，因此同一帧会
复制到 GRM 的三个相机槽位；数据没有真实 goal image，所以使用中性灰占位图。这个协议
避免把待评估的终帧当作完成目标，但相较于视频原生模型仍有信息限制。

```bash
CUDA_VISIBLE_DEVICES=0 python -m roborewardbench.run_benchmark \
  --dataset-root /home/dais/workspace/data/RoboRewardBench \
  --split test \
  --model pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview \
  --mode forward \
  --batch-size 8 \
  --temperature 0 \
  --limit 20 \
  --bootstrap-samples 0 \
  --output-dir outputs/roborewardbench/rd8b_forward_smoke
```

运行器不会修改 `CUDA_VISIBLE_DEVICES`。上面的命令明确使用 GPU 0；也可以由外部调度器
选择其他 GPU。`--batch-size` 在 forward 模式下会跨视频合批。

## 4. 完整 test 测评

```bash
CUDA_VISIBLE_DEVICES=0 python -m roborewardbench.run_benchmark \
  --dataset-root /home/dais/workspace/data/RoboRewardBench \
  --split test \
  --model pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview \
  --mode forward \
  --batch-size 8 \
  --temperature 0 \
  --bootstrap-samples 10000 \
  --output-dir outputs/roborewardbench/rd8b_forward_test
```

主要产物：

```text
run_manifest.json   数据、代码、依赖和模型文件的指纹及协议
predictions.jsonl   每条样本的原始输出、连续进度、标签和失败信息
metrics.json        Macro/Micro/per-subset 指标及 bootstrap 置信区间
```

预测会逐条 `fsync`。相同命令可断点续跑；若代码、metadata、所选视频内容、模型文件、
关键依赖或推理参数发生变化，运行器会拒绝混合旧结果。`--retry-invalid` 只重试失败样本。

## 5. 对已有预测重新计分

计分时传入准确的 metadata，才能证明结果具备完整可比性：

```bash
python -m roborewardbench.score \
  --predictions outputs/roborewardbench/rd8b_forward_test/predictions.jsonl \
  --metadata /home/dais/workspace/data/RoboRewardBench/test/metadata.jsonl \
  --bootstrap-samples 10000 \
  --output outputs/roborewardbench/rd8b_forward_test/metrics.json
```

## 6. Incremental 消融实验

Incremental 模式按相邻状态预测 hop，再用 Robo-Dopamine 的递推公式累计进度。1 FPS 且
不限制状态数：

```bash
CUDA_VISIBLE_DEVICES=0 python -m roborewardbench.run_benchmark \
  --dataset-root /home/dais/workspace/data/RoboRewardBench \
  --split test \
  --model pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview \
  --mode incremental \
  --frame-sampling 1fps \
  --max-states 0 \
  --batch-size 8 \
  --temperature 0 \
  --output-dir outputs/roborewardbench/rd8b_incremental_test
```

固定均匀采样 8 个状态时改用：

```text
--frame-sampling uniform --max-states 8
```

本实现不启用 backward：数据没有真实完成目标；把终帧用作 goal 会直接泄漏待评估状态。

## 7. 可选的 validation-only 校准

必须先以与 test 完全相同的推理协议生成 validation 预测：

```bash
CUDA_VISIBLE_DEVICES=0 python -m roborewardbench.run_benchmark \
  --dataset-root /home/dais/workspace/data/RoboRewardBench \
  --split val \
  --model pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview \
  --mode forward \
  --batch-size 8 \
  --temperature 0 \
  --output-dir outputs/roborewardbench/rd8b_forward_val

python -m roborewardbench.calibrate \
  --predictions outputs/roborewardbench/rd8b_forward_val/predictions.jsonl \
  --output outputs/roborewardbench/rd8b_forward_calibration.json

python -m roborewardbench.score \
  --predictions outputs/roborewardbench/rd8b_forward_test/predictions.jsonl \
  --metadata /home/dais/workspace/data/RoboRewardBench/test/metadata.jsonl \
  --calibration outputs/roborewardbench/rd8b_forward_calibration.json \
  --output outputs/roborewardbench/rd8b_forward_test/metrics_calibrated.json
```

原始 zero-shot 与 validation-calibrated 结果应分开报告，后者不能写成 zero-shot。
