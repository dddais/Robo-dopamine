# Robo-Dopamine 在 RoboRewardBench 上的实验记录

## 1. 实验目的与结论摘要

本实验评估 `Robo-Dopamine-GRM-2.0-8B-Preview` 在 RoboRewardBench test split
上的 zero-shot 端点进度判断能力。Robo-Dopamine 输出连续进度，而 benchmark 标签为
1–5 的离散序数，因此同时报告固定分箱、连续序数和区间序数三种指标，避免只用一种
离散化方式得出结论。

本次 2,831 条样本全部成功，metadata 精确匹配，固定分箱的 23-subset Macro MAE 为
**1.0414**，95% bootstrap CI 为 **[0.9936, 1.0901]**。连续序数 Macro MAE 为
**1.0560**，与固定分箱结论一致；所以当前差距不能主要归因于离散化。四类来源中，
OXE 时间截断最好（固定分箱 MAE 0.7434），OXE 原始成功样本最差（1.7709）。模型的
预测明显向中间收缩，尤其低估 reward=5 的完整成功样本。

## 2. 实验环境与可复现性

实验于 2026-07-22 在以下环境运行：

| 项目 | 设置 |
|---|---|
| 仓库 | `/home/dais/workspace/Robo-Dopamine` |
| Git 基线 revision | `81c9a12bc273085c0fe0ccea793a823a93d71e25` |
| 推理时未提交源码指纹 | `b3f6458385f21d021977764c5df014f8e1419f233416f0625ca309990534b866` |
| conda 环境 | `robo-dopamine` |
| Python | 3.10.20 |
| GPU | NVIDIA A100-SXM4-80GB；本次使用 GPU 0 |
| Driver | 580.82.07 |
| vLLM | 0.11.0 |
| PyTorch | 2.8.0 |
| Transformers | 4.57.0 |
| OpenCV headless | 4.12.0.88 |
| NumPy | 2.2.6 |

模型目录：

```text
/home/dais/workspace/Robo-Dopamine/pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview/
```

模型目录整体 SHA-256 指纹为
`3b4f8d67ee0191bd0985527fc32148676e57c7f756459530f825721422b4fd19`。
该指纹包含目录内的权重、配置、tokenizer 和 Hugging Face cache metadata；单个文件哈希
保存在 `run_manifest.json` 中。

数据目录：

```text
/home/dais/workspace/data/RoboRewardBench/test/
```

`metadata.jsonl` SHA-256：
`3901cea9a7981d2e2d9fde4225a4b2ff0ebf2f4af69fc96972a2cfbe557db0cd`。

预测文件 SHA-256：
`ee119b69f9885f880619ec934ea3ece2ae86c0db13ca9ee85ba59cd5aea0986e`。

完整产物位于：

```text
outputs/roborewardbench/rd8b_forward_test/
├── run_manifest.json
├── predictions.jsonl
└── metrics.json
```

## 3. test 数据的四类组成

分类依据公开 release 的文件名、subset 和人工复核 reward：

1. **RoboArena 自然 rollout**：`robo_arena`/`roboarena` 下的自然成功与失败轨迹。
2. **OXE 时间截断**：文件名满足 `*_attempt_<n>_score_<1-4>.mp4`；文件名中的 score
   与 metadata reward 逐条一致。
3. **OXE 反事实任务**：非时间截断、非 RoboArena，且 reward 为 1–4；它们是成功视频
   与不匹配任务描述构成的反事实负例。
4. **OXE 原始成功样本**：非时间截断、非 RoboArena，且 reward=5。

总体标签组成如下：

| 来源 | 数量 | reward=1 | reward=2 | reward=3 | reward=4 | reward=5 |
|---|---:|---:|---:|---:|---:|---:|
| RoboArena 自然 rollout | 1,000 | 266 | 263 | 174 | 141 | 156 |
| OXE 时间截断 | 791 | 195 | 266 | 170 | 160 | 0 |
| OXE 反事实任务 | 669 | 228 | 112 | 212 | 117 | 0 |
| OXE 原始成功样本 | 371 | 0 | 0 | 0 | 0 | 371 |
| **总计** | **2,831** | **689** | **641** | **556** | **418** | **527** |

OXE 的三个来源在各 subset 中的组成如下。时间截断和反事实任务均覆盖全部 22 个 OXE
subset；原始成功样本覆盖 21 个，不包含 `viola`。

| OXE subset | 时间截断 | 反事实任务 | 原始成功 |
|---|---:|---:|---:|
| `austin_sirius_dataset_converted_externally_to_rlds` | 44 | 23 | 20 |
| `berkeley_autolab_ur5` | 49 | 31 | 20 |
| `berkeley_fanuc_manipulation` | 44 | 27 | 20 |
| `berkeley_mvp_converted_externally_to_rlds` | 31 | 20 | 20 |
| `berkeley_rpt_converted_externally_to_rlds` | 43 | 23 | 20 |
| `bridge` | 36 | 44 | 20 |
| `cmu_play_fusion` | 41 | 31 | 20 |
| `dlr_edan_shared_control_converted_externally_to_rlds` | 4 | 11 | 4 |
| `droid` | 45 | 35 | 20 |
| `fractal20220817_data` | 34 | 46 | 20 |
| `iamlab_cmu_pickup_insert_converted_externally_to_rlds` | 14 | 49 | 20 |
| `jaco_play` | 33 | 47 | 20 |
| `kaist_nonprehensile_converted_externally_to_rlds` | 27 | 14 | 12 |
| `roboturk` | 39 | 38 | 20 |
| `stanford_hydra_dataset_converted_externally_to_rlds` | 66 | 14 | 11 |
| `taco_play` | 28 | 43 | 20 |
| `tokyo_u_lsmo_converted_externally_to_rlds` | 38 | 13 | 20 |
| `ucsd_kitchen_dataset_converted_externally_to_rlds` | 47 | 28 | 20 |
| `ucsd_pick_and_place_dataset_converted_externally_to_rlds` | 38 | 42 | 20 |
| `utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds` | 12 | 41 | 20 |
| `utokyo_xarm_bimanual_converted_externally_to_rlds` | 39 | 28 | 4 |
| `viola` | 39 | 21 | 0 |
| **总计** | **791** | **669** | **371** |

统计可通过以下命令重新生成：

```bash
python -m roborewardbench.data_stats \
  --metadata /home/dais/workspace/data/RoboRewardBench/test/metadata.jsonl
```

## 4. 推理协议

Robo-Dopamine 的输入协议与 RoboRewardBench 并不完全相同，本次采用以下明确适配：

- 模式为 `forward terminal`，每条视频只使用真实首帧和最后一个可解码终帧。
- RoboRewardBench 只有单视角，因此将同一视角复制到 Robo-Dopamine 的三个相机槽位。
- 数据没有独立 goal image，使用 224×224 的中性灰占位图；没有把终帧当 goal，避免标签泄漏。
- 跨视频 batch size 为 8；temperature=0，seed=0。
- 模型输出 `<score>百分比</score>`，除以 100 后裁剪至 `[0,1]` 作为连续进度。
- 不拟合 test 校准器，也没有在 test 标签上选择阈值。
- 指标以 10,000 次、seed=0 的分层 bootstrap 估计 23-subset Macro MAE 置信区间。

完整推理命令：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n robo-dopamine \
  python -m roborewardbench.run_benchmark \
  --dataset-root /home/dais/workspace/data/RoboRewardBench \
  --split test \
  --model /home/dais/workspace/Robo-Dopamine/pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview \
  --mode forward \
  --batch-size 8 \
  --temperature 0 \
  --bootstrap-samples 10000 \
  --output-dir outputs/roborewardbench/rd8b_forward_test
```

在新增来源分类指标后，没有重新推理，只对已保存预测重新计分：

```bash
conda run -n robo-dopamine python -m roborewardbench.score \
  --predictions outputs/roborewardbench/rd8b_forward_test/predictions.jsonl \
  --metadata /home/dais/workspace/data/RoboRewardBench/test/metadata.jsonl \
  --bootstrap-samples 10000 \
  --output outputs/roborewardbench/rd8b_forward_test/metrics.json
```

## 5. 连续输出与离散标签的公平比较

设 Robo-Dopamine 连续输出为 `p∈[0,1]`，RoboRewardBench 标签为 `y∈{1,…,5}`。
本适配器不把单一离散化结果伪装成官方唯一规则，而是并列报告：

| 指标 | 定义 | 解释 |
|---|---|---|
| 固定分箱 MAE | 以 `0.125, 0.375, 0.625, 0.875` 为阈值，将 `p` 映射为 1–5 后计算 MAE | 等价于对序数坐标 `1+4p` 做固定最近整数映射，最接近榜单的离散输出格式 |
| 连续序数 MAE | `|1+4p-y|` | 不舍入，避免刚越过阈值时误差突然跳变 |
| 区间序数 MAE | `1+4p` 落入真实标签的固定量化单元时记 0，否则计算到单元边界的距离 | 不惩罚同一标签单元内的连续差异，对阈值最宽容 |
| validation-only 校准 | 只在 validation 上拟合 isotonic 映射并冻结，再应用于 test | 可修正系统性的刻度偏差，但必须与 raw zero-shot 分开报告 |

固定阈值来自均匀覆盖 `[0,1]` 的五个序数代表点，不使用 test 标签。只报告固定分箱
确实可能对阈值两侧的样本不公平，因此连续序数和区间序数是必要的补充。本次固定分箱
Macro MAE 1.0414、连续序数 1.0560，二者十分接近；区间序数虽降至 0.6492，但四类来源
的相对排序不变。因此模型在本 benchmark 上的主要现象是刻度压缩与输入/任务协议错配，
而不是某个离散阈值碰巧选得不好。

不能使用 test reward 搜索最优阈值或拟合校准器，否则会造成 test leakage。若需要更公平
的离散榜单对比，应先在公开 validation split 上冻结 isotonic 校准器或阈值，然后原样
应用到 test，并同时保留未校准结果。

## 6. 完整结果

有效性检查：

| 检查 | 结果 |
|---|---:|
| 原始/去重记录数 | 2,831 / 2,831 |
| 有效/无效预测数 | 2,831 / 0 |
| invalid rate | 0 |
| subset 数 | 23 / 23 |
| ID、task、reward、subset 与 metadata 精确匹配 | 是 |
| `official_comparable` | `true` |

总体指标：

| 指标 | 23-subset Macro MAE | 95% CI | Micro MAE |
|---|---:|---:|---:|
| 固定分箱 | **1.0414** | [0.9936, 1.0901] | 1.0240 |
| 连续序数 | 1.0560 | [1.0139, 1.0994] | 1.0319 |
| 区间序数 | 0.6492 | [0.6103, 0.6893] | 0.6311 |

固定分箱下 RoboArena MAE 为 0.9770；22 个 OXE subset 等权 Macro MAE 为 1.0443。

按四类数据来源分解：

| 来源 | 数量 | 固定分箱 MAE | 连续序数 MAE | 区间序数 MAE |
|---|---:|---:|---:|---:|
| RoboArena 自然 rollout | 1,000 | 0.9770 | 0.9870 | 0.5963 |
| OXE 时间截断 | 791 | **0.7434** | **0.7507** | **0.3799** |
| OXE 反事实任务 | 669 | 1.0120 | 1.0265 | 0.6038 |
| OXE 原始成功样本 | 371 | **1.7709** | **1.7622** | **1.3098** |

固定分箱下表现最好的三个 subset：

| subset | MAE |
|---|---:|
| UCSD Pick and Place | 0.5400 |
| UTokyo PR2 Tabletop | 0.6438 |
| Austin Sirius | 0.7241 |

表现最差的四个 subset：

| subset | MAE |
|---|---:|
| Tokyo LSMO | 1.7324 |
| Taco Play | 1.3626 |
| Stanford HYDRA | 1.3516 |
| Jaco Play | 1.2700 |

各真实标签对应的平均预测离散标签为：

| 真实标签 | 平均预测标签 |
|---:|---:|
| 1 | 1.6473 |
| 2 | 2.0686 |
| 3 | 2.5917 |
| 4 | 2.8900 |
| 5 | 3.2770 |

## 7. 结果分析

### 7.1 模型输出存在向中间收缩

真实 reward 从 1 增至 5 时，平均预测只从 1.65 增至 3.28。reward=1 被高估，reward=4/5
被明显低估。最直接的证据是 371 条 OXE 原始成功样本全部为 reward=5，但固定分箱 MAE
达到 1.7709；相反，分布在 reward 1–4 的 OXE 时间截断只有 0.7434。阈值宽容的区间
指标仍分别为 1.3098 和 0.3799，说明差异不是固定分箱制造出来的。

### 7.2 端点-only 会丢失过程证据

RoboRewardBench 的 reward 是 end-of-episode progress，但视频中一些任务仅凭首末帧难以
判断动作过程、短暂接触、插入是否真正完成或中间是否发生失败。Robo-Dopamine 本身是
process reward model，本次 forward 端点协议没有充分利用其逐步判断能力。建议把
incremental 1 FPS 作为下一项消融，与当前 forward 结果并列，而不是替换本结果。

### 7.3 输入协议存在不可避免的域偏移

Robo-Dopamine 原生期望多相机状态和 goal image；benchmark 提供单视角视频和任务文本。
复制单视角不能补充被遮挡信息，中性灰 goal 也不包含目标外观。这种适配保证不泄漏
终帧，但会使模型缺少训练时可用的条件信息。它也是本结果不能被解释为模型原生场景
能力上限的主要原因。

### 7.4 与 RoboReward 8B 的结果只能谨慎对照

截至 2026-07-22，[RoboRewardBench HELM v0.0.1 leaderboard](https://crfm.stanford.edu/helm/robo-reward-bench/v0.0.1/#/leaderboard)
中 Qwen3-VL 8B RoboReward 的 23 个 subset MAE 等权平均为 0.6647（榜单约记 0.665）。
当前固定分箱 Macro MAE 1.0414 高约 0.3767，但这不是严格同协议的模型对比：RoboReward 直接针对
离散 end-of-episode reward 和视频输入训练，而当前实验是 Robo-Dopamine 的 zero-shot
连续输出适配，且只看端点、复制单视角并使用空白 goal。0.665 只能作为 benchmark
上下文，不能据此把差值全部归因于模型架构或训练质量。

## 8. 运行中发现并修复的问题

真实 smoke test 和完整测评前检查发现了以下问题，当前实现均已处理：

- 模型实际会输出小数百分比，例如 `<score>+66.7%</score>`；解析器已支持有符号整数和
  小数，并严格拒绝标签外文本及越界值。
- 少量 MP4 的 `CAP_PROP_FRAME_COUNT` 比实际可解码帧数多 1；终帧读取会向后寻找最后
  一个可解码帧，同时保证不越过前一采样状态。
- vLLM batch 返回顺序不应被隐式依赖；现在通过唯一输入 ID 对齐并校验缺失、重复和
  意外输出。
- forward 模式已实现跨视频 batching，而不是每个视频单独调用模型。
- 原示例会在 import 时强制设置 `CUDA_VISIBLE_DEVICES=0`；该覆盖已移除，GPU 由外部
  命令或调度器选择。
- 断点续跑会校验代码、metadata、模型文件、依赖版本和关键参数指纹，拒绝把不同实验
  的旧预测混入当前输出。
- 审查发现只校验 metadata 不能检测视频文件被替换；新运行的 manifest 还会记录所有
  所选视频的逐文件 SHA-256 和组合指纹。
- batch 中某个样本解析失败曾会连带影响其后的正常样本；现在生成/ID 对齐错误仍按整批
  处理，而输出解析错误隔离到对应样本。
- `official_comparable` 还要求 2,831 条样本全部有效，且 ID、task、reward、subset 与
  指定 metadata 精确匹配。

## 9. 后续建议

1. 在同一模型和 test split 上运行 incremental 1 FPS 消融，检验过程信息能否改善
   reward=4/5 和 OXE 原始成功样本。
2. 在 validation split 上拟合并冻结 isotonic 校准器，再报告 raw 与 calibrated 两组
   test 结果；不要在 test 上优化阈值。
3. 若能构造真实目标图或与训练一致的多视角输入，应单独作为新协议运行并使用新的输出
   目录，避免与本次 zero-shot 端点结果混合。
4. 除总体 Macro MAE 外，持续报告四类来源指标、标签条件均值和 invalid rate。当前总体
   分数掩盖了 OXE 时间截断与原始成功样本之间超过 1.0 MAE 的差异。
