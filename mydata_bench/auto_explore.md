# Qwen3-VL / RoboReward gaze-head 自动研究与 Debug 记录

## Material Passport

- Schema: ARS Material Passport 9（实验执行/验证记录）
- 状态: `COMPLETE`
- 开始时间: 2026-09-01（Asia/Shanghai）
- 用户指定依据:
  - `mydata_bench/exp_plan_crossmodel.md`
  - `mydata_bench/01_Codex_Qwen3VL_RoboReward_Debug_Guide_Concise.md`
- 数据: `/home/dais/workspace/data/mydata_v2/new` 与冻结 cohort `results/mydata_bench/cohorts/auto_grounded_v2`
- 新配置目录: `mydata_bench/configs/v2_debug`
- 新结果目录: `results/mydata_bench/experiments_v2_debug`
- 约束: 不执行 Git 操作；不删除或覆盖已有数据/结果；代码只作增量式、默认关闭的诊断扩展。
- Verification Status: `VERIFIED`（只适用于调用链、backend、zero-bias、hook、processor 映射与确定性复跑）；模型机制与总体效果结论为 `ANALYZED/SUPPORTED`，不外推为总体因果定论。

## 0. 问题定义与完成判据

已有跨模型实验把 8 帧 native video 改造成 8 张独立 image，并比较 text→images、images→text、GRM 式 interleaved 三种构造；在每种构造内重新做 gaze-head ranking，再对 target token 施加正 bias、对指定 non-target visual token 施加负 bias。用户观察到 GRM 增益明显，而 Qwen3-VL-8B / RoboReward-8B 的效果弱或不稳定。

本轮不以“某个离散指标偶然提高”作为完成，而按指定指南回答四个因果链问题：

1. **干预是否生效**：强 bias 后目标区域 attention mass 是否明显增加；zero bias 是否逐项复现 baseline。
2. **区域是否对齐**：processor 展开后的 image/video span、`grid_thw`、空间 merge、时间/视角顺序和逐帧 bbox 是否正确。
3. **对象理解是否改变**：模型对 instructed object、实际操作对象及二者一致性的 probe 是否改变。
4. **reward 是否使用变化**：候选 1–5 的完整序列 log-prob、`E[r]`、成功/反事实 margin 是否改变，而不只比较最终生成整数。

## 1. 实际推理调用链（修改前审计）

### 1.1 Baseline

```text
run_roboreward_eval.py / run_qwen_eval.py
→ CLI
→ roboreward_eval.runner / qwen_eval.runner
→ load_episodes + 均匀抽取 8 张独立 image（或 native video）
→ checkpoint AutoProcessor.apply_chat_template
→ Qwen3VLForConditionalGeneration.generate
→ decode 文本
→ parse_native_score("ANSWER: <1-5>")
→ metrics / exp_record
```

Baseline loader 没有显式指定 `attn_implementation`；checkpoint config 中该字段为 `null`，因此由 Transformers 运行时选择默认 backend。此点后续需用已实例化模块确认，不能只凭配置推断。

### 1.2 Attention ranking / steering

```text
run_{qwen,roboreward}_attention.py
→ qwen_eval.attention_cli
→ qwen_eval.attention_experiment
   prepare-ranking → validate-ranking → rank
   prepare-cohort → steer → score
→ QwenAttentionRuntime
→ AutoProcessor 展开 prompt/media token，绑定 image/video span 与 grid_thw
→ attn_implementation="eager" 的 Qwen3-VL decoder
→ rank: forward hook 读取最后 prompt query 对 bbox token 的 attention mass
→ steer: decoder.self_attn forward-pre-hook 给 attention_mask 加 pre-softmax bias
→ model.generate
→ 只解析生成的离散 ANSWER: 1–5
→ score 时才 join labels
```

RoboReward attention 入口是薄适配器，实际复用同一 `QwenAttentionRuntime`。两个 checkpoint 均为 36 层、32 个 query heads、8 个 KV heads；当前 ranking/steering 编号属于 **query head**，没有把 32 个 query head 错当成 8 个 KV head，但后续仍需验证每 checkpoint 的独立 ranking 及层/head 范围。

### 1.3 当前 hook 覆盖范围

- 旧 crossmodel 配置统一使用 `steering_query_scope: all`。
- `all` 会对整次 prefill 的所有 query row，以及带 cache 的每个 decode call 都加同一 key bias。
- 代码现已支持 `prefill`、`last_prompt`、`decode`，但旧实验没有做指定指南要求的阶段对照。
- hook diagnostics 记录调用次数和 applied/skipped calls，但旧结果没有记录 **干预后的 attention mass**，所以仅凭 reward 改变不能证明 (A(R)) 上升。

### 1.4 当前 reward 读取

- Qwen/RoboReward 都调用 `generate()` 并严格解析 `ANSWER: <1-5>`。
- 没有直接计算候选答案完整序列 log-prob，也没有保存 `p(r)`、`E[r]` 或 high-vs-low margin。
- 因而旧结果把连续但未跨 argmax 边界的 reward 变化全部视为“无变化”，也无法分辨 prompt/template 导致的评分 token 偏移或多 token 候选问题。

## 2. 修改入口与实际修改

修改前按上述范围声明；实现时选择了风险更低的独立诊断入口，未改动旧 ranking/steering 默认路径：

1. 新增 `mydata_bench/gaze_debug.py` 与 `run_gaze_debug.py`，复用 frozen cohort、`QwenAttentionRuntime`、processor span 和 steering hooks，独立记录 attention trace、完整 1–5 候选序列 log-prob、`E[r]`、high-low margin、object probe、matched-random heads、逐 token overlay。
2. 新增 `mydata_bench/configs/v2_debug/*.yaml` 与 `results/mydata_bench/experiments_v2_debug/*`；所有条件均为显式 debug 配置，旧实验默认值不变。
3. `qwen_eval/runner.py` 与 `roboreward_eval/runner.py` 新增可选 `attn_implementation`，并把实例化后的 backend 写进 input diagnostics；未配置时仍保持历史自动选择行为。
4. 新增 `generation_conditions` 和 cached-autoregressive reward scorer，用于把 prompt prefill、首个候选 token 与后续 decode 真正分开；默认关闭。
5. 新增 `tests/test_gaze_debug.py`、`tests/test_attention_backend_config.py`；没有 Git 操作，没有删除或覆盖旧数据/旧结果。

## 3. 初始静态证据

### 3.1 运行环境

- Torch `2.8.0+cu128`，Transformers `4.57.0`，CUDA `12.8`。
- 允许 GPU 0/1/2 均为空闲 A100-SXM4-80GB（初次盘点时显存占用 4 MiB，利用率 0%）。
- Qwen3-VL-8B-Instruct 与 RoboReward-8B 都是 `Qwen3VLForConditionalGeneration`，36 layers / 32 Q heads / 8 KV heads，image token id 151655，video token id 151656，spatial merge size 2。

### 3.2 旧实验不能直接证明“方法有效”的原因

旧 steering 确实经常降低 MAE/提高总准确率，但预测整体向低分移动，同时成功样本准确率常下降。例如：

| 实验 | 条件 | MAE | 总准确率 | suc 准确率 | fail 准确率 |
|---|---:|---:|---:|---:|---:|
| RoboReward text→images | baseline | 1.4965 | 17.49% | 48.13% | 3.29% |
| 同上 | target k32 | 1.3452 | 20.45% | 23.51% | 19.03% |
| Qwen text→images | baseline | 1.5934 | 28.25% | 84.33% | 2.25% |
| 同上 | target k64 | 1.2506 | 46.69% | 75.37% | 33.39% |
| RoboReward text→images all-frame | target k32 | 0.9716 | 52.25% | 45.52% | 55.36% |

这首先显示的是明显的 **global score shift / calibration change**，不是已经验证的 instruction-object binding。必须比较同视频 suc/counterfactual 的 raw reward margin，并用 wrong-region、random-head、object probe 排除非特异性低分漂移。

## 4. 指南检查清单与状态

| 指南项 | 状态 | 当前证据/缺口 |
|---|---|---|
| 真实 inference path | 已验证 | 普通 baseline=`sdpa`，attention runtime=`eager`；实际 backend 已记录 |
| zero-bias = baseline | 已验证 | 两模型各 8/8 token 序列完全一致；40 个候选/模型最大 log-prob 差 `0.0` |
| strong bias 提高 A(R) | 已验证 | bias=12 时 last-prompt 与首个生成 token 的目标 mass 接近饱和 |
| prefill/decode/both | 已验证 | attention、生成、cached-autoregressive reward、probe 均完成阶段对照 |
| media token / grid_thw | 已验证 | 12 个 image-sequence run 各 34/34 通过；另生成 overlay |
| 逐帧 bbox/object tube | 已验证 | 8 个 source index 与 tracking index 逐帧一致，末帧包含；语义分辨率仍偏粗 |
| checkpoint-specific heads / Q-KV | 已验证 | 两 checkpoint 独立 ranking；32 Q / 8 KV；matched-random 层分布完全匹配 |
| query 选择 | 已分析 | 指令词因 causal mask 对后置图像 mass=0；reward anchor、末 prompt、decode query 可见图像 |
| 完整候选 log-prob | 已验证 | 同时保存单 forward 与 cached-autoregressive 两种完整序列读出 |
| object probe | 已完成 | Qwen/RoboReward 同批 8 条；GRM 未测，明确记为缺失 |
| 三层诊断 | 已完成 | 见第 12 节同表对照 |
| evidence-preserving | 已完成 | bias=6/12 均与 target-vs-all 对照 |
| 最小矩阵 | 已完成 | 8 条/模型，2 个独立视频、6 个同视频反事实；只作机制 debug，不作总体评测 |

## 5. 当前假设（尚非结论）

- H1（高优先级，实现/测量缺口）：干预可能生效，但旧记录只看离散生成分数，遗漏了连续 reward leverage；必须用 attention delta 与候选序列 log-prob 验证。
- H2（高优先级，机制/设计）：`all` query + 对所有 non-target visual token 的强负 bias 会删除夹爪、接触、rival object、destination 和运动证据，产生非特异性低分漂移。
- H3（高优先级，head/query）：按最后 prompt token的 raw bbox mass 排 head 选择的是 salience heads，不一定是 reward-causal / instruction-contrastive heads；top-k 越大时全局漂移更明显与此一致。
- H4（中优先级，位置）：独立 image 的 span/grid 主契约看起来有防错检查，但必须通过 overlay 和 tracking-source frame 误差验证像素到 token 的实际几何对齐。
- H5（中优先级，阶段）：短答案的主要因果杠杆可能在 prefill/首评分 token；全 prefill+全 decode 的重复 exposure 使有效剂量依生成长度和调用次数变化。

后续所有结论都将标注为 `OBSERVED`、`SUPPORTED` 或 `VERIFIED`，避免把代码阅读或一次运行当作因果证明。

## 6. 真实路径中的实现问题

### 6.1 Baseline 与 attention runtime 的 backend 不一致（`VERIFIED`）

实例化模型后确认：普通 Qwen/RoboReward baseline 在当前 Transformers 版本下选择 `sdpa`，而 `QwenAttentionRuntime` 为读取 attention weights 强制 `attn_implementation="eager"`。旧 checkpoint config 的字段为 `null`，不能据此断言 backend 相同。

对四组相同 `example_id` 做 join；视频 SHA 全部一致，但普通 baseline 与 attention 流水线内部 `condition=baseline` 的离散输出并不一致：

| 模型/输入 | 不同输出 | 比例 | video SHA 不同 |
|---|---:|---:|---:|
| RoboReward text→images | 72/846 | 8.51% | 0 |
| RoboReward images→text | 28/846 | 3.31% | 0 |
| Qwen text→images | 54/846 | 6.38% | 0 |
| Qwen images→text | 31/846 | 3.66% | 0 |

因此这是一个真实的**实验比较 bug**：不能把普通 `sdpa` baseline 与 `eager` attention baseline 当成完全相同的零点。它不表示 steering hook 错误；同一 attention run 内的 eager baseline 与 eager steering 仍可比较。

修复采用默认兼容方式：两个 baseline runner 均接受可选 `attn_implementation`，记录实际 `attention_backend`；未配置时旧默认不变。今后用于 attention 因果对照的 baseline 必须显式设为 `eager`。

### 6.2 Debug 阶段 scope 传递 bug（`VERIFIED/FIXED`）

第一次补齐阶段 reward/probe 时，三个阶段结果完全相同。代码追踪发现新诊断器的 `reward_sequence_logprobs()` 和 `generate_probe()` 将 `query_scope` 写死为 `all`。修复后 scope 从 condition 逐层传入。

- 有问题的新增产物保留在 `qwen_interleaved_phase_complete`、`roboreward_interleaved_phase_complete`，标记为 **INVALID_FOR_PHASE_COMPARISON**，没有覆盖或删除。
- 修复后的 probe/generation/attention 在 `*_phase_corrected`。
- 单次完整 teacher-forcing forward 会把候选序列整体视为 prefill，无法触发真实 decode；因此又新增默认关闭的 cached-autoregressive scorer，在 `*_phase_autoregressive` 中逐 token 复现 prefill/decode。
- cached 与 non-cached bf16 logits 存在数值路径差异，故阶段因果只在 cached 路径内部作 paired contrast；两种路径的 zero-bias 都各自精确相等。

这是本轮 auto-debug 自己发现并公开保留的测量错误，不影响此前 strong-bias attention trace、生成阶段对照及主流水线结果。

## 7. 最小诊断实验设计

| 项目 | 设置 |
|---|---|
| 模型 | Qwen3-VL-8B-Instruct；RoboReward-8B |
| 输入 | GRM 式 interleaved 8 张独立 image，逐帧 tracking bbox |
| 样本 | 8 条记录：2 条 suc、6 条同视频 counterfactual fail；实际只有 2 个独立视频 |
| Head | 每 checkpoint 自己的 consensus ranking top-8；32 Q heads / 8 KV heads |
| Random control | 固定 seed，同层数量分布、排除 gaze heads |
| 剂量 | bias=12（强干预 sanity）与 bias=6（非饱和操作剂量） |
| 条件 | baseline、zero、prefill、decode、both、last-prompt、random、target-vs-all、evidence-preserving |
| Readout | 逐 query `A(R)`、生成 token、完整候选 log-prob、`p(r)`、`E[r]`、high-low margin、object probe、overlay |

`Δ_pair` 定义为同批 `mean(E[r]_suc) - mean(E[r]_fail)`；`C_reward=Δ_pair_int-Δ_pair_base`。由于只有两个独立视频，下面是机制 sanity，不是总体效应估计。

## 8. 干预是否真正生效

### 8.1 Zero-bias（`VERIFIED`）

| 模型 | 生成 token 完全一致 | 候选 sequence-logprob 最大绝对差 |
|---|---:|---:|
| Qwen | 8/8 | 0.0 |
| RoboReward | 8/8 | 0.0 |

cached-autoregressive 路径也分别得到最大差 `0.0`。这排除了“只要安装 hook 就改变结果”的实现副作用。

### 8.2 Strong bias 的 `A(R)`（`VERIFIED`）

bias=12、top-8 gaze heads、两个独立 4-sample run 合并：

| 模型 | last prompt baseline → intervention | 首个生成 token query baseline → intervention |
|---|---:|---:|
| Qwen | 0.00961 → 0.99811 | 0.000378 → 0.89523 |
| RoboReward | 0.01372 → 0.98718 | 0.000412 → 0.87616 |

hook 并非 no-op；目标注意力被推到近饱和。bias=6 仍明显但未完全饱和：

| 模型/条件 | last prompt `A(R)` | 首生成 query `A(R)` |
|---|---:|---:|
| Qwen baseline | 0.00961 | 0.000378 |
| Qwen gaze target-vs-all | 0.64207 | 0.12240 |
| Qwen gaze evidence-preserving | 0.61596 | 0.11999 |
| Qwen matched-random | 0.31249 | 0.03277 |
| Robo baseline | 0.01372 | 0.000412 |
| Robo gaze target-vs-all | 0.68251 | 0.13381 |
| Robo gaze evidence-preserving | 0.65538 | 0.13178 |
| Robo matched-random | 0.29931 | 0.05855 |

gaze heads 的 mass 增幅大于层分布匹配 random heads，但 random 也会在强 bias 下显著增大目标 mass；因此“mass 能被强推高”本身不是 reward-causal head 的充分条件。

### 8.3 Hook 确实覆盖 prefill 与 decode（`VERIFIED`）

旧 interleaved steering 记录中，每个选中 layer 通常有 6 次调用：1 次 prefill、5 次 decode，`applied_calls=6`、`skipped_calls=0`。新阶段 trace 进一步显示：

- prefill-only：last-prompt/reward-anchor mass 上升，decode mass 保持 baseline；
- decode-only：prompt rows 保持 baseline，首个及后续生成 query mass 上升；
- last-prompt：只改变 prefill 最后一行；
- both：两阶段均改变。

## 9. Query 与 causal mask 诊断

interleaved prompt 中任务指令仍在所有图像之前。实测 instruction target phrase、object category、attribute query 对后置视觉 token 的 `A(R)` 均精确为 `0.0`；这是 causal mask 的必然结果，不是索引 bug。

可见视觉证据的 query 是：图像之后的 reward anchor、最后 prompt token，以及生成/评分 token。因而旧方法按“最后 prompt 对 bbox raw mass”排 head，只能说明这些 heads 在该 query 上看目标区域，不能说明它们编码 instruction-object binding 或直接控制 reward。

cached-autoregressive bias=6 连续读出给出阶段局部效应：

| 模型/阶段 | `Δ_pair` | 相对 baseline 的 `C_reward` |
|---|---:|---:|
| Qwen baseline | 3.2013 | — |
| Qwen prefill-only | 3.3731 | +0.1718 |
| Qwen decode-only | 3.3583 | +0.1571 |
| Qwen both | 3.4147 | +0.2134 |
| Qwen last-prompt-only | 3.2095 | +0.0082 |
| RoboReward baseline | 2.6836 | — |
| RoboReward prefill-only | 2.7134 | +0.0298 |
| RoboReward decode-only | 2.7231 | +0.0395 |
| RoboReward both | 2.8363 | +0.1527 |
| RoboReward last-prompt-only | 2.6865 | +0.0029 |

结论：只干预最后 prompt 行几乎没有 reward leverage；Qwen 的 prefill 与 decode 都有小幅贡献，RoboReward 单阶段很弱、both 才出现较明显的非线性交互。这支持“query/phase 选择不对”是旧方法弱的重要机制原因。

## 10. 区域到 token 映射

### 10.1 Span/时序契约（`VERIFIED`）

对 crossmodel 的 12 个 image-sequence attention runs 系统扫描：每个 run 的 34 条 ranking 样本均为 34/34 `status=ok`、34/34 有 8 个独立 image spans、34/34 target positions 位于 target span、34/34 末图包含原视频末帧，总计 408/408 通过。

all-frame 样本的 8 个 source frame index 与 tracking frame index 逐项一致；没有把 image/video token 混用，没有把分隔符/时间文本当视觉 token，也没有把首帧 bbox 复制到所有帧。gross off-by-one、时序反转或末帧丢失未发现。

### 10.2 几何正确但语义过粗（`OBSERVED`）

每个 640×480 image 在 `max_pixels=50176` 下输出 raw grid `[1,12,16]`；spatial merge=2 后只有 `6×8=48` 个视觉 token，每格约 80×80 像素。约 50×50 的 bbox 常与 4 格相交，二值 “any overlap” 会实际增强约 160×160 的区域，容易同时包含 rival object、夹爪、接触区或背景。

四个 bias=12 小 run 各生成 32 张 overlay（共 128 张）。绿色为原 bbox，红色为最终 token cells；例如：

`results/mydata_bench/experiments_v2_debug/qwen_interleaved_minimal/overlays/fail__ljx_lfz_task_1_1__1__image_t7.png`

processor 分辨率审计：

| `max_pixels` | merge 后 grid | 每图 token |
|---:|---:|---:|
| 50,176 | 6×8 | 48 |
| 100,352 | 8×11 | 88 |
| 200,704 | 12×16 | 192 |
| 401,408 | 15×20 | 300 |

所以“区域是否对齐”的精确回答是：**span、时间、bbox→cell 几何契约正确；当前 tokenization 的语义空间精度不足。** 提高分辨率后必须重新 ranking，不能沿用旧 head/position 结果。

## 11. Reward 与 object probe

### 11.1 主 reward 结果：有小幅 gaze-specific leverage，但伴随全局 shift（`SUPPORTED`）

cached-autoregressive bias=6：

| 模型/条件 | mean `E[r]` | `Δ_pair` | `C_reward` |
|---|---:|---:|---:|
| Qwen baseline | 2.5990 | 3.2013 | — |
| Qwen gaze target-vs-all | 2.3750 | 3.4147 | +0.2134 |
| Qwen gaze evidence-preserving | 2.3875 | 3.4110 | +0.2097 |
| Qwen matched-random | 2.5988 | 3.2016 | +0.0003 |
| Robo baseline | 2.9800 | 2.6836 | — |
| Robo gaze target-vs-all | 2.8440 | 2.8363 | +0.1527 |
| Robo gaze evidence-preserving | 2.9079 | 2.7546 | +0.0711 |
| Robo matched-random | 3.2984 | 2.2668 | -0.4168 |

gaze 与 random 的方向明显不同，支持 top-8 gaze heads 在这 8 条上具有局部 reward leverage；但 target-vs-all 的 mean `E[r]` 同时下降，增益仍包含“fail 降得更多”的 calibration/global shift，并非对象绑定全面改善。

单-forward 完整序列读出的剂量比较也显示 bias=6 优于近饱和 bias=12：

| 模型 | bias | target-vs-all `C_reward` | evidence-preserving `C_reward` | random `C_reward` |
|---|---:|---:|---:|---:|
| Qwen | 6 | +0.2063 | +0.2145 | -0.0287 |
| Qwen | 12 | +0.0421 | +0.0696 | -0.2494 |
| RoboReward | 6 | +0.1794 | +0.0946 | -0.3722 |
| RoboReward | 12 | +0.0739 | +0.0534 | -0.5413 |

因此后续不应默认使用饱和强度；moderate-dose sweep 更合理。

### 11.2 Evidence-preserving 对照是 mixed result（`OBSERVED`）

- Qwen bias=6：evidence-preserving 减少平均分下移，`C_reward` 与 target-vs-all 基本相同（+0.2097 vs +0.2134 cached；单-forward 略高）。
- RoboReward bias=6：evidence-preserving 将平均分下移约减半，但 `C_reward` 也由 +0.1527 降到 +0.0711。
- bias=12 时两者因 attention 近饱和而非常接近。

故 H2 只得到部分支持：保留证据能减少全局分数扰动，但当前 8 条不能证明它普遍提高 reward margin；RoboReward 的 margin 增益反而部分依赖压低 non-target visual evidence。

### 11.3 离散生成会掩盖连续变化

bias=6 小样本：

| 模型/条件 | 总准确率 | suc | fail | mean 生成分 |
|---|---:|---:|---:|---:|
| Qwen baseline | 37.5% | 100% | 16.7% | 2.625 |
| Qwen gaze target/evidence | 62.5% | 100% | 50.0% | 2.375 |
| Qwen random | 50.0% | 100% | 33.3% | 2.500 |
| Robo baseline | 25.0% | 100% | 0% | 3.000 |
| Robo gaze target-vs-all | 37.5% | 100% | 16.7% | 2.750 |
| Robo evidence/random | 25.0% | 100% | 0% | 2.875 / 3.250 |

8 条离散准确率每变一条就是 12.5%，不可作为主要结论；连续 `E[r]` 与同视频 margin 更适合 debug。

### 11.4 Object probe（`OBSERVED`）

| 模型/条件 | MATCH 标签正确 | 观察对象保真/行为 |
|---|---:|---|
| Qwen baseline/zero | 8/8 | 8/8 与画面一致 |
| Qwen prefill-only | 8/8 | 标签保持；个别 noun 表述轻微变化 |
| Qwen decode-only | 7/8 | 多条 noun 被改写，其中一条把实际 blue cube 改成 instructed pen 并误判 YES |
| Qwen both target/evidence | 6/8 | 人工核对 parsed fields 约 5/8 保持实际对象；3 条发生目标吸附式改写 |
| Qwen last-prompt/random | 8/8 | 未见上述系统性改写 |
| RoboReward 所有已测条件 | 8/8 | 基线已正确，干预基本不改变 |

Qwen 的典型失败是实际操作 cup/blue cube，却生成 `MANIPULATED_OBJECT: pen, MATCH: YES`；另有一条同时输出 `MANIPULATED_OBJECT: blue block` 与 `MATCH: NO`，内部不一致。说明强 gaze 并未让 Qwen 更准确地识别错误对象，反而可能把视觉描述拉向 instructed target；主要风险来自 decode 累积，而不是 prefill-only。

## 12. Qwen、RoboReward、GRM 三层诊断

GRM 使用修复后的 official incremental-after 结果：`results/mydata_bench/experiments_v2/attention_12_grm_incremental_after_official`。其 846 条均有 baseline/target/low-rank，wrong-region 有 838 条；auto-grounding 尚未人工审核，ranking 与评测有 34/36 ID 重叠。

| 模型 | Attention | Object probe | Reward | 三层判断 |
|---|---|---|---|---|
| Qwen3-VL | `A(R)` 明显上升；bias=12 近 1.0 | 8/8 → 6/8，且对象描述向指令目标改写 | bias=6 cached `C_reward=+0.2134`；random≈0；同时 mean score 下移 | hook/映射有效；有 modest gaze-specific leverage，但对象绑定变化有害，未形成稳健机制链 |
| RoboReward | `A(R)` 明显上升；bias=12 近 1.0 | 始终 8/8，几乎不变 | bias=6 cached `C_reward=+0.1527`；random=-0.4168；evidence-only 仅 +0.0711 | 视觉对象理解本来已对；attention 改变只被 reward 部分使用，leverage 较弱且依赖 both 阶段 |
| GRM | `bbox_mass_increase_rate=1.0` | **未测量**，不能填“改善” | continuous suc-fail margin：0.1629 → target-k8 0.5769；wrong-k8 0.2580（n=838），low-rank-k8 0.2049 | margin 增幅远大于两 native 模型，且 target 大于 controls；但对象层缺失，不能宣称三层都改善 |

GRM 还必须加一个限定：其 formal per-example correction estimand 报告 `target_head_specific_causal_effect_supported=false`，target shift=-0.4204、spatial specificity=-0.2348、head specificity=-0.4264（95% cluster bootstrap CI 均为负）。这与较大的 suc-fail margin 不矛盾：前者问“每条样本是否按正确方向修正且优于 controls”，后者问“成功与失败均值是否拉开”。GRM 的大部分准确率提升来自 fail accuracy 飙升，suc accuracy 仍明显较弱，不能把 margin/总准确率直接解释为已验证的 instruction-object causal binding。

## 13. 对四个核心问题的明确回答

1. **干预是否生效？— 是，`VERIFIED`。** zero-bias 精确复现；强 bias 将目标 `A(R)` 从约 0.01 推到约 0.99；prefill/decode hooks 均被调用。
2. **区域是否对齐？— 契约对齐，但语义精度不足。** 408/408 span/time checks 通过，overlay 几何正确；当前 6×8 grid 使 bbox 扩张到很粗的 token 区域，可能混入非目标证据。
3. **对象理解是否改变？— Qwen 改变且偏有害；RoboReward 基本不变。** Qwen both 条件把错误操作对象改写为 instructed target，RoboReward 基线已正确并保持。
4. **Reward 是否使用变化？— 部分使用。** 两 native 模型在 matched-random 对照下有小幅 gaze-specific margin 增益，但伴随全局分数 shift，远弱于 GRM 的 between-class margin；GRM object layer 未测且 formal specificity gate 未通过。

## 14. 效果弱的根因排序

### 高置信度

1. **`VERIFIED` 实验比较错误：backend mismatch。** 普通 baseline 与 attention baseline 不是同一数值路径，造成 3.3%–8.5% 离散不一致。
2. **`VERIFIED` 不是 hook 失效。** zero/strong/phase trace 全部证明 intervention 正常。
3. **`VERIFIED` query 因果位置不合理。** 指令词位于图像前，不可能注意后置视觉 token；last-prompt-only reward leverage 接近零。
4. **`OBSERVED` 区域过粗。** bbox→token span 没错，但 48 token/image 无法精细隔离小物体。

### 中等置信度

5. **`SUPPORTED` raw bbox-mass ranking 不等于 reward-causal ranking。** random heads 也能推高 mass；gaze 在 reward 上仅小幅优于 random。
6. **`SUPPORTED` native 模型的 reward-causal leverage 较弱/非线性。** Qwen 两阶段各有贡献；RoboReward 单阶段弱、both 才明显；强剂量反而比 moderate 剂量差。
7. **`SUPPORTED` target-vs-all 主要改变 calibration。** mean score 系统下移，fail 改善多于 suc；evidence-preserving 只能部分缓解。
8. **`OBSERVED` Qwen decode 出现 instruction-target attraction。** 这解释了 attention/object/reward 没有形成“三者都改善”。

### 已排除或未获支持

- gross token off-by-one、image/video token 混用、末帧缺失、逐帧 bbox 未更新：当前证据排除。
- Q-head/KV-head 编号混淆、跨 checkpoint 直接复用 ranking：当前实现排除。
- “只要增强 target 且不压 rival 就一定更好”：mixed result，不成立为普遍结论。
- “GRM 三层机制已完整验证”：object probe 缺失且 formal specificity gate 未通过，不能成立。

## 15. 下一轮优先方案

### P0：先修实验合同

1. 所有用于 attention 对照的 Qwen/RoboReward baseline 显式配置 `attn_implementation: eager`，保存实际 backend 和 cache mode；重新建立统一 baseline，旧 sdpa/eager 结果不得横向混合。
2. 以 cached-autoregressive 完整候选 log-prob、`E[r]`、同视频 margin 为主 readout；离散 1–5 只作辅助。
3. 预注册主 estimand：`C_reward`、target-vs-random、target-vs-wrong、target-vs-low-rank；同时报告 suc/fail 与平均 score shift。

### P1：改方法而非继续堆 top-k

4. 把 head ranking 从 raw target mass 改为**contrastive reward-margin ranking**：奖励能增大 suc-vs-counterfactual margin 的 head，惩罚 wrong-region/random 同样有效的 head；在独立 ranking split 上选 head。
5. 将 `max_pixels` 先升到 200704（12×16 merge grid）并重新 ranking；显存允许再试 401408。不要只提高分辨率却沿用旧 heads。
6. 用 bbox/mask 与 token cell 的交并面积做 area-weighted soft bias，替代“任意相交即整格增强”；保留夹爪、接触、运动、destination 与 rival-object 证据。
7. 新增显式 post-visual reward anchor，并只作用于 reward-anchor + 前几个 scoring/decode queries；不再把 instruction 早期 token 当视觉 query，也不默认全 prefill/全 decode 无差别重复。
8. bias 做 2/4/6/8 小剂量 sweep；用 attention dose 与 reward delta 曲线选非饱和区，不再默认 12。

### P2：补机制与总体证据

9. 在 GRM 上补同构 object probe、matched-random heads、prefill/decode 与 evidence-preserving，才能真正比较三层机制。
10. 扩大到跨 task、跨视频的预注册 hold-out；ranking 样本不得与评测重叠。以视频为 cluster 做 CI/bootstrap，避免把同视频 counterfactual 当独立样本。
11. 先跑 20–50 个独立视频的机制确认，再决定是否进行 846 条全量重跑；成功标准应是 margin、suc、fail 同时改善且不依赖全局降分。

## 16. Statistical validation 与 11/11 fallacy scan

- Origin Skill: experiment-agent validate
- Overall Confidence: `CAUTION`
- 原因：确定性实现检查为 solid；机制样本只有 2 个独立视频，分析为 post-hoc exploratory，不能给总体 effect size/显著性结论。
- Coverage: **11/11 checked**

| Fallacy | 严重度 | 本轮检查结果 |
|---|---|---|
| Simpson's paradox | CAUTION | 聚合准确率会掩盖 suc 降/不升而 fail 大升；已强制分层报告，未证明各 task 方向一致 |
| Ecological fallacy | NOTE | 8 条记录/2 视频的均值不外推到 individual task 或完整 benchmark |
| Berkson's paradox | CAUTION | 846 cohort 由 auto-valid grounding 选择；debug 又选 task1_1 两视频，存在选择机制 |
| Collider bias | NOTE | 未做含 collider 的回归控制；当前 paired intervention 不依赖此类调整，未发现直接证据 |
| Base-rate neglect | CAUTION | 全量 suc/fail=268/578，debug=2/6；总准确率必须与 suc/fail、margin 同报 |
| Regression to mean | NOTE | 未按极端 reward 选择 debug 视频，且有同视频 baseline/control；未见主要风险 |
| Survivorship bias | CAUTION | 只分析 grounding 有效/运行成功记录；GRM wrong control 仅 838/846，已显式披露 |
| Look-elsewhere effect | CAUTION | 既有实验搜索多输入、top-k、scope、bias；本轮结果视为探索性，不使用未校正 p-value |
| Garden of forking paths | CAUTION | 多次 post-hoc debug、无预注册，且 GRM ranking/eval 重叠 34/36；下一轮需冻结 estimand/config |
| Correlation ≠ causation | NOTE | zero/strong/random/phase 是实际干预，可支持这 2 个视频的局部因果效应；不能升级为总体机制因果结论 |
| Reverse causality | NOTE | intervention 在推理前设定，局部效应不存在反向时间因果；旧观察性指标关联仍不作方向性解释 |

没有对 n=2 视频计算伪精确的 p-value/CI。GRM 已有 cluster bootstrap 结果按其原 formal gate 解读，并明确与 between-class margin 是不同 estimand。

## 17. Reproducibility、测试与产物

### Reproducibility

- 确定性 zero-bias：`REPRODUCIBLE`，两模型生成 token 8/8 exact、候选 log-prob 最大差 0.0。
- bias=12 分成两个独立 4-sample run，合并后覆盖同一 8 条 cohort；每 run 4/4 无 invalid。
- bias=6 与阶段修复/AR scorer 均 8/8 完成、0 invalid；全部有 manifest、diagnostics JSONL、summary。
- cached 与 non-cached 是不同数值路径，绝对 logits 不要求 exact；只在各自路径内作 paired contrast。

### 测试

最终执行 `python -m pytest -q mydata_bench/tests`：**36 passed**。覆盖现有 bench/crossmodel、随机 head 层分布、query subsequence、condition selection、backend 默认兼容与显式 override。

### 主要新增文件

- `mydata_bench/gaze_debug.py`
- `mydata_bench/run_gaze_debug.py`
- `mydata_bench/tests/test_gaze_debug.py`
- `mydata_bench/tests/test_attention_backend_config.py`
- `mydata_bench/configs/v2_debug/*.yaml`
- `results/mydata_bench/experiments_v2_debug/{qwen,roboreward}_interleaved_*`

排除项：`*_phase_complete` 是发现 scope bug 前的无效阶段读出，仅为审计保留；最终阶段结论只使用 `*_phase_corrected`（attention/generation/probe）和 `*_phase_autoregressive`（continuous reward）。

## 18. 最终结论

本轮可以把“效果弱”的原因从笼统猜测收敛为：**存在一个可修复的 baseline backend 比较错误，但 steering 本身确实生效；主要瓶颈不是 hook 或 span 索引，而是低分辨率区域、错误的 causal query/ranking 目标，以及 Qwen/RoboReward 对该 attention 改变只有较弱且伴随 calibration shift 的 reward leverage。** Qwen 还出现 decode 阶段把实际错误对象改写成指令目标的反作用；RoboReward 的对象理解本来正确，attention 改变没有显著改善该层。GRM 的 reward margin leverage 明显更强，但现有证据仍不足以声称其 instruction-object 三层机制已被完整验证。
