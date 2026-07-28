# RoboRewardBench 迁移部署与因果 Attention 评测

## Material Passport

- Artifact：clean-room RoboRewardBench evaluation implementation
- Schema：ARS Material Passport / rewardbench records `1.0.0`
- 状态：raw endpoint、SAM3 grounding 双人审核、76 条 held-out steering、111 条 all-eligible follow-up，以及 17 对 exact same-video reward=1/reward=5 paired steering 已完成；当前 attention 结果均属 exploratory，尚未达到 confirmatory 样本量门槛
- 数据边界：本地 `/home/dais/workspace/data/`，禁止上传外部服务
- 更新日期：2026-07-27
- 最新 paired 配置：`rewardbench/configs/pairs_attention_sam3_exact40_official.yaml`

## 1. 实现范围

本目录不导入、不恢复、也不依赖历史 `roborewardbench` 包或旧输出。三条流水线只通过版本化 JSONL 协议共享数据：

- `raw_eval/`：原生八图 endpoint forward 评测和独立 temporal-8 消融。
- `grounding/`：Qwen instruction parser、GroundingDINO、SAM3、双人盲审和后端比较。
- `attention_eval/`：冻结 consensus、discovery-only 域内排名、causal steering、剂量曲线、63 组同视频指令实验和单 episode 视频。

公共记录由 `schemas.py` 定义，静态 JSON Schema 位于 `schemas/`。`EpisodeRecord.model_payload()` 只暴露 `example_id/video_path/task/subset/video_sha256`；`reward` 和 `gpt5_mini_check` 只能在最终 metrics 阶段连接。

代码入口：

```text
rewardbench/
├── data.py, video.py, protocol.py, metrics.py, schemas.py
├── raw_eval/{cli.py,runner.py}
├── grounding/{parser.py,base.py,dino.py,sam3.py,pipeline.py,audit.py}
├── attention_eval/{dataset.py,ranking.py,masking.py,runtime.py,experiment.py,stats.py,video.py,visualize.py}
├── configs/{full.yaml,reward1.yaml,reward1_simplified_deterministic.yaml,...}
└── schemas/*.schema.json
```

## 2. 已验证事实、尚待验证推断与建议

### Evidence

- 本地完整 test inventory 实测为 2,831 条、23 subsets、2,551 个唯一视频哈希；reward 为 `689/641/556/418/527`。
- 来源为 OXE 1,831 条、RoboArena 1,000 条。
- 反事实 reward=1 与完整集按视频 SHA-256 连接后得到 63 个唯一视频组。若同一哈希存在多条 reward=5 metadata，代码按 `example_id` 确定性选一条，不重复计组。
- 三份 carrot/cube/bottle `mean` ranking 均包含完整 36×32 heads。consensus 会先按 `skip_early_layers=2` 剔除 layer 0--1，再重新归一化 Borda rank；因此产出 1,088 个 eligible heads，首位为 L19H16。
- 当前 `robo-dopamine` 环境实测为 PyTorch 2.8.0、Transformers 4.57.0、vLLM 0.11.0；GRM attention 与 SAM3 环境保持隔离。
- 当前单元测试 28/28 通过；真实 smoke dry-run 形成 8 条 endpoint、8 条 temporal-8、8 条 parser 和 16 条 grounding endpoint 记录。
- reward=1 反事实集已完成一轮 SAM3 审核：228 条中 144 条双端点有合法框（63.2%）；其中 112 条经双人审核首末框均正确（77.8%，条件于可检测样本）。排除 1 条 robot-part 后，111 条进入 attention 候选池，并冻结为 discovery=35 / evaluation=76（15 subsets、零视频哈希泄漏）。
- 当前 SAM3 consensus steering 在 76 条 evaluation 上运行完成：所有 candidate-target bbox mass 均增加；连续 target shift 均值为 -0.0678（95% CI `[-0.0910,-0.0465]`）。目标 bbox 干预的 reward=1 标签一致率为 50/76=65.79%，baseline 为 28/76=36.84%，空间错区域对照为 19/74=25.68%，低排名 head 对照为 26/76=34.21%。完整记录见 `attention_eval/outputs/counterfactual_reward1_sam3_consensus/exp_record.md`。
- 全审核 follow-up 以完全相同的冻结 consensus ranking 在 111 条 object/object-part 样本上复现主效应：target shift 为 -0.0589（95% CI `[-0.0801,-0.0357]`），空间/head specificity 分别为 -0.0755/-0.0599（Holm 后均为 0.00020），bbox mass 111/111 增加。reward=1 标签一致率由 baseline 的 40/111=36.04% 提高至 candidate-target 的 76/111=68.47%；wrong-region、low-rank、all-heads 对照分别为 29/107=27.10%、37/111=33.33%、44/111=39.64%。完整记录见 `attention_eval/outputs/counterfactual_reward1_sam3_consensus_all_eligible/exp_record.md`。
- all-eligible run 的主五条件完整；但强 bias / top-64 sensitivity 与 duplicate 条件有 62 条 strict-score invalid records。因此剂量曲线仅作诊断，不改变主五条件结论。
- 冻结的 40 个 exact same-video pair 经 instruction-specific SAM3 grounding 和双人审核后，17 对双侧 formal-eligible。counterfactual reward=1 side 的 candidate-own target 将标签一致率从 7/17=41.18% 提升到 13/17=76.47%，target shift `-0.05894`（CI `[-0.07982,-0.03982]`）；relative to wrong-region / low-rank 的差异为 `-0.12282/-0.06053`，两项 Holm 后均小于 0.05，且 bbox mass 17/17 增加。相同 pair 的 original reward=5 side 中，candidate-own 的 reward=5 命中保持 7/17=41.18%，baseline 已为 5 的 7 条全部保留，`5→<5` 翻转为 0；`Δ5=+0.01088`，CI `[-0.04600,+0.07212]`，高于预设 non-inferiority margin `-0.05`。完整记录见 `attention_eval/outputs/pairs_attention_sam3_exact40_official/exp_record.md`。
- 同两条视频的 eager-attention hook 均验证 prefill/decode 生效，candidate-head bbox mass 分别由 `0.0281→0.8760`、`0.0137→0.8374`。这是实现诊断，不是因果研究结果。
- Qwen3-4B parser 权重前向已验证 object-part；对模型漏报的 `followed by` 顺序结构，后置结构守卫会强制 `multi_target=true`，防止其进入正式单物体样本。

### Inference

- GroundingDINO 与已有 attention 实验连续可比，适合作为 confirmatory 主后端。
- SAM3 的 mask 与视频跟踪适合作为独立 replication；两次 reward=1 结果一致地支持方向为负的 target/head-specific sensitivity，但 held-out evaluation=76 小于预设 90 条，all-eligible 又包含 discovery，均不得称为 confirmatory。
- 17 对 exact pair 初步支持两件事同时成立：固定 target steering 可降低同视频反事实 instruction 的虚假进度，而在本小样本、预设 `-0.05` margin 下未观察到 original reward=5 判断被破坏。它不是“所有成功任务无害”的证明：最终 n=17、5 subsets，original-side SAM3 dual-endpoint coverage 仅 21/40，且 reward=5 baseline 自身仅 7/17 预测为 5。
- consensus head 的迁移能力仍须在 DINO confirmatory run 和更大 reward=5 cohort 上检验；域内 head 稳定性与视频时序实验尚无正式结果。null、方向相反或非特异扰动都是有效结果。

### Recommendation

严格按第 8 节门禁推进。不要在 discovery/evaluation 冻结后依据 attention 结果改 split、grounding threshold、top-k 或 bias。下一优先级是从完整 527 条 reward=5 中预先冻结独立 cohort，使用相同 frozen intervention 做大样本 non-destructive 检验；只有两类特异性对照通过 Holm 校正、target shift 双侧显著且 hook 显示 bbox mass 上升时，才表述为 target/head-specific causal effect。

## 3. 环境部署

### 3.1 GRM、raw、DINO 和 attention

```bash
cd /home/dais/workspace/Robo-Dopamine
conda activate robo-dopamine
python - <<'PY'
import torch, transformers, vllm
print(torch.__version__, transformers.__version__, vllm.__version__)
PY
```

预期为 `2.8.x / 4.57.x / 0.11.x`。Attention runtime 强制 eager attention；vLLM 仅用于 raw eval。两者不要在同一进程加载。
raw runner 将 `VLLM_WORKER_MULTIPROC_METHOD` 固定为 `spawn`，避免 vLLM
0.11 在已初始化 CUDA 的父进程上 fork 后失败。

### 3.2 SAM3 隔离环境

```bash
conda env create -f rewardbench/environments/rewardbench-sam3.yml
conda activate rewardbench-sam3
python - <<'PY'
from transformers import Sam3Model, Sam3Processor
model_path = "/home/dais/workspace/model/sam3"
processor = Sam3Processor.from_pretrained(model_path, local_files_only=True)
print(type(processor).__name__)
PY
pip freeze > rewardbench/grounding/sam3-validated-lock.txt
```

SAM3 已在隔离环境完成本次 reward=1 endpoint grounding；其运行产物和审核结果位于 `grounding/outputs/counterfactual_reward1_sam3/sam3/`。仍须把实际验证通过的依赖写入 `sam3-validated-lock.txt`；主 GRM 环境不应安装 SAM3 依赖，也不应以 DINO 静默替代 SAM3。

## 4. Raw Evaluation

### 4.1 协议

八图顺序固定为：

1. reference start = 真实首帧；
2. reference end = `examples/blank_goal.png`；
3. before front/left/right = 首帧复制三次；
4. after front/left/right = 最后可解码终帧复制三次。

只运行 forward。`raw_eval.prompt_mode` 提供两个固定选项：`official` 是 `examples/inference.py` 的完整官方 prompt（包括原始三视角和参考锚点文字），`simplified` 是针对单视角复制和 blank goal 的简化适配版。正式 official control 使用与官方示例一致的 `temperature=0.1`、`top_p=0.9`、`top_k=50`、`max_tokens=1024`；`reward1_simplified_deterministic.yaml` 保留 temperature=0、16-token 的适配消融。所有输出必须完整匹配 `<score>±NN%</score>`。保留 `signed_score∈[-1,1]`，benchmark progress 为 `clip(s,0,1)`，阈值 `0.125/0.375/0.625/0.875` 映射 1–5。

temporal-8 从每个 subset 按哈希固定选最多 10 条，均匀取不超过 8 帧，写入独立 `temporal8_ablation.jsonl`，不与 endpoint 主结果合并。

### 4.2 命令

```bash
python rewardbench/run_raw_eval.py inventory --config rewardbench/configs/full.yaml
python rewardbench/run_raw_eval.py run --config rewardbench/configs/full.yaml
python rewardbench/run_raw_eval.py score \
  --run-dir rewardbench/raw_eval/outputs/full
```

失败不会自动重试。修复原因后显式执行：

```bash
python rewardbench/run_raw_eval.py run \
  --config rewardbench/configs/full.yaml --retry-failed
```

`score` 写 `metrics.json` 和 `invalid.json`。主要报告 23-subset 等权 Macro MAE，另报 Micro MAE、连续序数误差、exact/within-one、混淆矩阵、预测分布、signed error、reward/subset/source 分层，以及 reward=1 的预测为 1 比例和高估率。置信区间按视频哈希聚类、subset 分层 bootstrap 10,000 次。该离散指标明确标为连续 GRM 输出适配指标，不称官方原生输出。

## 5. Grounding

### 5.1 Instruction parser

正式运行使用本地 `Qwen3-4B-Instruct-2507`、greedy decoding 和受约束 JSON。启发式 parser 只用于 dry-run/故障诊断。输出区分：

- `object`、`object_part`：可进入正式范围；
- `robot_part`、`spatial_region`、`unknown`：只计覆盖率；
- `multi_target=true` 或 `ambiguous=true`：保留但不进入正式因果分析。

parser 只接收 task 和 example_id。它处理 move/slide/push/rotate/turn/touch/insert、所有格部件、顺序任务和 robot-part。

### 5.2 双后端

```bash
python rewardbench/run_grounding.py parse --config rewardbench/configs/full.yaml
python rewardbench/run_grounding.py run \
  --backend grounding_dino --config rewardbench/configs/full.yaml

conda activate rewardbench-sam3
python rewardbench/run_grounding.py run \
  --backend sam3 --config rewardbench/configs/sam3_replication.yaml
```

DINO 对目标短语、部件+父物体、中心名词构造有优先级 queries，以低阈值保留 top-10。首末端点不是各自贪心选择，而是在候选笛卡尔积上按置信度、query 身份和 crop appearance 联合选择。

SAM3 对每个 query 输出 mask、bbox、score；mask 独立保存。视频命令使用 SAM3 官方 video predictor 的 text prompt 跟踪接口。两个后端的 fingerprint 包含权重路径、threshold、query/prompt 与策略；配置改变会改变 cache identity。无合法框时写 `no_detection`，绝不使用全图、上一帧或伪框。

### 5.3 双人盲审

第一次调用会生成模板：

```bash
python rewardbench/run_grounding.py audit \
  --run-dir rewardbench/grounding/outputs/full/grounding_dino
```

将 `audit_template.jsonl` 分别复制为 `reviewer1.jsonl` 和 `reviewer2.jsonl`，两位审核者独立填写：

- `reviewer_id`
- `first_label`、`last_label`：`correct/incorrect/uncertain`
- `error_categories`
- `reason`

审核者只看 instruction、原视频、候选框/mask 和 visualization，不看 reward、GRM、ranking 或 steering。再次运行 audit；分歧项写入同 schema 的 `adjudication.jsonl` 后第三次运行。fingerprint 不匹配会拒绝陈旧审核。只有两个端点最终均为 correct 的样本才设置 `formal_eligible=true`。

后端比较：

```bash
python rewardbench/run_grounding.py compare \
  --dino-run rewardbench/grounding/outputs/full/grounding_dino \
  --sam3-run rewardbench/grounding/outputs/full/sam3
```

分别报告 dual-endpoint coverage、Wilson CI、交集和共同样本 endpoint IoU，不进行逐样本后端择优或混合。

## 6. Attention Ranking 与 Steering

### 6.1 冻结数据

```bash
python rewardbench/run_attention_eval.py prepare \
  --config rewardbench/configs/full.yaml
```

`prepare` 只接受双端点人工确认、正式 object/object-part、非多目标/歧义样本。seed `20260724` 按 `video_sha256` 分组并按 subset×target type 分层，冻结约 1/3 discovery、2/3 evaluation。`split.json` 保存完整哈希列表和 fingerprint；同一视觉内容不得跨 split。

同时生成 63 组 `paired_reward1_reward5.jsonl`。pair manifest 含标签用于描述和最终分层，但模型样本始终从无 reward 的 `eligible.jsonl` 构造。

若 evaluation 少于 90 条或少于 15 subsets，`formal_gate.status=exploratory`。

### 6.2 两条 ranking

冻结迁移主线：

```bash
python rewardbench/run_attention_eval.py rank \
  --source consensus --config rewardbench/configs/full.yaml
```

程序验证每份 ranking 都是 36×32 完整排列并保存文件 SHA-256；再先剔除 `layer < skip_early_layers`（当前为 2），并在余下 heads 内以三个 normalized rank 的均值构造 Borda consensus。禁止直接对未过滤的 36×32 完整表做 Borda，否则会错误地让 layer 0--1 进入 consensus。

域内 replication：

```bash
python rewardbench/run_attention_eval.py rank \
  --source in_domain --config rewardbench/configs/full.yaml
```

只读取 discovery。last prompt token 到 after target bbox vision keys 的 post-softmax attention 产生 mean/median raw mass、top-5% frequency 和 `bbox_mass - bbox_token_fraction × image_mass`；按 mean excess 排序、raw mean 打破平局，跳过前两层。域内实验使用 `configs/in_domain_dino.yaml`，其 `ranking_path` 已指向 `in_domain_ranking.json`，不会覆盖 confirmatory output dir。

### 6.3 Steering

```bash
python rewardbench/run_attention_eval.py steer \
  --config rewardbench/configs/full.yaml
python rewardbench/run_attention_eval.py metrics \
  --run-dir rewardbench/attention_eval/outputs/confirmatory_dino_consensus \
  --config rewardbench/configs/full.yaml
```

主条件固定：

- query = last prompt；
- top-k=8，bias=6；
- after_cam_high；
- bbox keys `+bias`，同 image span 其余视觉 keys `-bias`；
- prefill 和 decode 都生效；新增 decode text keys 由右侧 zero padding 保证 bias=0。

每条样本运行 baseline、candidate_target、candidate_wrong、low_rank_target、all_target。wrong region 必须是同一 token grid 上等形、等 token 数、不重叠矩形；找不到只缺失该对照。low-ranked heads 从 ranking 尾部选且与 candidates 不重叠。

敏感性条件为 `top_k={8,64}`、`bias={0,2,4,6}` 和 `after_all_duplicates`。bias=0 直接复用 baseline 输出，确保 bitwise identity。每层 hook diagnostics 保存 prefill/decode 调用、heads、positions 和 bbox attention mass。

主要估计量：

- target shift = candidate_target − baseline；
- spatial specificity = candidate_target − candidate_wrong；
- head specificity = candidate_target − low_rank_target。

报告 10,000 次 paired video-cluster bootstrap、双侧 sign-flip p 值；两个 specificity p 值使用 Holm 校正。方向不预设。`target_head_specific_causal_effect_supported` 只有在 target shift 双侧通过、两种 specificity 经 Holm 通过、且超过半数可诊断样本 bbox mass 上升时才为 true。

63 组次级实验对同一视觉分别使用 counterfactual/original instruction，并交叉 steering 到两个 instruction 各自审核目标框，输出 `paired_steering.jsonl`。它不参与 head 选择。

### 6.4 当前 reward=1 SAM3 consensus run

当前结果目录为 `rewardbench/attention_eval/outputs/counterfactual_reward1_sam3_consensus`，配置为 `rewardbench/configs/reward1_attention_sam3_official.yaml`。执行顺序为：

```bash
python rewardbench/run_attention_eval.py prepare \
  --config rewardbench/configs/reward1_attention_sam3_official.yaml
python rewardbench/run_attention_eval.py rank --source consensus \
  --config rewardbench/configs/reward1_attention_sam3_official.yaml
python rewardbench/run_attention_eval.py steer \
  --config rewardbench/configs/reward1_attention_sam3_official.yaml
python rewardbench/run_attention_eval.py metrics \
  --run-dir rewardbench/attention_eval/outputs/counterfactual_reward1_sam3_consensus \
  --config rewardbench/configs/reward1_attention_sam3_official.yaml
```

该 run 使用 official prompt（SHA-256 `6baabeeecf35c731aa1d058147f741f064a1391956bd8aa39448fe9bacad1b94`）、原生八图布局、eager attention hook、greedy decoding、`top_k=8`、`swap_bias=6` 和 `after_cam_high`。attention runtime 的 greedy decoding 与 raw vLLM official sampling 不同，二者的绝对分数不得混合。固定 consensus top-8 为 `(19,16),(19,23),(19,10),(20,4),(19,0),(18,30),(20,13),(22,15)`，均不在前两层。当前 CI 的 bootstrap 已按视频哈希聚类，但尚未按 subset 额外分层；正式 confirmatory 汇报前必须修正并重算。

### 6.4.1 Query-scope 消融

历史 hook 将 `[1,H,1,K]` bias 广播到全部 prefill query rows，并继续作用于
cached decode。`attention_eval.steering_query_scope` 现在显式支持：

- `all`：历史行为，全部 prefill queries + cached decode；
- `prefill`：只作用于多 query 的 prompt prefill；
- `last_prompt`：只作用于 prefill 的最后一个 query row，与
  `query_mode=last_prompt` 的 head discovery 对齐；
- `decode`：只作用于后续 `q_len=1` 的 cached decode。第一个生成 token 的 logits
  来自 prefill 最后一行，因此不包含在该条件中。

默认仍为 `all`，所以既有配置和结果不变。独立消融配置
`configs/reward1_attention_sam3_query_scope_ablation.yaml` 复用已经人工审核的
SAM3 grounding、冻结 consensus top-8 和原 evaluation split，并写入新的 output
directory。它为每个 scope 运行 candidate-target、candidate-wrong 和
low-rank-target，不混入 top-k/bias sweep：

```bash
python rewardbench/run_attention_eval.py prepare \
  --config rewardbench/configs/reward1_attention_sam3_query_scope_ablation.yaml
python rewardbench/run_attention_eval.py steer \
  --config rewardbench/configs/reward1_attention_sam3_query_scope_ablation.yaml
python rewardbench/run_attention_eval.py metrics \
  --run-dir rewardbench/attention_eval/outputs/counterfactual_reward1_sam3_query_scope_ablation \
  --config rewardbench/configs/reward1_attention_sam3_query_scope_ablation.yaml
```

`steering.jsonl` 的每条记录及 hook diagnostics 都保存 `query_scope`；
`attention_metrics.json.query_scope_ablation` 分 scope 报告 target shift、
spatial/head specificity、bbox-mass increase rate，以及相对 legacy `all` 的
candidate-score 差。不得把 `decode` 解释为“第一个 score token only”。

### 6.4.2 All-eligible exploratory follow-up

如需在全部已审核的 SAM3 reward=1 物体/物体部件样本上验证同一套**冻结** consensus heads，使用独立配置 `rewardbench/configs/reward1_attention_sam3_official_all_eligible.yaml`：

```bash
python rewardbench/run_attention_eval.py prepare \
  --config rewardbench/configs/reward1_attention_sam3_official_all_eligible.yaml
python rewardbench/run_attention_eval.py steer \
  --config rewardbench/configs/reward1_attention_sam3_official_all_eligible.yaml
python rewardbench/run_attention_eval.py metrics \
  --run-dir rewardbench/attention_eval/outputs/counterfactual_reward1_sam3_consensus_all_eligible \
  --config rewardbench/configs/reward1_attention_sam3_official_all_eligible.yaml
```

该配置固定读取既有 76 条 evaluation run 的 `consensus_ranking.json`，因而不要执行 `rank`。`steering_partition: all_eligible` 显式读取新 run 的 `eligible.jsonl` 全部 111 条（112 条双人审核正确样本中的 1 条 robot-part 仍按预设范围排除）。它混入原来的 discovery 样本，metrics 会写 `exploratory_all_eligible_followup`，不得作为 held-out 或 confirmatory 结果替代 76 条 evaluation run。

实际结果为：target shift `-0.0589`（95% CI `[-0.0801,-0.0357]`），空间/head specificity `-0.0755/-0.0599`，两项 Holm 校正后 p 均为 `0.00020`，111/111 selected-head bbox mass 增加。离散 reward=1 标签一致率从 36.04%（40/111）变为 68.47%（76/111）；这与 76 条 held-out run 的 36.84%→65.79% 方向和量级一致。强 bias / top-64 sensitivity 的 strict-score invalid 使其仅能作为不完整诊断；完整结果、五档标签分布和限制见该输出目录的 `exp_record.md`。

### 6.4.3 已完成：同视频 reward=5 non-destructive exact-paired test

reward=1 结果本身不能证明同一干预对原始 reward=5 成功任务无害；因此该问题被预先定义为 non-destructive paired test，而非把“预期方向”当作验收条件。

完整反事实集与原 benchmark 共有 63 个“同一视频、两条 instruction” pair。paired raw baseline 已完成 63/63：原始 reward=5 instruction 的 signed score 平均比反事实 reward=1 instruction 高 0.3497（95% CI `[0.2924,0.4068]`，双侧 p=0.00010）。111 条 SAM3 审核合格 reward=1 object/object-part 样本中，只有 **40 条**能按视频 SHA-256 匹配到同视频 reward=5 原始 instruction；71 条没有对应 reward=5 metadata，而 raw pair 中另有 23 条未进入该 SAM3 formal 集。

2026-07-27 已将这 40 个候选固定为 `attention_eval/outputs/pairs_sam3_exact40/paired_reward1_reward5_exact40.jsonl`（fingerprint `adc703bfd0e5a034733ce0a621ebc321df933b2d8621d2f79cd7900860bb8b9c`）。构造使用 counterfactual 的**源 `example_id` 与视频 SHA-256 双重匹配**已有 SAM3 formal-eligible manifest；23 个排除 pair 与 71 个无原始配对的审核样本均已保存，避免事后按 GRM、steering 或 reward=5 grounding 结果改变候选集合。

实际执行严格使用 official prompt、冻结 consensus top-8、`after_cam_high`、`top_k=8`、`bias=6`，不重排 heads 或调参。每条 instruction 独立 parsing、SAM3 bbox 和双人端点审核；未共享另一条 instruction 的框。40 对中，counterfactual side 40/40 有 dual-endpoint detection、38/40 formal eligible；original side 21/40 有 dual-endpoint detection、17/40 formal eligible；最终 **17/40** 为双侧完整 pair（5 subsets）。

主结果如下：

1. counterfactual reward=1：candidate-own 使标签一致率 `7/17→13/17`，target shift `-0.05894`（CI `[-0.07982,-0.03982]`）；相对 wrong-region / low-rank 的差异为 `-0.12282/-0.06053`，两种 specificity 对照经 Holm 后均 <0.05，bbox mass 17/17 增加。
2. original reward=5：`Δ5=+0.01088`（CI `[-0.04600,+0.07212]`，双侧 p=0.71683），其 CI 下界高于预设 non-inferiority margin `-0.05`；baseline 与 candidate-own 都有 `7/17` 预测为 reward=5，baseline 为 5 的 7 条均保留，`5→<5` 为 0。
3. 2×2 cross steering 也显示 instruction-conditioned bbox 的影响：counterfactual instruction 下 own-counterfactual bbox 比 original bbox 低 `-0.24418`；original instruction 下 counterfactual bbox 比 own-original bbox 低 `-0.07688`。同一视觉的 baseline `original−counterfactual` 为 `+0.33735`（CI `[+0.18765,+0.49706]`）。

因此，该实验可表述为：**在小型、人工审核的 exact-paired cohort 中，结果初步支持固定 target steering 抑制反事实 instruction 的虚假进度，同时未观察到超过预设阈值的 original-success 判断破坏。** 不应表述为“已证明对所有成功任务无害”：n=17、5 subsets，且 original-side SAM3 覆盖有限。详细五档分布、对照、敏感性和限制见 `attention_eval/outputs/pairs_attention_sam3_exact40_official/exp_record.md`。

### 6.4.3 下一步：独立大样本 original reward=5 non-destructive cohort

优先级高于继续调 top-k/bias、从 paired 结果选样本或制作更多视频。由于 exact pair 的 original-side 最终 formal 覆盖仅 17/40=42.5%，为避免事后补样，已在不看 GRM/steering 结果的条件下冻结完整 benchmark 的 **527 条 reward=5**：`cohorts/outputs/reward5_full_527/episodes.jsonl`，fingerprint `651478ab5ff94b2c39c5fd92cf78deed409734e2ec0b418b114b5ed8f98cdb83`。该输入 manifest 与 ID allow-list 只含 `example_id/video_path/task/subset/video_sha256`，不含 reward 或 `gpt5_mini_check`；reward=5 仅用于这次离线 cohort 构造与最终 metrics。

流程固定为：独立 parser → SAM3 grounding → 双人盲审 → 冻结 eligible manifest → 使用**完全相同**的 official prompt、consensus top-8、`top_k=8`、`bias=6`、`after_cam_high` 跑 baseline/candidate-own/wrong/low-rank/all-head。primary 仍是 `Δ5`（margin `-0.05`）、reward=5 prediction rate、baseline-5 retention 与 `5→<5` 翻转；以视频哈希 cluster bootstrap 报告。该 cohort 不需要与 reward=1 同视频，目的是提供对“成功任务一般不被破坏”的足够分母。完成后再用 GroundingDINO 在冻结样本上独立复现，或将 DINO 作为 confirmatory 主后端。

配置已固定为 `configs/reward5_full_attention_sam3_official.yaml`。先在隔离的 SAM3 环境运行 parser、grounding 和人工审核；只有 `audit_summary.json` 为 `complete` 后，才在 `robo-dopamine` 环境运行 attention：

```bash
conda activate rewardbench-sam3
python rewardbench/run_grounding.py parse \
  --config rewardbench/configs/reward5_full_attention_sam3_official.yaml
python rewardbench/run_grounding.py run --backend sam3 \
  --config rewardbench/configs/reward5_full_attention_sam3_official.yaml
python rewardbench/run_grounding.py audit \
  --run-dir rewardbench/grounding/outputs/reward5_full_sam3/sam3

# 完成 reviewer1/reviewer2（必要时 adjudication）后：
conda activate robo-dopamine
python rewardbench/run_attention_eval.py prepare \
  --config rewardbench/configs/reward5_full_attention_sam3_official.yaml
python rewardbench/run_attention_eval.py steer \
  --config rewardbench/configs/reward5_full_attention_sam3_official.yaml
python rewardbench/run_attention_eval.py metrics \
  --run-dir rewardbench/attention_eval/outputs/reward5_full_sam3_consensus \
  --config rewardbench/configs/reward5_full_attention_sam3_official.yaml
```

不得运行 `rank`：该 cohort 读取既有冻结 consensus ranking。`metrics` 会通过
`expected_reward_for_metrics_only: 5` 在推理结束后输出五档分布、`Δ5`、5 档 retention 和
`5→<5` 翻转；该标签字段不进入模型路径。

### 6.5 Endpoint heatmap 可视化（当前推荐）

当前审阅 attention 的首选不是逐帧视频，而是从该 run 的冻结 evaluation split 中抽取 `N` 条 endpoint，直接显示 GRM 实际使用的单视角代表帧和三组结果：baseline、candidate heads + 等 token 数 wrong-region 空间对照、candidate heads + target bbox 实验组。

```bash
python rewardbench/run_attention_eval.py visualize \
  --run-dir rewardbench/attention_eval/outputs/counterfactual_reward1_sam3_consensus \
  --count N \
  --seed 20260724
```

抽样母集是 `run-dir/eligible.jsonl` 与 `split.json` 定义的 `evaluation` 分区；本 run 为已审核的 76 条，不包含 discovery=35 或其余未审核 reward=1 数据。选择键为 `seed:video_sha256:example_id` 的 SHA-256 排序，取前 `N` 条，因此是与结果无关、可复现的伪随机抽样。改变 `run-dir` 会严格改用该 run 自己冻结的 evaluation split。

每条会重新运行三次 GRM forward 以取得生成 attention（baseline、spatial control、target experiment），但**不会加载 SAM3/DINO、不会读取中间视频帧、不会重新 grounding**：bbox 直接复用 `eligible.jsonl` 中已经人工确认的首/末端点框。图中不再显示 before/after 原图；只渲染三个 `after_cam_high` 的 `JET`、alpha=0.45、BICUBIC 插值 heatmap overlay 和 lime bbox。模型 forward 仍使用原生八图输入。标题写入最终 `progress=clip(signed_score,0,1)` 与 selected-attention mass。空间对照无法构造时明确显示 unavailable，不伪造框或缩小 bbox。

输出互相隔离，避免覆盖正式 steering 记录：

```text
attention_eval/outputs/<run>/
  endpoint_visualizations/n<N>_seed<seed>/<video_sha256>/endpoint_attention.png
  endpoint_visualizations_n<N>_seed<seed>.jsonl
  endpoint_visualizations_n<N>_seed<seed>.json
```

目前 heatmap 为实际干预的 top-8 heads 平均 attention；它服务于解释整体 intervention。若研究问题需要与旧 `visualize_stage3_head_attention.py` 完全一致的“单一 head”图，应另行固定并记录 visualized head，不能在看过结果后择优选择。

### 6.6 时序视频（可选，不是当前审阅流程）

```bash
python rewardbench/run_attention_eval.py video \
  --run-dir rewardbench/attention_eval/outputs/confirmatory_dino_consensus \
  --count 12 --seed 20260724
```

该命令用于真正的 1 FPS 时间过程：每秒构造 start→current forward，并为每帧重新定位目标。DINO 逐帧检测，SAM3 配置则需加载 SAM3 video tracking，因此它比 endpoint visualizer 显著更重。它只适合研究物体移动过程，不是查看当前 GRM endpoint 输入与 heatmap 的必需步骤。

## 7. 四 GPU、resume 与产物

在四份配置副本中分别设置：

```yaml
raw_eval:       {shard_id: 0, num_shards: 4}
grounding:      {shard_id: 0, num_shards: 4}
attention_eval: {shard_id: 0, num_shards: 4}
```

把 shard_id 改为 0/1/2/3，并在四张 GPU 上分别设置 `CUDA_VISIBLE_DEVICES=0..3`。分片函数固定为 `int(video_sha256[:16],16) % num_shards`。每张 GPU 写独立 JSONL；所有 shard 出现后确定性合并。不要让四个进程写同一个非 shard 文件。

所有逐样本记录 append-only。默认 resume 跳过已有成功或失败记录；失败只能用 `--retry-failed` 追加新 attempt。每个 run 保存配置、命令、Git revision、Python/platform/GPU、config/model/ranking/grounding/split fingerprints。大输出、frames、runs 和 caches 已加入 `.gitignore`。

关键产物：

```text
raw_eval/outputs/<run>/
  manifest.json, records.shard-*.jsonl, temporal8_ablation.jsonl,
  inventory.json, metrics.json, invalid.json
grounding/outputs/<run>/<backend>/
  manifest.shard-*.json, grounding.jsonl, grounding_summary.json,
  masks/, visualizations/, audit_final.jsonl, audit_summary.json
attention_eval/outputs/<run>/
  prepare_manifest.json, split.json, eligible.jsonl,
  consensus_ranking.json | in_domain_ranking.json,
  steering.jsonl, paired_steering.jsonl, attention_metrics.json,
  endpoint_visualizations_n<N>_seed<seed>.json,
  endpoint_visualizations_n<N>_seed<seed>.jsonl,
  endpoint_visualizations/, attention_video_manifest*.json, video*/
```

## 8. 门禁与验收顺序

1. 单元测试和 schema：必须全通过。
2. `smoke8.yaml`：检查不同 subset、分辨率、时长和目标类型。
3. discovery grounding：两后端完成检测和双人审核；只在 discovery 做 attention pilot。
4. 冻结 `split.json`、DINO config、consensus fingerprint 和统计协议。
5. full raw eval：必须 2,831/2,831 有最终记录，正式汇报前 invalid=0。
6. evaluation steering：至少 90 条、15 subsets，否则只称 exploratory。
7. SAM3 replication、63 组 paired reward=1/reward=5 的双 instruction grounding/audit 与 non-destructive test，以及按需要生成 endpoint heatmap（当前优先）或固定 episode 时序视频。
8. 汇总 evidence/inference/recommendation；不以得到预期方向作为验收条件。

测试和 dry-run：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python rewardbench/run_raw_eval.py run \
  --config rewardbench/configs/smoke8.yaml --dry-run
python rewardbench/run_grounding.py parse \
  --config rewardbench/configs/smoke8.yaml --dry-run
python rewardbench/run_grounding.py run --backend grounding_dino \
  --config rewardbench/configs/smoke8.yaml --dry-run
python rewardbench/run_attention_eval.py rank --source consensus \
  --config rewardbench/configs/full.yaml
```

GPU integration gate应另选两条真实视频，分别执行 DINO + GRM baseline/steer，并检查：

- image span 长度等于 `t×(h/merge)×(w/merge)`；
- hook 只修改指定 layer/head；
- prefill/decode 都有调用；
- decode 新 text keys bias=0；
- bias=0 与 baseline 完全相同；
- 正 bias 后 selected-head bbox mass 实际上升。

## 9. 解释边界与故障处理

- `invalid` 不能按 0 分计入，也不能静默丢弃；先修复，再显式 retry。
- 某端点无合法框则样本不进入正式因果结论。
- 未经双人审核的框不能进入 evaluation steering。
- label、`gpt5_mini_check`、paired reward 不得进入 parser、grounder、ranking 或 model prompt。
- evaluation attention 可以用于预声明的估计和描述，不得反过来选 heads、threshold、top-k、bias 或样本。
- 若 target、wrong、low-rank 同方向同量变化，应报告非特异扰动。
- 若 target shift 为 null，应报告置信区间和可检测效应范围，不改协议追求显著。
- 当前 reward=1 SAM3 consensus 有 `n=76` held-out 与 `n=111` all-eligible 数值结果；前者样本量不足、后者包含 discovery，均不得升级为完整 benchmark、DINO confirmatory 或跨 reward 等级的结论。
