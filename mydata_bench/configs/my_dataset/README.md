# LJX/LFZ counterfactual baseline 配置与运行说明

这些配置与 `rewardbench/configs/` 完全隔离。新数据集的所有命令都必须通过：

```bash
python mydata_bench/run_my_dataset.py
```

启动。复制过来的旧入口已经改为仅导入 `mydata_bench.*`，不会再跳回
`rewardbench/`；但它们仍使用旧 RoboRewardBench schema，因此新数据集继续只使用
`run_my_dataset.py`。

## 输入协议

- RoboReward/Qwen：使用 checkpoint-native `faceImg.mp4`，输出离散的 `ANSWER: <1-5>`。
- RoboReward 提供两种显式媒体顺序：已有 `roboreward_8b_native_front.yaml`
  复现 HELM 的 `text → video`；`roboreward_8b_model_card_native_front.yaml`
  遵循模型卡引用的 Qwen3-VL 示例，使用 `video → text`。两者写入不同目录。
- Qwen 同样提供两种顺序。`qwen3vl_8b_native_front.yaml` 是已完成的
  `text → video`；`qwen3vl_8b_model_card_native_front.yaml` 只改变媒体顺序，使用
  processor 默认帧数；`qwen3vl_8b_model_card_native_front_attention8.yaml` 固定最多8帧，
  专门作为显存可行的白盒 attention runtime 对照。默认帧数和8帧结果不能混称为同一个
  baseline。
- GRM：在 official 八图 prompt 中使用同步的 front/left/right 三视角首末帧，不再把 front 帧重复填入腕部相机位置。
- 模型输入 manifest 不包含 reward、matched/fail 标签；只有 `score` 命令可以读取标签。

RoboReward/Qwen 的 native-front 与 GRM 的 multiview 属于不同的 model-native 输入协议，因此它们的绝对分数不能被解释为严格的同输入跨模型比较。

## 数据准备与审计

在 `Robo-Dopamine` 仓库根目录运行：

```bash
conda activate robo-dopamine
python mydata_bench/run_my_dataset.py prepare \
  --config mydata_bench/configs/my_dataset/prepare_ljx_lfz.yaml
python mydata_bench/run_my_dataset.py audit \
  --prepared-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1
python mydata_bench/run_my_dataset.py inventory \
  --inputs mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/model_inputs/inputs.jsonl
```

冻结后的 inventory 必须包含 755 个样本、169 个 source groups，且模型输入中不能出现标签字段或 `suc/fail` 路径。`prepare` 会对 canonical 视频与所有副本进行哈希核验；重复执行只会覆盖 `mydata_bench/artifacts/` 下的 prepared manifests。

## 小规模 dry-run

Dry-run 不加载模型，并应写入独立的 sanity 目录，不能与正式输出混合：

```bash
python mydata_bench/run_my_dataset.py run \
  --config mydata_bench/configs/my_dataset/roboreward_8b_native_front.yaml \
  --dry-run --limit-groups 2 \
  --output-dir mydata_bench/outputs/my_dataset/sanity/roboreward
python mydata_bench/run_my_dataset.py run \
  --config mydata_bench/configs/my_dataset/qwen3vl_8b_native_front.yaml \
  --dry-run --limit-groups 2 \
  --output-dir mydata_bench/outputs/my_dataset/sanity/qwen
python mydata_bench/run_my_dataset.py run \
  --config mydata_bench/configs/my_dataset/grm_8b_multiview_endpoints.yaml \
  --dry-run --limit-groups 2 \
  --output-dir mydata_bench/outputs/my_dataset/sanity/grm
```

## 首轮完整 baseline

有三张空闲 GPU 时，可让每个模型占用一张 GPU 并行运行：

```bash
mkdir -p mydata_bench/logs/my_dataset/ljx_lfz_cf_v1
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py run \
  --config mydata_bench/configs/my_dataset/roboreward_8b_native_front.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/roboreward.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python mydata_bench/run_my_dataset.py run \
  --config mydata_bench/configs/my_dataset/qwen3vl_8b_native_front.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/qwen.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python mydata_bench/run_my_dataset.py run \
  --config mydata_bench/configs/my_dataset/grm_8b_multiview_endpoints.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/grm.log 2>&1 &
wait
```

监控命令：

```bash
tail -f mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/roboreward.log
nvidia-smi
wc -l mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/*/records.shard-00.jsonl
```

一个全新的完整 run 应有 755 条最新有效预测。若存在 invalid，应先检查失败原因，再决定是否显式使用 `--retry-failed`；程序不会自动重试。

## RoboReward 模型卡顺序对照实验

该实验保留原始 MP4、checkpoint processor、greedy decoding 等设置，只把媒体顺序改为
`video → text`，不会覆盖已经完成的 HELM-aligned RoboReward baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py run \
  --config mydata_bench/configs/my_dataset/roboreward_8b_model_card_native_front.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/roboreward_model_card.log 2>&1 &
```

推理完成后评分：

```bash
python mydata_bench/run_my_dataset.py score \
  --run-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/roboreward_8b_model_card_native_front \
  --inputs mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/model_inputs/inputs.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl
```

## 推理完成后的评分

标签只能传给以下后处理命令：

```bash
python mydata_bench/run_my_dataset.py score \
  --run-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/roboreward_8b_native_front \
  --inputs mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/model_inputs/inputs.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl
python mydata_bench/run_my_dataset.py score \
  --run-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/qwen3vl_8b_native_front \
  --inputs mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/model_inputs/inputs.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl
python mydata_bench/run_my_dataset.py score \
  --run-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/grm_8b_multiview_endpoints \
  --inputs mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/model_inputs/inputs.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl
```

完整评分应输出：

```text
ready=True examples=755 groups=169 pairs=586
```

主要人类可读结果为 `scoring/metrics.md`；机器可读的逐样本、逐 pair、逐 group 结果保存在同一 `scoring/` 目录下。

## Qwen video → text 补充实验

先运行只改变媒体顺序、保留 processor 默认帧数的主对照：

```bash
mkdir -p mydata_bench/logs/my_dataset/ljx_lfz_cf_v1
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py run \
  --config mydata_bench/configs/my_dataset/qwen3vl_8b_model_card_native_front.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/qwen_model_card.log 2>&1 &
```

完成后评分：

```bash
python mydata_bench/run_my_dataset.py score \
  --run-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/qwen3vl_8b_model_card_native_front \
  --inputs mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/model_inputs/inputs.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl
```

之后若进入白盒实验，还需单独运行8帧兼容 baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py run \
  --config mydata_bench/configs/my_dataset/qwen3vl_8b_model_card_native_front_attention8.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/qwen_model_card_attention8.log 2>&1 &
```

## 白盒实验准备流程

### 1. 语义角色与三路切分

这两步不打开标签，也不加载模型：

```bash
python mydata_bench/run_my_dataset.py roles \
  --config mydata_bench/configs/my_dataset/roles_ljx_lfz_cf_v1.yaml
python mydata_bench/run_my_dataset.py split \
  --config mydata_bench/configs/my_dataset/split_ljx_lfz_cf_v1.yaml
```

当前冻结结果为33个 discovery groups、33个 validation groups、103个 test groups；
对应145、145、465条 instruction variants。所有755条 instruction 均成功解析，且 split
审计确认不存在 group/media hash 泄漏。

### 2. 协议冻结

```bash
python mydata_bench/run_my_dataset.py freeze \
  --config mydata_bench/configs/my_dataset/protocol_ljx_lfz_cf_v1.yaml
```

该命令会冻结输入、split、roles、模型、prompt/decoding 和 primary intervention 参数的
fingerprint。当前只有14个 task subsets，所以生成的 test 即使样本数足够，也仍属于
exploratory white-box test。

### 3. processor 对齐的 grounding requests

必须等 RoboReward model-card、Qwen attention8 和 GRM 三个 baseline 都完成后执行：

```bash
python mydata_bench/run_my_dataset.py ground-prepare \
  --config mydata_bench/configs/my_dataset/grounding_ljx_lfz_cf_v1.yaml
```

输出 `grounding/requests.jsonl`。RoboReward/Qwen 使用各自 processor 记录的真实
`frames_indices`；GRM 使用真实 front/left/right 首末帧。ordinal、左右、最近/最远任务必须
人工指定实例，不能直接接受 SAM3 文本检索的第一候选。

生成 SAM3 proposals：

```bash
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py ground-propose \
  --config mydata_bench/configs/my_dataset/grounding_ljx_lfz_cf_v1.yaml
```

该命令只生成 candidates、masks 和 `review_template.jsonl`，所有记录均为
`auto_accepted=false`；必须人工选择 target instance 和 wrong/background control。

审核结果写到 `grounding/reviews.jsonl`。每条 eligible 记录至少包含：

```json
{
  "example_id": "...",
  "status": "eligible",
  "review_id": "reviewer/version",
  "models": {
    "roboreward": {
      "target": {"image_path": "...", "bbox": [x1, y1, x2, y2]},
      "wrong_region": {"image_path": "同一张图", "bbox": [x1, y1, x2, y2]}
    },
    "qwen": {
      "target": {"image_path": "...", "bbox": [x1, y1, x2, y2]},
      "wrong_region": {"image_path": "同一张图", "bbox": [x1, y1, x2, y2]}
    },
    "grm": {
      "target": {"image_path": "...after_cam_high...", "bbox": [x1, y1, x2, y2]},
      "wrong_region": {"image_path": "同一张after_cam_high", "bbox": [x1, y1, x2, y2]}
    }
  }
}
```

审核后运行：

```bash
python mydata_bench/run_my_dataset.py ground-audit \
  --requests mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding/requests.jsonl \
  --proposals mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding/proposals.jsonl \
  --reviews mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding/reviews.jsonl \
  --output-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding
python mydata_bench/run_my_dataset.py attention-prepare \
  --config mydata_bench/configs/my_dataset/attention_ljx_lfz_cf_v1.yaml
```

### 4. bias=0 gate、ranking、validation

先用三个 `*_steer_validation.yaml` 做20条 equivalence gate：

```bash
python mydata_bench/run_my_dataset.py equivalence --config <steer_validation.yaml>
```

RoboReward/Qwen 要求 attention runtime 与各自帧数一致的 canonical baseline 逐样本相同。
GRM 的 vLLM baseline 与 Transformers eager runtime 只做诊断比较；正式 causal baseline 使用
同一 eager runtime 的 no-hook 输出。

通过 gate 后，每个模型分别执行自己的 discovery ranking：

```bash
python mydata_bench/run_my_dataset.py rank --config <model_rank_terminal_skip8.yaml>
```

再运行 validation controls：

```bash
python mydata_bench/run_my_dataset.py steer --config <model_steer_validation.yaml>
```

三个模型不能共用 primary heads。默认方法为 terminal/after_cam_high、last-prompt ranking、
skip前8层、top-8、all-query steering；candidate-target 必须同时对照 candidate-wrong、
low-rank-target 和 layer-matched-random-target。

冻结 heads、k、bias 后才能运行 `*_steer_test.yaml`。在此之前不要打开 test attention 或
steering 输出。

### 5. steering 评分

标签只允许在所有 intervention condition 完成后加入：

```bash
python mydata_bench/run_my_dataset.py score-steering \
  --records <steering_validation_or_test.jsonl> \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl \
  --output-dir <scoring_output_dir>
```

主要输出包括 fail correction、suc harm、balanced net correction、exact delta、spatial
specificity、low-rank specificity、layer-matched random specificity，以及按 group 聚类、
task_id 分层的 bootstrap 区间。

<details>
<summary>已弃用：旧 strict-only v2 草案（仅保留历史记录，不要执行）</summary>

## 自动假设 grounding 的 v2 探索性矩阵

这一流程与上面的人工审核 white-box 流程严格分目录。它只用于快速探索
N={5,10,20} 个外部 ranking source 与 K={8,32,64} 个 steering heads 的九宫格，
不会覆盖 `grounding/`、`attention_inputs/` 或已有 baseline。

### 1. 每个 query 独立保留 SAM3 候选

v2 使用与 v1 完全相同的三个 baseline run，但写入新的
`grounding_per_query_v2/`。SAM3 threshold 为0.10、mask threshold 为0.50，
并为每个 query 独立保留最多20个候选，避免 reference/destination query 被高分
target query 挤掉：

```bash
python mydata_bench/run_my_dataset.py ground-prepare \
  --config mydata_bench/configs/my_dataset/grounding_ljx_lfz_cf_v2_per_query.yaml
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py ground-propose \
  --config mydata_bench/configs/my_dataset/grounding_ljx_lfz_cf_v2_per_query.yaml
```

### 2. 生成自动假设 review 与全量 attention inputs

```bash
python mydata_bench/run_my_dataset.py assume-grounding \
  --config mydata_bench/configs/my_dataset/assumed_grounding_ljx_lfz_cf_v2.yaml
python mydata_bench/run_my_dataset.py attention-prepare \
  --config mydata_bench/configs/my_dataset/attention_assumed_ljx_lfz_cf_v2.yaml
```

`require_all_valid=true` 会在任一自动选择无效时停止；attention 阶段只接受
`assumed_valid`，要求755条输入对三个模型全部覆盖，并额外写出每个模型的
`all.jsonl`。这些 review 没有人工确认：机器字段固定为
`grounding_status=auto_assumed_unreviewed`、`human_reviewed=false`、
`claim_status=exploratory`。正式 matrix 还会再次强制
`grounding_resolution=strict`；任何 `assumed_proxy` 或
`auto_proxy_unreviewed` 输入都会在加载模型前被拒绝，不能混入结果。

### 3. 冻结共享 S5/S10/S20 ranking cohort

```bash
python mydata_bench/run_my_dataset.py ranking-cohort \
  --config mydata_bench/configs/my_dataset/ranking_cohort_ljx_lfz_cf_v1.yaml
```

输出写入 `ranking_cohort_v1/`。三个模型必须使用完全相同且严格嵌套的20条排序：
S5 是 S10 前5条，S10 是 S20 前10条。选择过程不读评分标签。

### 4. 运行四个探索性矩阵

每个矩阵都固定为 excess-mass ranking、skip前8层、N=5/10/20、K=8/32/64、
bias=6、all-query scope、20条 ranking inputs 和755条 evaluation inputs。
RoboReward/Qwen 使用 `roborewardbench_native` 且 attention frame cap 为8；GRM
使用 official 八图输入、`after_cam_high` target，且不保存 generation attentions。

```bash
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/roboreward_matrix_text_then_video.yaml
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/roboreward_matrix_video_then_text.yaml
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/qwen_matrix_video_then_text.yaml
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/grm_matrix.yaml
```

若最新 attempt 为 invalid，先审查错误，再对同一 config 显式增加
`--retry-failed`；不要换 config 继续写同一个 output dir，因为 run fingerprint gate
会拒绝混合运行。

Qwen 的这个矩阵明确对应8帧 attention-compatible runtime。processor 默认帧数的
`qwen3vl_8b_model_card_native_front` 仍是另一个 baseline，不能与 attention8 结果
混称或覆盖。

### 5. 标签后加入与四 scope 评分

只有矩阵全部完成后才能运行 `score-matrix`。四个 variant 分别执行以下命令；
其余三个命令只需替换 variant 为 `roboreward_video_then_text`、
`qwen_video_then_text_attention8`、`grm_official_eight_image`：

```bash
python mydata_bench/run_my_dataset.py score-matrix \
  --records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/roboreward_text_then_video/steering/records.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl \
  --selection-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/ranking_cohort_v1/selection_manifest.json \
  --output-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/roboreward_text_then_video/scoring \
  --expected-count 755
```

评分必须同时报告以下口径：

- `all_including_rank_sources` 为755条，但包含 ranking/eval overlap，是明确的
  in-sample contaminated 探索性结果；
- `common_unseen_s20` 排除全部 S20 source groups，固定为682条，是跨 N 的
  primary comparison；
- `n_specific_unseen` 对 N=5/10/20 分别为738/721/682条；
- `ranking_source_groups_only` 对 N=5/10/20 分别为17/34/73条，只作 overlap 诊断。

该矩阵没有 wrong-target、low-rank、layer-matched-random controls，因此不能据此声称
target-specific 或 head-specific causal specificity。所有结果只能标为 exploratory；
自动假设 grounding 不能替代人工审核。

</details>

## 全量 exploratory ranking-steering 矩阵（当前正式入口）

当前正式流程允许全部755条进入探索性实验，但严格区分两类 grounding：

- 453条 `strict / auto_assumed_unreviewed`；
- 302条 `proxy / auto_proxy_unreviewed`。

两类均为 `human_reviewed=false`、`claim_status=exploratory`。proxy 使用可追溯
fallback，不是 target 真值；最终评分必须分层报告，不能把755条混称为真实
target-grounded 结果。

### 1. 构建并审计 label-free 输入

现有冻结来源为755条 requests、2265条主 proposals（全部 `ok`）以及864条
supplement proposals（全部 `no_candidate`、不贡献候选）。主 proposals 的可复现
SAM3 配置为 threshold=0.15、mask threshold=0.50、全 query 合计 top-40，见
`grounding_exploratory_ljx_lfz_cf_v1.yaml`。已有 artifacts 不需要重跑 SAM3。

```bash
python mydata_bench/run_my_dataset.py assume-grounding \
  --config mydata_bench/configs/my_dataset/assumed_grounding_ljx_lfz_cf_v1.yaml
python mydata_bench/run_my_dataset.py attention-prepare \
  --config mydata_bench/configs/my_dataset/attention_assumed_ljx_lfz_cf_v1.yaml
python mydata_bench/run_my_dataset.py ranking-cohort \
  --config mydata_bench/configs/my_dataset/ranking_cohort_ljx_lfz_cf_v1.yaml
```

硬门应得到：0 invalid；RoboReward、Qwen、GRM 的 `all.jsonl` 各755条；三模型
ranking manifest 各20条且顺序相同；S5⊂S10⊂S20；S20 common-unseen 为682条。

### 2. 运行四个矩阵

共同设置为 excess-mass、skip前8层、N=5/10/20、K=8/32/64、bias=6、
all-query、8帧 attention runtime。第一批使用 GPU 0、1、3：

```bash
mkdir -p mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/exploratory_matrix_roboreward_video_then_text.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/roboreward_video_text.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/exploratory_matrix_qwen_video_then_text_attention8.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/qwen_video_text_attention8.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/exploratory_matrix_grm.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/grm.log 2>&1 &
```

任一 GPU 空闲后，再运行 RoboReward text→video：

```bash
CUDA_VISIBLE_DEVICES=<空闲的0、1或3> python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/exploratory_matrix_roboreward_text_then_video.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/roboreward_text_video.log 2>&1 &
```

每个 variant 只加载模型一次。20条 ranking mass 仅收集一次，再派生三个 N；每条
evaluation 运行1个同-runtime baseline和9个 steering 条件。完整最新逻辑状态为
20条 mass、3个 ranking JSON、7550条有效 evaluation records。输出为 append-only；
失败后审查原因，再对同一 config 使用 `--retry-failed`。run fingerprint 会拒绝
配置、模型、实现代码、帧数、顺序、prompt 或推理合同不一致的混合续跑。

Qwen 矩阵对应 `video → text + 固定8帧`，不能与 processor 默认帧 baseline 混称。

### 3. 全部完成后评分

以 RoboReward video→text 为例：

```bash
python mydata_bench/run_my_dataset.py score-matrix \
  --records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/roboreward_8b_model_card_native_front_video_then_text/steering/records.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl \
  --selection-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/ranking_cohort/selection_manifest.json \
  --output-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/roboreward_8b_model_card_native_front_video_then_text/scoring \
  --expected-count 755
```

另外三个 variant 目录为：

- `roboreward_8b_native_front_text_then_video`；
- `qwen3vl_8b_video_then_text_attention8`；
- `grm_8b_multiview_endpoints`。

评分器会在 `scoring/` 写机器可读 artifacts，并在 variant 根目录写中文
`exp_record.md`。报告同时给出：

- all-including-rank-sources：755条，明确标记 in-sample overlap；
- common-unseen-S20：682条，作为跨 N 的 primary comparison；
- N-specific unseen：N=5/10/20 分别738/721/682条；
- ranking-source-only：N=5/10/20 分别17/34/73条；
- strict/proxy `grounding_strata` 的类别准确率与 intervention effect。

该矩阵没有 wrong-target、low-rank、layer-matched-random controls，不能支持
target-specific 或 head-specific causal specificity 声明。

## 当前主线：人工审核后重跑 N×K matrix

下面是当前 `grounding_auto_v2_low015_top40` 的人工审核与重跑流程。它只改变
grounding 是否经过人工确认，N×K 方法仍固定为 excess-mass、skip8、
N={5,10,20}、K={8,32,64}、bias=6、all-query。由于未经审核的矩阵结果已经被查看，
本轮必须报告为 **human-reviewed exploratory robustness rerun**，不能称为
confirmatory、formal 或预注册验证。

### 1. 冻结来源与输出隔离

在仓库根目录运行：

```bash
cd /mnt/public1/dais/workspace/Robo-Dopamine
conda activate robo-dopamine

sha256sum \
  mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_auto_v2_low015_top40/requests.jsonl \
  mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_auto_v2_low015_top40/proposals.jsonl
```

预期 SHA-256：

- requests：`b07b284aed78b3a078b093d6ddcf3b55296b291b19bf5585da16cf1470c18977`；
- proposals：`0f92ef98a486c1a152aaf38129caa7f9cf56ea509b15bfccad50cb3ba26e5db9`。

不要修改或覆盖 `grounding_auto_v2_low015_top40`、`attention_assumed`、
`ranking_cohort` 或 `exploratory_matrix`。本轮新产物分别写入
`grounding_reviewed_v1`、`attention_reviewed_v1`、
`ranking_cohort_reviewed_v1` 和 `reviewed_matrix_v1`。

### 2. 审满 755 条 grounding

把 `reviewer01` 替换成固定、非空且能追溯到审核者的 ID；同一输出目录以后必须继续
使用相同 ID：

```bash
python mydata_bench/review_sam3_grounding_web.py \
  --run-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_auto_v2_low015_top40 \
  --output-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_reviewed_v1 \
  --reviewer reviewer01 \
  --port 8766
```

服务只监听 `127.0.0.1`。本机打开 `http://127.0.0.1:8766`；远程机器需先通过
SSH 或 IDE 转发 8766 端口。页面不展示 reward 标签、模型预测或旧矩阵结果。每条记录：

- `eligible`：三个模型都必须确认 target 与 wrong-region；可选 SAM3 candidate，也可
  手动画框。target 和 wrong-region 的像素 bbox 不得有正面积重叠。
- `ineligible`：必须填写原因，不要为了凑数伪造 bbox。
- 页面只有在图像 SHA、尺寸和候选完全一致时才安全同步三个模型的选择；否则逐模型审。
- 必须对全部 755 条作出决定，但允许其中一部分为 `ineligible`。

页面把 SAM3 candidate bbox 叠加在冻结图像上，并列出候选 query/label 与 score；它不
读取或渲染 SAM3 mask，所以这里的人工确认不是 mask-level 审核。RoboReward/Qwen
页面也只展示一个冻结 terminal PNG。若 processor 的 temporal merge 把多个采样帧组成
同一个 token group（例如固定 8 帧时，末组可能同时包含倒数两帧），确认末帧 PNG 上的
bbox 不等价于也确认了该末组中的前一采样帧。因此本轮只能称为 terminal-image
grounding 已审核；若要解释整个末 temporal token group，后续还需逐采样帧/token-group
可视化 preflight。

`review_history.jsonl` 是 append-only 审核历史；`reviews.jsonl` 是按
`example_id` 原子物化的最新唯一决定。按 `Ctrl-C` 停止服务后，使用完全相同的
`--run-dir`、`--output-dir`、`--reviewer` 再启动即可从未完成处继续，不要手工拼接
或覆盖这些文件。

### 3. 严格审核与 reviewed attention manifests

审满后运行；`--proposals` 是必填参数：

```bash
python mydata_bench/run_my_dataset.py ground-audit \
  --requests mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_auto_v2_low015_top40/requests.jsonl \
  --proposals mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_auto_v2_low015_top40/proposals.jsonl \
  --reviews mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_reviewed_v1/reviews.jsonl \
  --output-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_reviewed_v1
```

`review_audit.json` 必须满足 `passed=true`、
`expected_count=request_count=review_count=755`，且 missing、unknown、duplicate、
invalid 均为空。`ineligible_example_count` 可以大于 0；这不妨碍 audit 通过。

随后生成 reviewed attention manifests：

```bash
python mydata_bench/run_my_dataset.py attention-prepare \
  --config mydata_bench/configs/my_dataset/attention_reviewed_ljx_lfz_cf_v1.yaml
```

每个模型的 `all.jsonl` 只包含所有逐条 eligible 样本，供冻结 S20 ranking source
取数；`complete_groups.jsonl` 只包含组内全部 counterfactual variants 都 eligible
的完整 groups。矩阵评估只能读取后者，不能把残缺 group 混入成组指标。实际评估样本数
由完整 groups 决定，因此后续配置和评分都使用 `auto`，不再假设是 755。

审核同时收集了 wrong bbox，但当前矩阵只消费 target bbox；本轮没有运行
wrong/random/low-rank controls，也没有完成 wrong-region 的 processor-token 等量且
不相交 preflight。像素框不重叠不能替代 token-level preflight，不能据此声称
target-specific 或 head-specific causal specificity。

### 4. 冻结 reviewed ranking cohort

```bash
python mydata_bench/run_my_dataset.py ranking-cohort \
  --config mydata_bench/configs/my_dataset/ranking_cohort_reviewed_ljx_lfz_cf_v1.yaml
```

该命令保留旧实验的固定、嵌套 S5⊂S10⊂S20 顺序，不根据审核结果重选。固定 S20 中
任意一条若为 ineligible，命令必须硬失败；不得用第 21 名或其他样本替换。只有确认属于
漏审/误审时才可回到审核历史修正。若样本确实不可审核，则原 S20 协议不能继续；如要换
cohort，必须另建协议版本、配置和输出目录。

### 5. 四个 reviewed matrix 与旧参照的绑定

| Variant | reviewed config | 新输出目录 | 逐例 baseline 参照 |
|---|---|---|---|
| RoboReward text→video | `exploratory_matrix_reviewed_roboreward_text_then_video.yaml` | `reviewed_matrix_v1/roboreward_8b_native_front_text_then_video` | `exploratory_matrix/roboreward_8b_native_front_text_then_video` |
| RoboReward video→text | `exploratory_matrix_reviewed_roboreward_video_then_text.yaml` | `reviewed_matrix_v1/roboreward_8b_model_card_native_front_video_then_text` | `exploratory_matrix/roboreward_8b_model_card_native_front_video_then_text` |
| Qwen video→text 8帧 | `exploratory_matrix_reviewed_qwen_video_then_text_attention8.yaml` | `reviewed_matrix_v1/qwen3vl_8b_video_then_text_attention8` | `exploratory_matrix/qwen3vl_8b_video_then_text_attention8` |
| GRM official 八图 | `exploratory_matrix_reviewed_grm.yaml` | `reviewed_matrix_v1/grm_8b_multiview_endpoints` | `exploratory_matrix/grm_8b_multiview_endpoints` |

#### Reviewed checkpoint 全内容校验

三个 checkpoint 的内容身份已经冻结在
`mydata_bench/configs/my_dataset/checkpoint_manifests/`。两个 RoboReward variant
共用同一份 RoboReward manifest；Qwen 和 GRM 各用一份。manifest 对 checkpoint
payload 中的每个普通文件（包括所有 `.safetensors` shard）记录完整 SHA-256，因此即使
大权重 shard 被替换为相同字节数，校验也会失败。只排除本地
`from_pretrained` 不读取、且可能被下载器异步改写的 `.cache/`。

冻结的 content fingerprint 为：

- RoboReward-8B：`73022aae0a16509737cbef22e759d7df9b35afb4bb556016db168391cee2ca47`；
- Qwen3-VL-8B-Instruct：`e17e6dfc4668ad809e3ba6467f0ae0204cd2640938a88fdcec6200ad121e9c81`；
- Robo-Dopamine-GRM-2.0-8B-Preview：`c5f0dd90db13d18d6e0e272e80d8edee8c5c74293614f2f19e7aa86976e6fe4d`。

reviewed matrix 在加载 processor/model 之前会执行同样的 full-byte verify。也可以先手动
执行以下只读检查；命令不会加载 GPU 模型，也不会修改 checkpoint 或 manifest：

```bash
python mydata_bench/freeze_checkpoint_manifest.py verify \
  --model-path /mnt/public1/dais/workspace/model/RoboReward-8B \
  --manifest mydata_bench/configs/my_dataset/checkpoint_manifests/roboreward_8b.json

python mydata_bench/freeze_checkpoint_manifest.py verify \
  --model-path /mnt/public1/dais/workspace/model/Qwen3-VL-8B-Instruct \
  --manifest mydata_bench/configs/my_dataset/checkpoint_manifests/qwen3_vl_8b_instruct.json

python mydata_bench/freeze_checkpoint_manifest.py verify \
  --model-path /mnt/public1/dais/workspace/Robo-Dopamine/pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview \
  --manifest mydata_bench/configs/my_dataset/checkpoint_manifests/robo_dopamine_grm_2_0_8b_preview.json
```

不要再次运行 `freeze` 覆盖这三份 manifest。若确实更换了模型或任一 checkpoint
payload，必须先确认变更来源，再新建协议版本、manifest、reviewed config 和输出目录；
不能通过重冻 manifest 让旧协议继续写入原 `reviewed_matrix_v1`。

三张 GPU 可按下面方式调度。GPU 0 的第一个任务完成后再在 GPU 0 跑第四个任务；
日志和结果都与旧矩阵隔离：

```bash
mkdir -p mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1

CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/exploratory_matrix_reviewed_roboreward_video_then_text.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/roboreward_video_text.log 2>&1 &
reviewed_rr_vt_pid=$!

CUDA_VISIBLE_DEVICES=1 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/exploratory_matrix_reviewed_qwen_video_then_text_attention8.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/qwen_video_text_attention8.log 2>&1 &
reviewed_qwen_pid=$!

CUDA_VISIBLE_DEVICES=3 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/exploratory_matrix_reviewed_grm.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/grm.log 2>&1 &
reviewed_grm_pid=$!

wait "$reviewed_rr_vt_pid"

CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/exploratory_matrix_reviewed_roboreward_text_then_video.yaml \
  > mydata_bench/logs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/roboreward_text_video.log 2>&1 &
reviewed_rr_tv_pid=$!

wait "$reviewed_qwen_pid" "$reviewed_grm_pid" "$reviewed_rr_tv_pid"
```

同一配置重跑会按 append-only latest-state 语义续跑并跳过成功记录。只有检查清楚最新
invalid 的原因后才加 `--retry-failed`。不要删除旧目录、不要把 reviewed config 改回
旧 `exploratory_matrix` 输出目录，也不要让不同代码/模型/输入继续写同一结果目录。

### 6. 先把四份旧矩阵限制到相同 complete-group IDs 重评分

不能把 reviewed 子集分数直接减去旧 755 条总分。以下四条命令只过滤旧 records 并在
新隔离目录写指标，不会覆盖原 `exploratory_matrix` 的评分：

```bash
python mydata_bench/run_my_dataset.py score-matrix \
  --records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/roboreward_8b_native_front_text_then_video/steering/records.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl \
  --selection-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/ranking_cohort_reviewed_v1/selection_manifest.json \
  --evaluation-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/attention_reviewed_v1/roboreward/complete_groups.jsonl \
  --output-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/unreviewed_same_population_v1/roboreward_8b_native_front_text_then_video/scoring \
  --expected-count auto

python mydata_bench/run_my_dataset.py score-matrix \
  --records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/roboreward_8b_model_card_native_front_video_then_text/steering/records.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl \
  --selection-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/ranking_cohort_reviewed_v1/selection_manifest.json \
  --evaluation-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/attention_reviewed_v1/roboreward/complete_groups.jsonl \
  --output-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/unreviewed_same_population_v1/roboreward_8b_model_card_native_front_video_then_text/scoring \
  --expected-count auto

python mydata_bench/run_my_dataset.py score-matrix \
  --records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/qwen3vl_8b_video_then_text_attention8/steering/records.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl \
  --selection-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/ranking_cohort_reviewed_v1/selection_manifest.json \
  --evaluation-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/attention_reviewed_v1/qwen/complete_groups.jsonl \
  --output-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/unreviewed_same_population_v1/qwen3vl_8b_video_then_text_attention8/scoring \
  --expected-count auto

python mydata_bench/run_my_dataset.py score-matrix \
  --records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/grm_8b_multiview_endpoints/steering/records.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl \
  --selection-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/ranking_cohort_reviewed_v1/selection_manifest.json \
  --evaluation-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/attention_reviewed_v1/grm/complete_groups.jsonl \
  --output-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/unreviewed_same_population_v1/grm_8b_multiview_endpoints/scoring \
  --expected-count auto
```

### 7. 评分 reviewed matrix，并执行逐例 baseline parity gate

reviewed records 的评分必须同时传 `--expected-count auto`、
`--evaluation-manifest` 和同 variant 的旧 `--reference-records`：

```bash
python mydata_bench/run_my_dataset.py score-matrix \
  --records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/roboreward_8b_native_front_text_then_video/steering/records.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl \
  --selection-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/ranking_cohort_reviewed_v1/selection_manifest.json \
  --evaluation-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/attention_reviewed_v1/roboreward/complete_groups.jsonl \
  --reference-records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/roboreward_8b_native_front_text_then_video/steering/records.jsonl \
  --output-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/roboreward_8b_native_front_text_then_video/scoring \
  --expected-count auto

python mydata_bench/run_my_dataset.py score-matrix \
  --records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/roboreward_8b_model_card_native_front_video_then_text/steering/records.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl \
  --selection-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/ranking_cohort_reviewed_v1/selection_manifest.json \
  --evaluation-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/attention_reviewed_v1/roboreward/complete_groups.jsonl \
  --reference-records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/roboreward_8b_model_card_native_front_video_then_text/steering/records.jsonl \
  --output-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/roboreward_8b_model_card_native_front_video_then_text/scoring \
  --expected-count auto

python mydata_bench/run_my_dataset.py score-matrix \
  --records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/qwen3vl_8b_video_then_text_attention8/steering/records.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl \
  --selection-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/ranking_cohort_reviewed_v1/selection_manifest.json \
  --evaluation-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/attention_reviewed_v1/qwen/complete_groups.jsonl \
  --reference-records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/qwen3vl_8b_video_then_text_attention8/steering/records.jsonl \
  --output-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/qwen3vl_8b_video_then_text_attention8/scoring \
  --expected-count auto

python mydata_bench/run_my_dataset.py score-matrix \
  --records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/grm_8b_multiview_endpoints/steering/records.jsonl \
  --labels mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/scoring/labels.jsonl \
  --selection-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/ranking_cohort_reviewed_v1/selection_manifest.json \
  --evaluation-manifest mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/attention_reviewed_v1/grm/complete_groups.jsonl \
  --reference-records mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/grm_8b_multiview_endpoints/steering/records.jsonl \
  --output-dir mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/grm_8b_multiview_endpoints/scoring \
  --expected-count auto
```

评分器会在每个 complete-group example 上比较新旧 shared no-hook baseline 的原始输出、
预测、signed score 和 progress 等字段；任何不一致都会硬失败，表示可能存在
runtime/code drift，此时不能解释 steering 差异，也不能换一个旧 variant 绕过 gate。
只有 parity 通过后，才比较 reviewed 指标与
`unreviewed_same_population_v1` 中同 ID population 的指标。不要与旧 755 条总分直接作差。

## 当前新主线：首帧锁定 + official SAM3 全视频 tracking v2

旧 `grounding_reviewed_v1` 只审核 terminal image，不能证明首帧与 terminal 是同一
实例。tracking v2 使用独立目录，并遵守
`mydata_bench/configs/my_dataset/tracking_v2_protocol.md`；不会覆盖旧 artifact。
完整、可复制的运行记录见
`mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exp_use.md` 第 15 节。

`ground-track-run`、`ground-track-manual` 和 `matrix` 会加载模型并执行 GPU
推理；`ground-track-prepare`、审核 Web、`ground-audit`、
`attention-prepare`、`ranking-cohort` 和 `score-matrix` 不加载模型，但会写
artifact。环境/path/SHA/JSON 检查只读。

### 1. 独立环境与无推理预检

~~~bash
conda env create --file mydata_bench/environments/rewardbench-sam3.yml
# 若环境已存在，改用：
# conda env update --name rewardbench-sam3 --file mydata_bench/environments/rewardbench-sam3.yml --prune
conda activate rewardbench-sam3

test -f /mnt/public1/dais/workspace/model/sam3/model.safetensors
test -f /mnt/public1/dais/workspace/model/sam3/sam3.pt
test -f /mnt/public1/dais/workspace/cap-x/capx/third_party/sam3/sam3/model_builder.py
test -f /mnt/public1/dais/workspace/cap-x/capx/third_party/sam3/sam3/model/sam3_video_predictor.py

python -c "import psutil, sam3; from sam3.model_builder import build_sam3_video_predictor; print('official editable SAM3 import: OK')"
~~~

环境清单把 official SAM3 源码以 editable package 安装，并显式包含 `psutil` 和
`setuptools<81`。上述预检只导入模块，不构造 predictor，也不读取 checkpoint。

### 2. 冻结 request，再运行自动 proposal + tracking

配置中的 `roboreward_content_order_runs` 同时绑定 text→video 的
`roboreward_8b_native_front` 与 video→text 的
`roboreward_8b_model_card_native_front`。`ground-track-prepare` 会逐例比较两条 run 的
source video、sampled frame indices、`video_grid_thw`、frame count、尺寸、fps 与
terminal 绑定；只有完全相同，两个 RoboReward matrix 才能共用 tracked bbox。任何
mismatch 都会硬失败，此时必须按内容顺序拆成独立 tracking。此前对 755 条的只读预检
为 frames/grid/video 0 mismatch，但运行时门禁仍不可跳过。

~~~bash
python mydata_bench/run_my_dataset.py ground-track-prepare \
  --config mydata_bench/configs/my_dataset/grounding_tracking_ljx_lfz_cf_v2.yaml

CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py ground-track-run \
  --config mydata_bench/configs/my_dataset/grounding_tracking_ljx_lfz_cf_v2.yaml
~~~

第二条才会执行 GPU 推理。普通 resume 重复同一命令；确认 latest invalid 原因后，
才可在命令末尾加 `--retry-failed`。输出固定为
`grounding_tracking_v2/{requests.jsonl,tracks.jsonl,manifest.json}`。
所有普通 resume 与失败重试必须在首次启动审核 Web 前完成。审核 session 一旦在
`grounding_tracking_reviewed_v2` 建立，就不得再修改或重跑该版本的
`tracks.jsonl` / `manifest.json`；若必须修正，另立版本化 tracking/review 目录。

### 3. 首轮人工审核

~~~bash
python mydata_bench/review_sam3_grounding_web.py \
  --mode tracking_v2 \
  --run-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_v2 \
  --output-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_reviewed_v2 \
  --reviewer REPLACE_WITH_FIXED_REVIEWER_ID \
  --port 8766
~~~

同一 output-dir 必须始终使用同一个 reviewer ID。状态/source 只有：

- `eligible`：`accept_default`、`select_alternative` 或 `accept_manual_track`；
- `needs_retrack`：`manual_first_bbox`；
- `skipped`：只能是 `reviewer_skip`。

禁止自由文本、`wrong_region` 和旧 skip code。无 algorithmic default 时，绿色框只是
人工预选，保存仍是 `select_alternative`。首轮先审完全部样本，再进入批量 retrack。

### 4. 批量 manual retrack 与二次确认

只有 `manual_anchors.jsonl` 非空时运行：

~~~bash
test -s mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_reviewed_v2/manual_anchors.jsonl

CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py ground-track-manual \
  --config mydata_bench/configs/my_dataset/grounding_tracking_ljx_lfz_cf_v2.yaml \
  --anchors mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_reviewed_v2/manual_anchors.jsonl \
  --output mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_v2/manual_tracks.jsonl
~~~

完成全部失败重试后，用第 3 节完全相同的 Web 命令重启。所有 `needs_retrack` 会再次
进入 pending；查看 keyframes 和三模型 terminal 后，才能批准 manual track。
`manual_tracks.jsonl` 按整文件 SHA 冻结：第一条 `accept_manual_track` 保存后，不得
再重画、append、重试或重写该文件；需要修正时必须另立版本。

### 5. 严格 audit 与 SHA 回填

没有任何 `accept_manual_track` 时：

~~~bash
python mydata_bench/run_my_dataset.py ground-audit \
  --requests mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_v2/requests.jsonl \
  --reviews mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_reviewed_v2/reviews.jsonl \
  --output-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_reviewed_v2 \
  --tracking-artifact mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_v2/tracks.jsonl
~~~

只要存在 `accept_manual_track`，改用下面这条完整命令：

~~~bash
python mydata_bench/run_my_dataset.py ground-audit \
  --requests mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_v2/requests.jsonl \
  --reviews mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_reviewed_v2/reviews.jsonl \
  --output-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_reviewed_v2 \
  --tracking-artifact mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_v2/tracks.jsonl \
  --manual-tracking-artifact mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_v2/manual_tracks.jsonl
~~~

必须确认 `review_audit.json` 中 `passed=true`。然后只把下列四项回填到
`attention_tracking_reviewed_ljx_lfz_cf_v2.yaml`：

- `expected_requests_sha256`；
- `expected_tracking_artifact_sha256`；
- `expected_tracking_manifest_sha256`；
- `expected_manual_tracking_artifact_sha256`：没有批准 manual track 时写 YAML `null`，否则写
  audit 中真实的整文件 SHA；
- 同时把配置的 `manual_tracking_artifact_path` 保持为 `null`，或在采用 manual track
  时改为 `grounding_tracking_v2/manual_tracks.jsonl` 的固定绝对路径。

配置中的 `REQUIRED_AFTER_AUDIT_*` 是故意无效的 fail-closed 占位；未回填时不能
运行 attention。reviews/audit 的 SHA 与 fingerprint 由构建器重算并传播，不手填。

### 6. Attention、固定 cohort 与四个 reviewed matrix

~~~bash
python mydata_bench/run_my_dataset.py attention-prepare \
  --config mydata_bench/configs/my_dataset/attention_tracking_reviewed_ljx_lfz_cf_v2.yaml

python mydata_bench/run_my_dataset.py ranking-cohort \
  --config mydata_bench/configs/my_dataset/ranking_cohort_tracking_reviewed_ljx_lfz_cf_v2.yaml
~~~

四个 matrix 配置分别为：

- `exploratory_matrix_tracking_reviewed_roboreward_text_then_video.yaml`；
- `exploratory_matrix_tracking_reviewed_roboreward_video_then_text.yaml`；
- `exploratory_matrix_tracking_reviewed_qwen_video_then_text_attention8.yaml`；
- `exploratory_matrix_tracking_reviewed_grm.yaml`。

~~~bash
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/exploratory_matrix_tracking_reviewed_roboreward_text_then_video.yaml

CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/exploratory_matrix_tracking_reviewed_roboreward_video_then_text.yaml

CUDA_VISIBLE_DEVICES=1 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/exploratory_matrix_tracking_reviewed_qwen_video_then_text_attention8.yaml

CUDA_VISIBLE_DEVICES=3 python mydata_bench/run_my_dataset.py matrix \
  --config mydata_bench/configs/my_dataset/exploratory_matrix_tracking_reviewed_grm.yaml
~~~

它们统一读取 `attention_tracking_reviewed_v2` 与
`ranking_cohort_tracking_reviewed_v2`，只写 `reviewed_tracking_matrix_v2`。固定 S20
若包含 skipped source，cohort 必须硬失败，不能用第 21 名以后替换。
四套完整 `score-matrix` 命令见 `exp_use.md` 第 15.10 节；它们固定同 population
和对应旧 variant，并强制 shared no-hook baseline 逐例 parity。
