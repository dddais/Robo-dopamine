# mydata_bench：ljx_lfz_cf_v1 评测与 tracking-v2 主线

本目录只服务本地数据集 `ljx_lfz_cf_v1`。所有新实验统一从
`python mydata_bench/run_my_dataset.py <command>` 进入，不再使用复制自
`rewardbench` 的旧入口。

## 当前状态

- 四个 tracking-v2 输入 baseline 已完成，均为 755/755、invalid=0。
- `ground-track-prepare`、SAM3 proposal 和 official full-video tracking 已完成；
  最终 manifest 为 `complete`，latest 755 条中
  `ok=360`、`needs_review=352`、`invalid=43`。
- 43 条 invalid 全部是首帧 `No valid target proposal`，后续由人工首帧框进入
  manual retrack；没有 CUDA 或 tracker 系统失败。
- 人工审核、reviewed attention、ranking cohort 和四个 reviewed matrix 尚未运行。
- 旧 assumed/proxy grounding 矩阵只作为未经人工审核的 reference；不得作为正式
  reviewed 结论。

## 权威文档

- 协议约束：`configs/my_dataset/tracking_v2_protocol.md`。
- 逐步复现命令：`outputs/my_dataset/ljx_lfz_cf_v1/exp_use.md`。
- 已完成结果：`outputs/my_dataset/ljx_lfz_cf_v1/exp_record.md`。
- 旧 exploratory head 统计：`outputs/my_dataset/ljx_lfz_cf_v1/ranking_summary.md`。

协议文档优先于运行说明；配置与文档冲突时必须先停下来修正，不能绕过 fail-closed gate。

## 保留的代码结构

- `run_my_dataset.py`：唯一评测 CLI。
- `review_sam3_grounding_web.py`：tracking-v2 人工审核 Web。
- `my_dataset/`：数据、tracking、审核、attention、cohort、matrix 与评分逻辑。
- `roboreward_eval/`、`qwen_eval/`、`raw_eval/`：三个模型的最小 baseline/runtime。
- `attention_eval/`：共享 attention masking/runtime。
- `grounding/`：SAM3 grounder runtime。
- `tests/`：主线回归测试。

## 当前有效配置

数据与 baseline：

- `prepare_ljx_lfz.yaml`
- `roles_ljx_lfz_cf_v1.yaml`
- `split_ljx_lfz_cf_v1.yaml`
- `roboreward_8b_native_front.yaml`（text→video）
- `roboreward_8b_model_card_native_front.yaml`（video→text）
- `qwen3vl_8b_model_card_native_front_attention8.yaml`（video→text，最多 8 帧）
- `grm_8b_multiview_endpoints.yaml`（official canonical 八图）

tracking-v2：

- `grounding_tracking_ljx_lfz_cf_v2.yaml`
- `attention_tracking_reviewed_ljx_lfz_cf_v2.yaml`
- `ranking_cohort_tracking_reviewed_ljx_lfz_cf_v2.yaml`
- 四个 `exploratory_matrix_tracking_reviewed_*.yaml`
- `checkpoint_manifests/*.json`

以下旧配置仅用于复现已冻结的 unreviewed reference，不是当前主线：

- `grounding_exploratory_ljx_lfz_cf_v1.yaml`
- `assumed_grounding_ljx_lfz_cf_v1.yaml`
- `attention_assumed_ljx_lfz_cf_v1.yaml`
- `ranking_cohort_ljx_lfz_cf_v1.yaml`
- 四个不带 `tracking_reviewed` 的 `exploratory_matrix_*.yaml`

## 环境

模型 baseline/matrix 使用：

```bash
cd /mnt/public1/dais/workspace/Robo-Dopamine
conda activate robo-dopamine
```

SAM3 tracking 使用独立环境：

```bash
conda env create --file mydata_bench/environments/rewardbench-sam3.yml
conda activate rewardbench-sam3
```

不要在 SAM3 环境运行 RoboReward/Qwen/GRM matrix，也不要在模型环境临时拼装
SAM3 依赖。

## 数据审计

```bash
python mydata_bench/run_my_dataset.py audit --prepared-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1
python mydata_bench/run_my_dataset.py inventory --inputs mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/model_inputs/inputs.jsonl
```

预期为 755 条样本、169 个 source groups。

## 四个 baseline

只有需要完整复跑时才执行；已有输出不要原地覆盖。

```bash
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py run --config mydata_bench/configs/my_dataset/roboreward_8b_native_front.yaml
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py run --config mydata_bench/configs/my_dataset/roboreward_8b_model_card_native_front.yaml
CUDA_VISIBLE_DEVICES=1 python mydata_bench/run_my_dataset.py run --config mydata_bench/configs/my_dataset/qwen3vl_8b_model_card_native_front_attention8.yaml
CUDA_VISIBLE_DEVICES=3 python mydata_bench/run_my_dataset.py run --config mydata_bench/configs/my_dataset/grm_8b_multiview_endpoints.yaml
```

这四个输出既是 standalone baseline，也是 tracking-v2 的 processor/frame contract。
尤其不要删除 `grm_8b_multiview_endpoints/frames/`。

## tracking-v2 流程

### 1. 冻结 requests

```bash
python mydata_bench/run_my_dataset.py ground-track-prepare --config mydata_bench/configs/my_dataset/grounding_tracking_ljx_lfz_cf_v2.yaml
```

该步必须保持 label-free，并逐例验证两种 RoboReward content order、Qwen 8 帧和
GRM endpoint 输入契约。

### 2. SAM3 自动 proposal 与视频追踪

```bash
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py ground-track-run --config mydata_bench/configs/my_dataset/grounding_tracking_ljx_lfz_cf_v2.yaml
```

普通 resume 使用相同命令；只有检查失败原因后才添加 `--retry-failed`。审核一旦开始，
不得再原地改写自动 tracks/manifest。

多卡分片的四条实际命令以及分片完成后必须执行的
`ground-track-reconcile`、`ground-track-finalize` 见 `exp_use.md`。

### 3. 人工审核

```bash
python mydata_bench/review_sam3_grounding_web.py --mode tracking_v2 --run-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_v2 --output-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_reviewed_v2 --reviewer REPLACE_WITH_FIXED_REVIEWER_ID --port 8766
```

默认框正确时直接接受；不正确时选择候选或提供首帧手工框。`wrong-region` 不由人工
填写。需要 retrack 的样本先统一冻结 `manual_anchors.jsonl`，再批量传播并二次确认。

### 4. 可选 manual retrack

```bash
CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py ground-track-manual --config mydata_bench/configs/my_dataset/grounding_tracking_ljx_lfz_cf_v2.yaml --anchors mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_reviewed_v2/manual_anchors.jsonl --output mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_v2/manual_tracks.jsonl
```

manual phase barrier、两种 `ground-audit` 命令以及 SHA 回填方式见 `exp_use.md`。

### 5. attention、ranking 与 matrix

审核 audit 通过并回填真实 SHA 后：

```bash
python mydata_bench/run_my_dataset.py attention-prepare --config mydata_bench/configs/my_dataset/attention_tracking_reviewed_ljx_lfz_cf_v2.yaml
python mydata_bench/run_my_dataset.py ranking-cohort --config mydata_bench/configs/my_dataset/ranking_cohort_tracking_reviewed_ljx_lfz_cf_v2.yaml
```

随后分别运行四个 `exploratory_matrix_tracking_reviewed_*.yaml`。完整 matrix 和
`score-matrix` 命令统一记录在 `exp_use.md`，评分必须读取对应旧
`exploratory_matrix/*/steering/records.jsonl` 并通过逐例 no-hook parity gate。

## 不可删除的 reference

在四个 reviewed matrix 正式评分完成前，不要删除：

- `outputs/my_dataset/ljx_lfz_cf_v1/exploratory_matrix/*/steering/records.jsonl`；
- `artifacts/my_dataset/ljx_lfz_cf_v1/ranking_cohort/`；
- 旧 exploratory 的 ranking JSON、selection manifest、metrics 与 grounding provenance。

## 测试

```bash
python3 -m pytest mydata_bench/tests -q
```

单测主要验证数据契约与 orchestration；涉及真实 checkpoint/SAM3 的改动还必须做
最小真实 smoke test。
