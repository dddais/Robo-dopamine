# mydata_bench：ljx_lfz_cf_v1 的 v3 更正实验主线

本目录只服务本地数据集 `ljx_lfz_cf_v1`。统一入口是：

```bash
python mydata_bench/run_my_dataset.py <command>
```

完整可复制命令见
`mydata_bench/outputs/my_dataset/ljx_lfz_cf_v1/exp_use.md`，实验语义与硬门见
`tracking_v3_protocol.md`。

## 当前状态

- 四个 label-free baseline 已完成，可作为 v3 的 processor/frame 输入契约。
- v2 自动 tracking 已完成，但 v2 的关系/顺序实例选择、全程审核覆盖和原生视频
  temporal-patch target 语义存在需要更正的问题。
- v2 代码、requests、tracks、审核与输出保留为历史证据，不再原地修改或继续写入。
- v3 代码和配置已经独立建立；真实 v3 SAM3 tracking、人工审核、attention、
  ranking cohort 和 matrix 尚需按 `exp_use.md` 重新执行。
- 旧 assumed/proxy grounding 以及 v2 结果只能作为 exploratory reference，不能由
  代码修复追溯性地变成 v3 结论。

## “冻结”、resume、provenance 和 cache

冻结不是冻结模型权重。这里指某个实验版本的输入、代码、配置、人工审核和结果形成一条
不可原地改写的证据链。

最小使用规则只有三条：

1. 同一 fingerprint、同一 output directory：允许 resume 缺失工作。
2. 输入、配置、代码、checkpoint、关键运行库版本、visual scope 或审核产物改变：
   使用新的 output
   directory。
3. SAM3 track cache 只在 video SHA、首帧 bbox/mask、tracker、quality contract 和
   required review frames 全部相同时复用。

`expected_*_sha256: from_audit` 不要求手工复制 SHA。程序会从已经通过的
`review_audit.json` 读取预期值，再重新计算配置路径上实际文件的 SHA。它的作用只是
防止同名 example ID 下混入旧 requests、tracks、reviews 或结果，不改变模型输入。

## v2 与 v3

### v2：只读历史证据

- tracking：`grounding_tracking_v2/`
- review：`grounding_tracking_reviewed_v2/`
- attention：`attention_tracking_reviewed_v2/`
- ranking：`ranking_cohort_tracking_reviewed_v2/`
- matrix：`reviewed_tracking_matrix_v2/`
- 源码：`my_dataset/tracked_grounding.py`

不要删除，也不要把新的运行结果写进这些目录。

### v3：当前更正主线

- tracking 配置：`grounding_tracking_ljx_lfz_cf_v3.yaml`
- tracking：`grounding_tracking_v3/`
- review：`grounding_tracking_reviewed_v3/`
- attention 配置：`attention_tracking_reviewed_ljx_lfz_cf_v3.yaml`
- attention：`attention_tracking_reviewed_v3/`
- ranking 配置：`ranking_cohort_tracking_reviewed_ljx_lfz_cf_v3.yaml`
- ranking：`ranking_cohort_tracking_reviewed_v3/`
- matrix：四个文件名以 `_v3.yaml` 结尾，输出到
  `reviewed_tracking_matrix_v3/`
- 源码：`my_dataset/tracked_grounding_v3.py`

v3 不继承 v2 的人工通过状态。

## SAM3 v3 审核语义

SAM3 image proposer 只在首帧提供 target/reference 候选。对于 ordinal、left/right、
closest/farthest 或 `requires_instance_review=true` 的样本：

- 算法几何结果只能作为提示；
- 必须先由人确认正确的首帧具体实例；
- 在确认前不自动传播所有候选；
- 人工选择/画出的首帧框必须重新进行 full-video tracking；
- retrack 后必须进行第二阶段人工审核。

即使首帧正确，locked obj_id 也不能证明后续语义身份没有漂移。页面会展示 processor
sampled frames、首尾帧、默认 16 个均匀全视频帧和漂移警告帧。应重点检查换到邻近同类
实例、粘到机械臂、目标消失后错误重现、面积突变与终帧错误。

## Qwen/RoboReward 多帧 target span

原生 Qwen3-VL 视频不是“一个连续 token run 对应一帧”。当前官方 processor 为每个
temporal patch 生成一个 timestamp-delimited video-token run，当前模型的位置编码也
按这些独立 run 拆分 `video_grid_thw`。v3 因此只接受“一 patch 一 run”的官方布局；
多 patch 单连续 run 即使总 token 数相同也会失败闭合。

v3 验证：

- processor sampled frame indices；
- `video_grid_thw`；
- model/processor `temporal_patch_size`；
- token run 数和每个 run 的 token 数；
- 最后 temporal patch 的 source-frame 组成及末帧 padding。

Qwen/RoboReward target 是最后 temporal patch 中所有可见 source frames 的 tracked
bbox token 并集，不是孤立的最后一帧。GRM target 仍是单个
`after_cam_high` 图像槽。

## Steering 条件

主矩阵每个样本共有 28 个 paired conditions：

- 1 个 baseline；
- 9 个 candidate-target：N={5,10,20} × K={8,32,64}；
- 9 个 candidate-wrong-region；
- 9 个 layer-matched low-rank-target。

wrong region 与 target 位于同一 target span，token footprint/count 完全一致、互不
重叠；没有合法平移时失败闭合。low-rank heads 与 candidate heads 不重叠，并匹配每层
head 数。

可选 visual scope：

- `target_slot_only`：GRM 只作用于 after_cam_high；原生视频只作用于最后 temporal
  patch。这是 v3 主配置默认值。
- `all_visual`：GRM 作用于八图全部视觉 token；原生视频作用于整个视频视觉 token。

运行 `all_visual` 消融时，复制对应 v3 matrix 配置并同时修改：

```yaml
intervention_visual_scope: all_visual
variant_id: ..._all_visual_v3
output_dir: .../reviewed_tracking_matrix_v3_all_visual/...
```

不要复用主矩阵 output directory。相同 bias/K 在不同模型中不是等剂量干预，跨模型
只比较各自内部 paired contrasts，不用 raw shift 直接排序模型依赖强弱。

## 有效配置

数据和 baseline：

- `prepare_ljx_lfz.yaml`
- `roles_ljx_lfz_cf_v1.yaml`
- `split_ljx_lfz_cf_v1.yaml`
- `roboreward_8b_native_front.yaml`
- `roboreward_8b_model_card_native_front.yaml`
- `qwen3vl_8b_model_card_native_front_attention8.yaml`
- `grm_8b_multiview_endpoints.yaml`

v3：

- `grounding_tracking_ljx_lfz_cf_v3.yaml`
- `attention_tracking_reviewed_ljx_lfz_cf_v3.yaml`
- `ranking_cohort_tracking_reviewed_ljx_lfz_cf_v3.yaml`
- `exploratory_matrix_tracking_reviewed_roboreward_text_then_video_v3.yaml`
- `exploratory_matrix_tracking_reviewed_roboreward_video_then_text_v3.yaml`
- `exploratory_matrix_tracking_reviewed_qwen_video_then_text_attention8_v3.yaml`
- `exploratory_matrix_tracking_reviewed_grm_v3.yaml`
- `checkpoint_manifests/*.json`

## 环境

模型 baseline/matrix：

```bash
cd /mnt/public1/dais/workspace/Robo-Dopamine
conda activate robo-dopamine
```

SAM3：

```bash
conda env create --file mydata_bench/environments/rewardbench-sam3.yml
conda activate rewardbench-sam3
```

不要混用两个环境。

## 快速入口

```bash
python mydata_bench/run_my_dataset.py ground-track-prepare \
  --config mydata_bench/configs/my_dataset/grounding_tracking_ljx_lfz_cf_v3.yaml

CUDA_VISIBLE_DEVICES=0 python mydata_bench/run_my_dataset.py ground-track-run \
  --config mydata_bench/configs/my_dataset/grounding_tracking_ljx_lfz_cf_v3.yaml

python mydata_bench/review_sam3_grounding_web.py \
  --mode tracking \
  --run-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_v3 \
  --output-dir mydata_bench/artifacts/my_dataset/ljx_lfz_cf_v1/grounding_tracking_reviewed_v3 \
  --reviewer REPLACE_WITH_FIXED_REVIEWER_ID \
  --port 8766
```

完整的分片、reconcile、finalize、manual retrack、audit、attention、ranking、四个 matrix
和 scoring 命令见 `exp_use.md`。

## 测试边界

```bash
python3 -m pytest mydata_bench/tests -q
```

单元测试证明的是数据契约、token 索引、mask 语义和 orchestration。它不能替代：

- 真实 SAM3 轨迹的人工检查；
- 实际 processor/token span 可视化；
- 真实模型 prefill/decode hook trace；
- v3 全量实验重跑。
