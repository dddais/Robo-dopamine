# mydata_bench 使用说明

所有命令从仓库根目录运行：

    cd /home/dais/workspace/Robo-Dopamine

## 配置

新数据使用以下配置：

- baseline_01_roboreward_text_video.yaml
- baseline_02_roboreward_video_text.yaml
- baseline_03_qwen_text_video.yaml
- baseline_04_qwen_video_text.yaml
- baseline_05_grm_forward.yaml
- baseline_06_grm_incremental.yaml
- attention_06_roboreward_last_frame.yaml
- attention_07_roboreward_all_frames.yaml
- attention_08_qwen_last_frame.yaml
- attention_09_qwen_all_frames.yaml
- attention_10_grm_forward_after.yaml
- attention_11_grm_forward_before_after.yaml
- attention_12_grm_incremental_after.yaml
- attention_13_grm_incremental_before_after.yaml
- mydata_grounding.yaml
- mydata_ranking_grounding.yaml

以上文件均位于 `mydata_bench/configs/`。正文编号虽然把 baseline 6
和 attention 6 重复使用，但明确列出了 6 个 baseline 加 8 个 attention，
因此实际执行 14 项。输出统一写到 `results/mydata_bench/experiments/`。

## 快速自检

    python3 -m pytest mydata_bench/tests/test_mydata_bench.py -q
    python3 -m compileall -q mydata_bench

pytest 使用系统 Python（`robo-dopamine` 环境未安装 pytest）。最终完整测试为
`21 passed`；验收命令见本文末尾，真实数据量为 755/169/586。

## 1. 独立 ranking_data grounding

这 30 条数据作为三个模型各自的 attention head ranking 输入，并使用独立的
grounding/ranking artifact。注意：当前 30 个 ID 全部也属于最终 336 条 attention
cohort，因此两者不是样本互斥集合；严格样本外实验需要另建按视频 hash 互斥的 ranking 集。

    python mydata_bench/run_grounding.py parse \
      --config mydata_bench/configs/mydata_ranking_grounding.yaml

    conda run -n rewardbench-sam3 python mydata_bench/run_grounding.py run \
      --backend sam3 \
      --config mydata_bench/configs/mydata_ranking_grounding.yaml

结果位于 results/mydata_bench/ranking_grounding。配置默认不生成 tracking preview，以节约磁盘，但仍执行首帧 bbox 加全程 tracking。

## 2. 评测集自动 grounding 与 cohort

本轮按 `exp_plan.md` 跳过人工审核，直接把 SAM3 首帧定位加后续 tracking
成功的双端点样本视为自动 eligible。先解析完整 755 条：

    python mydata_bench/run_grounding.py parse \
      --config mydata_bench/configs/mydata_grounding.yaml

可在三张 GPU 上并行运行稳定 hash 分片：

    CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n rewardbench-sam3 \
      python mydata_bench/run_grounding.py run --backend sam3 \
      --config mydata_bench/configs/mydata_grounding.yaml --retry-failed \
      --shard-id 0 --num-shards 3

    CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n rewardbench-sam3 \
      python mydata_bench/run_grounding.py run --backend sam3 \
      --config mydata_bench/configs/mydata_grounding.yaml --retry-failed \
      --shard-id 1 --num-shards 3

    CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n rewardbench-sam3 \
      python mydata_bench/run_grounding.py run --backend sam3 \
      --config mydata_bench/configs/mydata_grounding.yaml --retry-failed \
      --shard-id 2 --num-shards 3

三个分片都结束后会确定性合并为 `grounding.jsonl`。随后冻结自动 cohort：

    python mydata_bench/prepare_reward_cohort.py \
      --dataset-root /home/dais/workspace/data/ljx_lfz_task/new \
      --split all \
      --grounding-run results/mydata_bench/grounding/sam3 \
      --output-dir results/mydata_bench/cohorts/auto_grounded

该 cohort 是未经人工审核的 exploratory 集合，不能表述为 audited 结果。

## 3. Baseline

RoboReward-8B 的两种 content order：

    conda run -n robo-dopamine python mydata_bench/run_roboreward_eval.py run --config mydata_bench/configs/baseline_01_roboreward_text_video.yaml
    conda run -n robo-dopamine python mydata_bench/run_roboreward_eval.py score --run-dir results/mydata_bench/experiments/baseline_01_roboreward_text_video

    conda run -n robo-dopamine python mydata_bench/run_roboreward_eval.py run --config mydata_bench/configs/baseline_02_roboreward_video_text.yaml
    conda run -n robo-dopamine python mydata_bench/run_roboreward_eval.py score --run-dir results/mydata_bench/experiments/baseline_02_roboreward_video_text

Qwen3-VL-8B 的两种 content order：

    conda run -n robo-dopamine python mydata_bench/run_qwen_eval.py run --config mydata_bench/configs/baseline_03_qwen_text_video.yaml
    conda run -n robo-dopamine python mydata_bench/run_qwen_eval.py score --run-dir results/mydata_bench/experiments/baseline_03_qwen_text_video

    conda run -n robo-dopamine python mydata_bench/run_qwen_eval.py run --config mydata_bench/configs/baseline_04_qwen_video_text.yaml
    conda run -n robo-dopamine python mydata_bench/run_qwen_eval.py score --run-dir results/mydata_bench/experiments/baseline_04_qwen_video_text

GRM forward 与 incremental：

    conda run -n robo-dopamine python mydata_bench/run_raw_eval.py run --config mydata_bench/configs/baseline_05_grm_forward.yaml
    conda run -n robo-dopamine python mydata_bench/run_raw_eval.py score --run-dir results/mydata_bench/experiments/baseline_05_grm_forward

    conda run -n robo-dopamine python mydata_bench/run_raw_eval.py run --config mydata_bench/configs/baseline_06_grm_incremental.yaml
    conda run -n robo-dopamine python mydata_bench/run_raw_eval.py score --run-dir results/mydata_bench/experiments/baseline_06_grm_incremental

## 4. Attention steering

RoboReward 使用 `attention_06_roboreward_last_frame.yaml` 和
`attention_07_roboreward_all_frames.yaml`。以下命令对两个配置分别执行：

    conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py prepare-ranking --config mydata_bench/configs/attention_06_roboreward_last_frame.yaml
    conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py validate-ranking --config mydata_bench/configs/attention_06_roboreward_last_frame.yaml
    conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py rank --config mydata_bench/configs/attention_06_roboreward_last_frame.yaml
    conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py prepare-cohort --config mydata_bench/configs/attention_06_roboreward_last_frame.yaml
    conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py steer --retry-failed --config mydata_bench/configs/attention_06_roboreward_last_frame.yaml
    conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py score --config mydata_bench/configs/attention_06_roboreward_last_frame.yaml

Qwen 使用 `attention_08_qwen_last_frame.yaml` 和
`attention_09_qwen_all_frames.yaml`，同样对两个配置分别执行：

    conda run -n robo-dopamine python mydata_bench/run_qwen_attention.py prepare-ranking --config mydata_bench/configs/attention_08_qwen_last_frame.yaml
    conda run -n robo-dopamine python mydata_bench/run_qwen_attention.py validate-ranking --config mydata_bench/configs/attention_08_qwen_last_frame.yaml
    conda run -n robo-dopamine python mydata_bench/run_qwen_attention.py rank --config mydata_bench/configs/attention_08_qwen_last_frame.yaml
    conda run -n robo-dopamine python mydata_bench/run_qwen_attention.py prepare-cohort --config mydata_bench/configs/attention_08_qwen_last_frame.yaml
    conda run -n robo-dopamine python mydata_bench/run_qwen_attention.py steer --retry-failed --config mydata_bench/configs/attention_08_qwen_last_frame.yaml
    conda run -n robo-dopamine python mydata_bench/run_qwen_attention.py score --config mydata_bench/configs/attention_08_qwen_last_frame.yaml

GRM forward 使用 attention 10、11，incremental 使用 attention 12、13：

- `attention_10_grm_forward_after.yaml`、`attention_11_grm_forward_before_after.yaml`
- `attention_12_grm_incremental_after.yaml`、`attention_13_grm_incremental_before_after.yaml`

对每个 GRM 配置运行：

    conda run -n robo-dopamine python mydata_bench/run_attention_eval.py prepare --config mydata_bench/configs/attention_10_grm_forward_after.yaml
    conda run -n robo-dopamine python mydata_bench/run_attention_eval.py rank --source in_domain --retry-failed --config mydata_bench/configs/attention_10_grm_forward_after.yaml
    conda run -n robo-dopamine python mydata_bench/run_attention_eval.py steer --retry-failed --config mydata_bench/configs/attention_10_grm_forward_after.yaml
    conda run -n robo-dopamine python mydata_bench/run_attention_eval.py metrics \
      --run-dir results/mydata_bench/experiments/attention_10_grm_forward_after \
      --config mydata_bench/configs/attention_10_grm_forward_after.yaml

把示例配置和 run directory 依次替换为 11、12、13。这里必须显式使用
`--source in_domain`，因为该分支读取 `ranking_data.jsonl` 的独立 grounding；
`all` 子命令仍是旧的 consensus-ranking 工作流，不适用于本实验计划。
三类模型都会运行 baseline、target、wrong-target、low-rank，并分别汇总 top-8/32/64。

## 5. negative scope 消融

当前实验矩阵使用：

- native video 的 last-frame 配置：`temporal_intervention_scope: last_frame`、`negative_scope: target_span`。
- native video 的 all-frames 配置：`temporal_intervention_scope: all_frames`、`negative_scope: all_visual`。
- GRM after/before-after 配置：对应 `intervention_location`，并使用 `negative_scope: target_span`。

三类模型当前配置均设置 `steering_query_scope: all`。`negative_scope` 还可选
`other_spans` 或 `none`；展开额外消融时必须使用新的配置与 output directory。
当前每个主实验已经同时运行 top-k 8、32、64，bias 固定为 6。

每条 steering 输出的 hook_diagnostics 至少检查：

- negative_scope
- selected_span_labels
- all_visual_span_labels
- selected_token_count
- negative_token_count
- selected_negative_disjoint
- video_metadata.target_source_frame_indices

## 6. native video temporal span 消融

RoboReward/Qwen native attention 支持两个互斥输入模式：

- `temporal_span_mode: native_pairs`：保持模型原生时间 tubelet。8 个采样帧
  形成 4 个 temporal spans，每个 span 含相邻两帧。
- `temporal_span_mode: duplicate_frames`：仍先按原生规则采样 8 帧，再在
  内存中构造成 `[f1,f1,...,f8,f8]` 的 16 帧输入。模型的
  `temporal_patch_size=2` 不变，最终得到 8 个单源帧 temporal spans。

`attention_video_max_frames` 表示复制前的源采样帧上限。重复模式不会生成
中间视频文件，原始帧号和时间戳会保存在 processor alignment 和 steering
diagnostics 中。由于 Qwen3-VL 使用总视频 pixel budget，代码会同步把该 budget
放大两倍，使方案 B 的单帧 spatial grid 与方案 A 相同，避免混入空间分辨率变化。

原生二帧 span 的 bbox 可用 `temporal_bbox_reduce` 控制：

- `last`：使用该 span 最后一帧的 tracking bbox；
- `union`：使用两帧 bbox 并集；
- `intersection`：使用交集，空交集会使该样本明确失败。

改变 temporal span 模式或 bbox reducer 后必须使用新的 output directory，
并重新执行 validate-ranking、rank 和 steer；head ranking 不能跨输入模式复用。

已提供两份互不覆盖的 last-frame 消融配置：

- `attention_16_roboreward_last_pair_union.yaml`：方案 A，8→4，最后二帧
  tracking bbox 取并集；
- `attention_17_roboreward_last_frame_duplicated.yaml`：方案 B，8→16→8，
  每个 temporal span 只含一个源帧。两份配置都使用 `union`；方案 B
  每组两帧相同，所以 union 仍等于该单帧 bbox。

对两份配置分别按第 4 节的顺序执行 prepare-ranking、validate-ranking、rank、
prepare-cohort、steer、score。validate-ranking 生成的
`processor_alignment_diagnostics.jsonl` 应分别显示 temporal_grid 4 和 8。

forward 的目标通常应是 after_cam_high；native video 应是最后一个 video_t 时间片且包含 terminal source frame。

## 7. 分片与恢复

GPU 任务可通过 YAML 中的 shard_id 和 num_shards 分片。JSONL 为 append-only；成功样本会跳过，失败样本需要显式加入 --retry-failed。不同模型、协议、scope、top-k 或 bias 必须使用独立 output directory。

## 8. 本轮最终状态与结果位置

`exp_plan.md` 的编号重复了 6，但实际矩阵是 6 个 baseline 加 8 个 attention，
共 14 项，均已完成。baseline 为 755/755、invalid=0；attention 使用同一批
336 条自动 grounding cohort：native 每项 336×10 条当前条件，GRM 每项 336×13，
全部为 `ok`。

grounding 最终状态：

- `results/mydata_bench/grounding/sam3/grounding.jsonl`：2495 个历史/重试行。
- 最新 1510 个端点：680 `ok`、696 `no_detection`、134 `invalid`。
- 首末端点双 `ok`：336；tracking terminal audit error：0。
- `results/mydata_bench/cohorts/auto_grounded/`：冻结的 336 条 ID 与无标签 episode manifest。
- `results/mydata_bench/ranking_grounding/sam3/`：30/30 样本、60/60 端点 `ok`。

每个实验的最终目录为：

    results/mydata_bench/experiments/baseline_01_roboreward_text_video
    results/mydata_bench/experiments/baseline_02_roboreward_video_text
    results/mydata_bench/experiments/baseline_03_qwen_text_video
    results/mydata_bench/experiments/baseline_04_qwen_video_text
    results/mydata_bench/experiments/baseline_05_grm_forward
    results/mydata_bench/experiments/baseline_06_grm_incremental
    results/mydata_bench/experiments/attention_06_roboreward_last_frame
    results/mydata_bench/experiments/attention_07_roboreward_all_frames
    results/mydata_bench/experiments/attention_08_qwen_last_frame
    results/mydata_bench/experiments/attention_09_qwen_all_frames
    results/mydata_bench/experiments/attention_10_grm_forward_after
    results/mydata_bench/experiments/attention_11_grm_forward_before_after
    results/mydata_bench/experiments/attention_12_grm_incremental_after
    results/mydata_bench/experiments/attention_13_grm_incremental_before_after

baseline 读取 `metrics.json`、`metrics.md` 和 `completion.json`；native attention
读取 `steering_metrics.json`，GRM attention 读取 `attention_metrics.json` 与
`attention_metrics.md`。完整效果量、置信区间和解释边界记录在 `experience.md`。

如推理已完成，只需重算指标，不会重新加载模型推理：

    conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py score --config mydata_bench/configs/attention_06_roboreward_last_frame.yaml
    conda run -n robo-dopamine python mydata_bench/run_qwen_attention.py score --config mydata_bench/configs/attention_08_qwen_last_frame.yaml
    conda run -n robo-dopamine python mydata_bench/run_attention_eval.py metrics \
      --run-dir results/mydata_bench/experiments/attention_10_grm_forward_after \
      --config mydata_bench/configs/attention_10_grm_forward_after.yaml

分别替换为 07、09、11、12、13 即可。恢复 steering 时 native 和 GRM 都应显式
使用 `--retry-failed`；append-only 文件中的旧指纹行会保留，评分只读取当前
grounding/ranking 指纹的最新条件。

## 9. 生成实验记录

所有实验推理和 metrics 完成后，在仓库根目录运行：

    python3 mydata_bench/write_exp_records.py

脚本会在每个已完成实验目录生成或覆盖 `exp_record.md`，内容包括 MAE、总体/suc/fail/
各 task 准确率、预测 label 分布，以及基于 `metadata.jsonl:source_suc_id` 的同视频
pairwise 区分度。GRM 同时报告项目原有四边界离散 MAE、连续 ordinal MAE；
`0.125/0.875` 和 `0.2/0.8` 两套端点阈值用于准确率，其中间值记为 `uncertain`。

`steering.jsonl` 是 append-only 文件，脚本与正式 metrics 保持一致：baseline 按
`example_id`、attention 按 `(example_id, condition)` 取最后一条记录。脚本还会把可用
MAE/accuracy 与已有 `metrics.json` 或 `steering_metrics.json` 交叉校验，不一致时拒绝
生成。GRM 的 `0.125/0.875` 和 `0.2/0.8` 两套端点阈值只用于准确率；预测分布固定
按 `[0,20%) / [20%,40%) / [40%,60%) / [60%,80%) / [80%,100%]` 分为 label 1–5。
默认路径也可用 `--experiments-root`、`--metadata`、`--ranking-metadata` 覆盖。

## 10. 最终验收

    python3 -m pytest tests/test_mydata_*.py mydata_bench/tests/test_mydata_bench.py -q
    python3 -m compileall -q mydata_bench
    git diff --check -- mydata_bench
    find mydata_bench -type f \( -name '*.orig' -o -name '*.rej' \) -print

统计结果只适用于未经人工审核的自动 grounding cohort，属于 exploratory；
不能写成 audited/formal causal result。当前 ranking 清单与 cohort 有 30/30 ID 重叠，
也必须在报告中披露。

## 11. 2026-08-10：v2 当前实验矩阵（最终）

> 本节是 `mydata_v2/new` 的当前口径；前文 `755/336/30` 与 `experiments/` 是 v1 历史记录。

v2 必须使用 `mydata_bench/configs/v2/`，不要把同名 v1 配置与结果混用。

- 完整评测集：1213 条（407 suc、806 fail）。
- attention 自动 grounding cohort：846 条（268 suc、578 fail），未经人工审核。
- ranking 清单：36 条；当前 grounding 后可用 34 条，代码自动适配可用子集，不要求 36/36。
- ranking 排除前 8 层，在 896 个 heads 中按 raw mass 排名；steering 使用 all-query、bias=6、top-8/32/64。

最终输出根目录：

    results/mydata_bench/experiments_v2/

已完成 `exp_plan.md` 的 16 项：

- baseline_01--06：每项 1213 条，全部 `status=ok`，均有 `metrics.json`。
- native attention_06--09、14--15：每项 846×10=8460 条，全部 `status=ok`，均有 `steering_metrics.json`。
- GRM attention_10--13：每项 846×13=10998 条，全部 `status=ok`，均有 `attention_metrics.json`。
- 每项均已有 `exp_record.md`，包括 MAE、总体/suc/fail/task 准确率、预测分布和 pairwise 区分度。

v2 实验记录使用显式路径重新生成：

    python3 mydata_bench/write_exp_records.py \
      --experiments-root results/mydata_bench/experiments_v2 \
      --metadata /home/dais/workspace/data/mydata_v2/new/metadata.jsonl --ranking-metadata /home/dais/workspace/data/mydata_v2/new/ranking_data.jsonl

    python3 mydata_bench/write_ranking_overlap.py

汇总文件：

- `results/mydata_bench/experiments_v2/*/exp_record.md`：16 份逐实验记录。
- `results/mydata_bench/experiments_v2/ranking_overlap.md` 与 `.json`：10 个 attention ranking 的 top-8 清单和 top-8/32/64 重合度。

仅重算 v2 指标时使用 v2 配置，不会重跑模型推理：

    conda run -n robo-dopamine python mydata_bench/run_roboreward_attention.py score --config mydata_bench/configs/v2/attention_15_roboreward_all_frames.yaml
    conda run -n robo-dopamine python mydata_bench/run_qwen_attention.py score --config mydata_bench/configs/v2/attention_09_qwen_all_frames.yaml
    conda run -n robo-dopamine python mydata_bench/run_attention_eval.py metrics \
      --run-dir results/mydata_bench/experiments_v2/attention_13_grm_incremental_before_after --config mydata_bench/configs/v2/attention_13_grm_incremental_before_after.yaml

最终验收：

    python3 -m pytest tests/test_mydata_*.py mydata_bench/tests/test_mydata_bench.py -q
    python3 -m py_compile mydata_bench/write_exp_records.py mydata_bench/write_ranking_overlap.py
    python3 -m compileall -q mydata_bench

v2 的 formal gate 仍为 `exploratory_unaudited_auto_grounding`；不得表述为已人工审核或正式因果结果。

## 12. 2026-08-10：重跑官方 GRM incremental

旧的 `baseline_06_grm_incremental`、`attention_12_grm_incremental_after` 和
`attention_13_grm_incremental_before_after` 只评估了 terminal 前约 20 帧的一个局部
hop，不能代表 `examples/inference.py` 定义的 incremental episode progress。旧目录保留
作 `local-hop` 历史结果；正式重跑只使用文件名带 `_official` 的配置和新输出目录。

raw baseline（完整序列 `0→20→...→terminal`，逐 hop 推理并按官方递推累计）：

    conda run -n robo-dopamine python mydata_bench/run_raw_eval.py run \
      --config mydata_bench/configs/v2/baseline_06_grm_incremental_official.yaml
    conda run -n robo-dopamine python mydata_bench/run_raw_eval.py score \
      --run-dir results/mydata_bench/experiments_v2/baseline_06_grm_incremental_official

两组 GRM attention 都必须重新 `prepare → rank → steer → metrics`。incremental ranking
现在对所有 hop 的 last-prompt raw/excess mass 取均值；steering 对每个 hop 使用该帧的
tracking bbox 重新映射 token，并让每个 condition 独立累计最终 progress。

    GRM_CFG=mydata_bench/configs/v2/attention_12_grm_incremental_after_official.yaml
    GRM_OUT=results/mydata_bench/experiments_v2/attention_12_grm_incremental_after_official
    conda run -n robo-dopamine python mydata_bench/run_attention_eval.py prepare --config "$GRM_CFG"
    conda run -n robo-dopamine python mydata_bench/run_attention_eval.py rank --source in_domain --config "$GRM_CFG"
    conda run -n robo-dopamine python mydata_bench/run_attention_eval.py steer --config "$GRM_CFG" --retry-failed
    conda run -n robo-dopamine python mydata_bench/run_attention_eval.py metrics --run-dir "$GRM_OUT" --config "$GRM_CFG"

将上面的 `attention_12...after_official` 同时替换成
`attention_13...before_after_official` 后，再完整执行一次。不要直接运行 `all`：当前 `all`
子命令固定走 consensus ranking，而这两份配置使用独立 ranking grounding 的
`in_domain_ranking.json`。

推理结果的 `incremental_steps` 保存每个 before/after index、原始 hop score 和未裁剪
累计值；标准 `progress` 是最终官方累计值裁剪到 `[0,1]` 后的统计字段。attention 结果
还会记录每 hop bbox 对应的 tracking frame index、token 数量，以及 `all_hops` steering
协议。全量完成后，再对 `experiments_v2` 运行第 11 节的 `write_exp_records.py` 命令。
