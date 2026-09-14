# Basic method 实验使用

使用 `mydata_bench.basic_method` 新入口，配置位于 `mydata_bench/configs/v2_basic_method/`，结果位于 `results/mydata_bench/experiments_v2_basic_method/`。旧实验入口和结果继续保留。

## 1. 准备数据

```bash
cd /home/dais/workspace/Robo-Dopamine
conda activate robo-dopamine
python -m mydata_bench.basic_method.prepare
```

当前已生成全量输入和 20 份配置。重复执行会校验已有文件；内容不一致时拒绝覆盖。无需重跑 SAM3。

**本轮运行范围：排除 SOLE 的全部配置和 `qwen_official` 原生视频配置。** 保留 GRM、RoboReward、Robometer 各四种输入，以及 Qwen 的 `text_image`、`image_text`、`interleaved`，共 **15 个 baseline、270 个 SAS／对照条件**。GRM 的两份等价配置暂时保留。以下 20／360 是配置库完整矩阵的规模。

- 评测：全部 **1213** 条；release 中双端点可用 **1116** 条。
- 独立 ranking：**36** 条，使用 `ranking_grounding_v2_release`，双端点可用 **33** 条。
- 5 个模型 × 4 种输入 = **20 个 baseline**。
- 每种输入有 2 个范围 × 3 个 top-k × 3 种干预 = 18 个条件，共 **360 个 SAS／对照条件**。
- 默认 bias 固定为 **+6 / −6**，top-k 为 **8、32、64**；三种干预是 `target`、`wrong_region`、`low_rank`。

四种输入配置名为 `official`、`text_image`、`image_text`、`interleaved`。模型名为 `grm`、`roboreward`、`qwen`、`meter`、`sole`。

`official` 表示官方输入结构，生成模型采用本实验统一的 **Transformers + greedy** 解码，并不表示完整复刻上游服务的采样设置。例如 SOLE 上游使用 vLLM、temperature=1、max_tokens=200，本实验使用 greedy、max_new_tokens=512；Robometer 使用专用进度头的 forward，没有文本生成。Qwen 的官方项是原生视频接口搭配本仓库的 RoboReward 评分问题。

## 2. 运行

可先执行只加载处理器、不加载模型权重的检查：

```bash
python -m mydata_bench.basic_method.run \
  --config mydata_bench/configs/v2_basic_method/grm_official.yaml \
  --phase preflight --limit 2 --ranking-limit 2 --output-suffix _check
```

先查看本轮配置、当前 GPU 显存／利用率／计算进程，命令不会启动推理或写入实验结果：

```bash
python -m mydata_bench.basic_method.schedule --gpus 0 1 2 3 \
  --jobs-per-gpu 2 \
  --models grm roboreward qwen meter --exclude-configs qwen_official --dry-run
```

运行本轮 15 份配置；GPU 暂时不可用时，调度器会等待：

```bash
python -m mydata_bench.basic_method.schedule --gpus 0 1 2 3 \
  --jobs-per-gpu 2 \
  --models grm roboreward qwen meter --exclude-configs qwen_official
```

`--exclude-configs` 按 YAML 文件名去掉扩展名匹配，不会删除配置或已有结果；名字不存在会报错。不传筛选参数仍运行完整 20 份配置。

调度规则：

- 默认每 10 秒检查可分配 GPU，连续两次满足条件才启动。该卡的第一份配置要求 **无计算进程、空闲显存至少 60 GiB、利用率不超过 10%**。可以用 `--min-free-memory-mib`、`--max-utilization`、`--poll-seconds`、`--idle-checks` 调整阈值；显存门槛不是模型峰值显存的保证。
- 默认每卡一份配置；本轮命令指定 **`--jobs-per-gpu 2`，每卡最多两份，四卡最多八份**。先分散到各卡，再填第二个名额。每份独立完成 baseline、ranking、SAS 和对照，结束后检测资源并补位。
- 同卡补第二份时，只允许存在本调度器的工作进程；利用率可以高于首次启动阈值。显存采用保守预算：默认给新旧进程各预留 24 GiB，再留 8 GiB 余量，要求**实际空闲显存至少 56 GiB**，覆盖模型尚未完成加载的情况。可用 `--job-memory-mib`、`--memory-headroom-mib` 调整；不满足条件就等待，因此两份是上限。GPU 查询失败或存在外部计算进程时也等待。
- SOLE 全部输入及 `qwen_official` 尚未验证同卡并发，因此仍独占一张卡。含 SOLE 时优先启动其官方配置，再启动其余 SOLE 配置。**一份配置仍在一张卡上串行执行**，不会把 SOLE 七步或 18 个条件拆给多张卡；也不会迁移运行中的任务或驱逐外部任务。GPU 按 `nvidia-smi` 的物理编号选择，子进程用对应 UUID 绑定。
- 调度日志在结果目录的 `scheduler/`，每分钟记录队列状态。每条已完成预测立即追加保存；代码、配置和输入不变时，再次执行同一命令可续跑，完整配置会校验后跳过。
- 某份配置报错时记录失败并继续其余配置；最后只要存在失败就返回非零退出码，**不会自动重试模型／数据错误，也不代表所有配置成功**。修复后再续跑。Ctrl-C／SIGTERM 会停止本调度器的工作进程并保留已写入结果。

建议在 `tmux` 会话中运行长实验。调度命令只完成预测和 ranking，统计报告还需执行下面的 `reporting` 命令，并传入相同筛选参数。

**并发收益实测：** 同卡两个进程相对一个进程，总吞吐在 GRM 官方输入提升约 31%、RoboReward 官方输入约 102%、Qwen `image_text` 约 51%、Robometer 官方输入约 12%。共 432 次预测，测量样本在两种并发下输出完全一致；双进程整卡观察到的占用峰值约 21.5–35.5 GiB。只测了四份代表配置和固定小样本，不能将整个矩阵耗时直接减半，也未验证三进程。详见 [同卡并发测量](results/mydata_bench/experiments_v2_basic_method/verification/gpu_concurrency_20260914/summary.md)。

**此前每卡一份的耗时估算（4 张 A100 80GB 持续可用）：** 15 份配置预留 **10–14 小时**；完整 20 份配置粗估 **4–7 天**。前者依据 150 次小样本端到端计时，四卡队列外推约 8 小时，再预留 ranking、验证及样本差异；后者主要受 SOLE 官方七步串行推理限制，仅该配置推理外推约 91 小时。两进程的整矩阵耗时尚未实测；上述规划值不含等待外部 GPU 的时间，长生成可能使耗时超出范围。此前计算依据见 [调度与耗时核查](results/mydata_bench/experiments_v2_basic_method/verification/scheduler_timing_20260913/estimate.md)。

只跑某个模型：

```bash
python -m mydata_bench.basic_method.schedule --gpus 0 --models grm
```

只跑一份配置或单个阶段：

```bash
CUDA_VISIBLE_DEVICES=0 python -m mydata_bench.basic_method.run \
  --config mydata_bench/configs/v2_basic_method/grm_official.yaml --phase all
```

`--phase` 支持 `baseline`、`rank`、`steer`、`all`。执行 `steer` 前须完成对应 baseline 和 ranking。小规模推理可加 `--limit 2 --ranking-limit 2 --output-suffix _pilot`；如果这些 ranking 样本不足以覆盖某个范围，会明确报错，应扩大 pilot 的 ranking 数量。

## 3. 汇总

```bash
python -m mydata_bench.basic_method.reporting --output-name analysis_both_heads_03_07_v1 \
  --models grm roboreward qwen meter --exclude-configs qwen_official
```

输出包括 `summary.md`、本轮四份 `<model>_summary.md`、`metrics.csv`、各配置的详细 JSON，以及 `head_overlap.json`。汇总必须使用与调度一致的筛选参数；`index.json` 记录所选配置，并仅按这 15 份配置判断完成情况。`complete` 表示预测记录齐全，`all_predictions_valid` 另检查各输出 head 是否全部有效；解析失败和不可用对照不会伪装成有效预测。详细结果包括全量／排除 ranking 视频组的 holdout 指标、各 task 指标、预测分布、pairwise 区分度和回退数量。MAE 使用原 1–5 分档；端点准确率同时报告 `0.125/0.875`、`0.2/0.8`、`0.3/0.7` 三套阈值。

新增 `0.3/0.7` 口径：输出 **≤0.3 判失败、≥0.7 判成功**，中间区间和无效输出不计为端点正确，准确率仍以全部应评测样本为分母。对归一化的离散 1–5 评分，这一宽松口径对应失败接受 1–2 分、成功接受 4–5 分；原严格指标保留。Markdown 和 CSV 新增总／suc／fail 三列，CSV 字段为 `accuracy_03_07`、`suc_accuracy_03_07`、`fail_accuracy_03_07`。JSON 中的 `accuracy["0.3/0.7"]` 同时覆盖 full、holdout、task、实际施加子集及共同有效对照子集。MAE、预测分布、pairwise 和原两套准确率均不改变。新报告见 [三套端点阈值汇总](results/mydata_bench/experiments_v2_basic_method/analysis_basic_endpoints_03_07_20260914_v1/summary.md)。

如需以某套阈值作为主列，可直接从已有完整分析导出独立版本，无需重跑推理或重新计算已有指标。例如导出 `0.2/0.8` 的总／suc／fail 主表：

```bash
python -m mydata_bench.basic_method.reporting.endpoint_view \
  --source results/mydata_bench/experiments_v2_basic_method/analysis_basic_endpoints_03_07_20260914_v1 \
  --output results/mydata_bench/experiments_v2_basic_method/analysis_basic_endpoints_02_08_custom \
  --threshold 0.2/0.8
```

`--output` 必须是新目录。CSV 的 `endpoint_threshold` 标注本版口径，`accuracy`、`suc_accuracy`、`fail_accuracy` 均使用该阈值；详细 JSON 保留来源分析的全部统计，`index.json` 记录来源及文件哈希。已生成的 [0.2/0.8 专用汇总](results/mydata_bench/experiments_v2_basic_method/analysis_basic_endpoints_02_08_20260914_v1/summary.md)包含本轮 15 份配置及 Robometer 两个 head。

比较阈值时，请使用同样格式的 [0.3/0.7 专用汇总](results/mydata_bench/experiments_v2_basic_method/analysis_basic_endpoints_03_07_20260914_v2/summary.md)和上面的 0.2/0.8 专版；两者主表及 CSV 的总／suc／fail 均使用标题标明的阈值。较早的 `analysis_basic_endpoints_03_07_20260914_v1` 是三套阈值并列报告，其无后缀的“总准确率”／`accuracy` 仍为 0.125/0.875，0.3/0.7 要读取单独命名的列。RoboReward 和 Qwen 的 0.3/0.7 为接受失败 1–2 分／成功 4–5 分的宽松辅助指标；原计划的正式准确率仍要求预测分数等于标签。

**Robometer 两个 head 均纳入报告。** `meter_summary.md` 分别列出 progress head 和 success head 的 baseline、最佳 SAS、全部 SAS／对照；总报告也包含两种读出，CSV 用 `head` 列区分。JSON 保留原 `conditions` 中的 progress 统计，新增 `secondary_success_head` 和 `secondary_success_head_matched_controls`。Success 读取原始预测的 `success_probability`，沿用同一套 1–5 分档和端点阈值，包含 holdout、task、分布、pairwise 和回退统计，无需重跑模型。

旧 `basic_method.score` 仅报告 progress 等原主要读出，漏掉了 Robometer success head。新入口放在独立的 `reporting` 子包，保留冻结的推理代码指纹，避免影响正在运行的 SOLE 和现有实验续跑。旧报告保留；请用新入口和新的输出名字重新汇总。已补齐的本轮报告见 [两种 head 的汇总](results/mydata_bench/experiments_v2_basic_method/analysis_basic_both_heads_20260914_v1/summary.md)。

运行中查看部分结果可加 `--allow-incomplete`。再次汇总请换新名字，例如 `analysis_v2`，避免覆盖已有分析。

## Grounding 规则

只使用精确源帧的框，禁止借用邻帧框。任一所需帧缺框，该条件下整条样本复用**同模型、同输入、同解码设置**的 baseline，正负 bias 均为 0；每个条件仍保留 1213 条记录。Baseline 的输入构造不读取 bbox。

| 输入／干预范围 | 当前静态检查可用数 |
|---|---:|
| GRM forward：最后正面帧／所有正面槽位 | 1116 / 1116 |
| 独立 8 帧图像：最后帧／全部帧 | 1116 / 531 |
| Robometer official：最后帧／全部帧 | 1116 / 531 |
| SOLE official：逐步最后帧／逐步全部帧 | 531 / 531 |

这些是框的可用数，实际对照还需通过 token 几何检查。RoboReward、Qwen 的 `official` 使用各自处理器原生 MP4 采样，帧数可能不同；程序根据实际采样索引检查覆盖，最后视频 token 需要其编码的两帧均有框。这里不能直接套用固定 8 帧的统计。

- GRM 只做 forward。`last_frame` 为 `after_cam_high`；`all_frames` 为 `reference_start`、`before_cam_high`、`after_cam_high`。腕部相机和空白 goal 不施加 bias。其官方输入本身就是交错格式，因此 `official` 与 `interleaved` 保留为两份等价配置。
- SOLE 官方模式先检查全部七步，再开始递归；中途缺框不会拼接 baseline 历史。`last_frame` 仅使用完整落在当前帧拼图区域内的 token，排除跨入前一帧的边界 token；若目标只占据跨界 token，整条回退并记录 `no_isolated_target_tokens`。全量几何检查后仍为 531 条评测／29 条 ranking 可用。Ranking 对对应范围的有效样本，在未干预的七步历史上平均 attention mass。
- Bias 施加在语言模型指定 head 的 attention logits 上、softmax 之前；目标 key 为 +6，同一干预范围内其余视觉 key 为 −6，文本 key 为 0，并保留因果掩码。生成模型同时覆盖 prefill 和后续解码；Robometer 只做 forward。零 bias 会直接旁路干预，避免显式零掩码改变数值计算路径。
- `wrong_region` 是同范围、等 token 数且与目标不重叠的区域，**不代表另一个真实物体**。无法构造时记录 `control_unavailable`；不会作为 grounding 回退。统计另给 SAS／对照共同有效子集。
- 预测保存 `sas_applied`、`baseline_fallback`、`fallback_reason`、`grounding_check` 和 baseline 来源。缺文件、模型错误和解析失败不会转换成成功的 baseline 回退；baseline 自身解析失败也会原样保留。
- Ranking 按模型、输入和范围分别生成，缺框样本不作为零 attention 加入平均；视频相同但指令不同的样本按各自 ID 读取 grounding。

## 4. GRM / SAS 主视角 attention 可视化

新增入口 `python -m mydata_bench.basic_method.visualization`，使用对应 GRM 配置的冻结输入、精确帧 grounding、独立 ranking 和原有 SAS controller。支持 `grm_official`、`grm_text_image`、`grm_image_text`、`grm_interleaved` 四份配置；默认 `grm_official`、`last_frame`、top-8、`after_cam_high`。需要该配置已经生成 `run_identity.json` 和对应范围的 ranking。

先查看样本 ID，不加载模型或处理器：

```bash
python -m mydata_bench.basic_method.visualization \
  --list-samples --split fail --limit 10
```

生成原图、GRM baseline、GRM + SAS、SAS − baseline 四列对照：

```bash
CUDA_VISIBLE_DEVICES=0 python -m mydata_bench.basic_method.visualization \
  --config mydata_bench/configs/v2_basic_method/grm_official.yaml \
  --scope last_frame --top-k 8 \
  --example-id suc/ljx_lfz_task_1_1/1 \
  --example-id fail/ljx_lfz_task_1_1/1 \
  --output-dir results/mydata_bench/experiments_v2_basic_method/visualizations/grm_last_top8_v1
```

`--output-dir` 必须是新目录。默认画所选 top-k head 的平均图，以及其中前两个 head 的单独图。每条样本会重新运行 baseline 和 SAS 的 greedy generation，以捕获 attention 并显示本次输出的 progress；已有预测只用于核对，不会被覆盖。有缓存时检查输入指纹，并在输出分数不一致时于终端和网页标注。缺少所需精确帧框时只运行 baseline，再复用其 attention 和分数，明确标为 fallback，实际 bias 为 0。

可选参数：

| 参数 | 用途 |
|---|---|
| `--focus-images reference_start before_cam_high after_cam_high` | 同时画三个主视角输入槽位；默认只画 `after_cam_high` |
| `--scope all_frames --top-k 32` | 使用 all_frames 的独立 ranking，按原实验干预三个主视角槽位；top-k 须存在于该配置 |
| `--heads L19H23,L20H4` | 指定要观察的 head，层号和 head 号均从 0 开始；SAS 仍干预 ranking 的 top-k |
| `--per-head 8` | 除平均图外，画前 8 个观察 head；`0` 只画平均图 |
| `--subset task1_1 --split fail --holdout-only` | 按任务、成功／失败、holdout 筛选 |
| `--limit 8` | 筛选后取前 8 条；默认前 4 条，显式提供 ID 时默认保留全部指定 ID；`0` 为全部匹配样本 |
| `--normalization image_fraction` | 每幅图内 attention 归一化为总和 1，便于比较空间分布；默认 `raw` 保留原始概率 |
| `--preflight` | 只加载处理器、检查框和 token 几何，不使用 GPU；也需要新的输出目录 |

Query 与本轮独立 ranking 一致，为 **prefill 的最后一个 prompt token**。参考 `scan_localization_heads_best.py` 的 `vector_to_grid` 还原图像 token 网格；在原实验 SDPA 调用前，以 float32 Q/K 和已经叠加 SAS bias 的实际 mask 重建所选行的 softmax。模型本身仍使用原来的 SDPA 推理路径。图中展示的是该 query 的诊断概率，不是所有文本 query 的平均，也不是生成 score 数字 token 的 attention；SAS 本身仍按配置覆盖所有 query 和后续解码。

Baseline/SAS 每对图使用**相同 head、相同色标**，不分别做 min-max 拉伸。默认 raw 色标保留 attention mass 的变化；不同样本、槽位或单 head 图可有不同上限。绿色框是精确源帧目标框，差值图红色为增加、蓝色为减少。图下注明未归一化的 image mass 和 target/image 比例；target 使用实验相同的 bbox-to-token 规则。

输出文件：

- `index.html`：可直接打开的图片浏览页。
- 每条样本目录中的 `after_cam_high_mean.png`、`after_cam_high_L*H*.png` 等对照图。
- `attention.npz`：`heads` 为行顺序；`baseline_rows` / `sas_rows` 为各 head 的完整 prompt-key 概率行；`baseline_<slot>` / `sas_<slot>` 为 `[观察head数, 网格高, 网格宽]` 的原始 attention；有框时另存 `target_<slot>` 布尔 token mask。
- `metadata.json`：精确帧与 bbox、观察 head、每个 head 的 image/target mass、实际 bias、回退信息、本次模型输出和已有预测核对结果。
- `manifest.json` / `summary.json`：配置、ranking 哈希、干预 head、代码来源、样本清单与完成状态。

可视化代码放在独立的 `basic_method/visualization/` 子包，不改变冻结的推理代码指纹，原实验可继续续跑。可视化也不会重新生成 ranking 或调用 SAM3。

已生成 [两条样本的可视化示例](results/mydata_bench/experiments_v2_basic_method/visualizations/grm_sas_last_frame_top8_demo_20260914/index.html)（一条 suc、一条 fail；official / last_frame / top-8）。这两条样本重新生成的 baseline/SAS 输出均与已有实验记录一致。针对捕获概率、推理透传、因果 mask、统一色标和回退的测试可运行：

```bash
python -m unittest discover -s tests -p test_basic_method_visualization.py -v
```

## 文件与续跑

`inputs.json` / `ranking_inputs.json` 冻结实际输入及 release track 路径；`grounding_coverage.json` 是静态覆盖检查；每份配置的 `predictions/` 保存全量有效条件输出，`independent_ranking/` 保存独立排名。

不要只复制 cohort，也不要删除 release JSONL 引用的 `grounding_v2_*`／`ranking_grounding_v2_*` 目录，或新 `inputs.json` 中 `image_paths` 引用的旧图像缓存。切分支不会自动删除这些 Git 忽略的结果文件，但新分支必须有本次新增的 `basic_method` 代码和配置。

代码、模型处理器、输入或配置发生变化时会拒绝混用旧缓存。需要新实验时使用新目录重新准备：

```bash
python -m mydata_bench.basic_method.prepare \
  --output-dir results/mydata_bench/experiments_v2_basic_method_v2 \
  --config-dir mydata_bench/configs/v2_basic_method_v2
python -m mydata_bench.basic_method.schedule \
  --root results/mydata_bench/experiments_v2_basic_method_v2 --gpus 0 1 2 3 \
  --models grm roboreward qwen meter --exclude-configs qwen_official
```

汇总时同样传入该 `--root`，并保持相同的 `--models` 和 `--exclude-configs`。默认配置的 bias 为固定值；若要做 bias 扫描，请在新配置中明确设置，并使用独立输出目录。

协议与 bias 的详细复查见 [核查报告](results/mydata_bench/experiments_v2_basic_method/verification/protocol_bias_audit_summary.md)。可用下列命令重新执行小样本审计，`--output` 必须是新目录：

```bash
CUDA_VISIBLE_DEVICES=0 python -m mydata_bench.basic_method.audit_protocols \
  --output results/mydata_bench/experiments_v2_basic_method/verification/my_protocol_audit
```

该审计逐张量比较 Robometer／SOLE 的官方输入，并拦截五模型实际 SDPA 调用检查 head、正负 bias、文本 key 和因果掩码；不是全量效果评测。本次代码更新会使之前的小规模验证缓存失效，保留它们并使用新的验证后缀即可。正式 20 份配置尚无全量推理结果，无需重新准备 grounding。
