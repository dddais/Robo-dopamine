# SOLE 剩余实验使用四张 GPU

仅保留 `sole_official` 和 `sole_image_text`。`text_image` 与 `interleaved` 已停止，其结果保留。

## 当前安排

- GPU 0、1、3：分别执行 `sole_official` 的不同剩余 SAS／对照条件。
- GPU 2：继续原来的 `sole_image_text`；结束并释放 GPU 后，自动加入 `sole_official` 队列。
- 单个样本内部仍为原来的七步递归；模型、提示词、greedy、生成上限、grounding、heads、bias 和 batch=1 均保持原设置。
- 复用完整 baseline、ranking 和已有预测。每个条件仅允许一个 worker 写入，已完成样本直接跳过。
- 输出继续追加到原来的 `sole_official/predictions` 和 `steps`，原始结果不覆盖，原来的汇总入口可直接使用。

新的入口位于 `mydata_bench/basic_method/parallel/`，没有修改冻结的推理源文件；并行程序自身另存代码指纹和执行记录。

## 查看运行

```bash
nvidia-smi
tail -n 20 results/mydata_bench/experiments_v2_basic_method/parallel_sole_official_20260914_v1/events.jsonl
tail -n 30 results/mydata_bench/experiments_v2_basic_method/scheduler/monitor_sole_parallel_20260914_v1.log
```

`events.jsonl` 中的 `condition_started` 记录 GPU worker 正在执行的条件，`condition_finished` 表示完整 1213 条预测记录（含 baseline 回退）已保存，不代表全部解析成功。`status` 同时列出待运行和运行中的条件。

## 手动恢复

当前后台监控会自动续跑并在两份配置完成后生成 SOLE 汇总和包含之前 15 份配置的 combined 双 head 汇总。后台监控仍运行时无需重复启动。

若整个并行任务已经退出，可从仓库根目录运行：

```bash
python -m mydata_bench.basic_method.parallel \
  --config mydata_bench/configs/v2_basic_method/sole_official.yaml \
  --execution-dir results/mydata_bench/experiments_v2_basic_method/parallel_sole_official_20260914_v1 \
  --gpus 0 1 2 3
```

该命令只续跑 official；完整后台流程使用 `results/mydata_bench/experiments_v2_basic_method/scheduler/monitor_sole_parallel_20260914_v1.sh`。运行中的配置持有实验锁，重复启动会被拒绝。

程序只在无计算进程、显存足够且连续两次利用率低于阈值时使用空闲卡；不会占用仍在运行 image_text 或外部任务的 GPU。一张 GPU 一个 worker，条件完成后领取下一个；最终少于四个剩余条件时，部分 GPU 提前空闲属于正常情况。

## 验证记录

CPU 检查覆盖条件锁、继承实验锁、部分结果续跑、baseline 回退、重复/损坏记录拒绝、完成标记与冻结文件校验。

换卡重放结果分别保存在：

- `results/mydata_bench/experiments_v2_basic_method/verification/parallel_sole_official_20260914_v1/verification.json`
- `results/mydata_bench/experiments_v2_basic_method/verification/parallel_sole_image_text_20260914_v1/verification.json`

重放逐字段比较已有预测（仅排除耗时和完成时间），并再次调用原入口验证已完成样本不会重复推理。少量样本一致性检查不等同于全量重复推理。
