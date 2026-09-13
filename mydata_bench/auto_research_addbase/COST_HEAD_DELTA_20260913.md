# 第33轮方案：训练成本与单请求延迟补测

## 测量范围

- 日期：2026-09-13；当前 Robo-Dopamine 工作目录。
- 不读取评分标签、不训练、不选择新参数，不修改已有checkpoint或效果报告。
- 两个模型各占一张空闲 NVIDIA A100-SXM4-80GB：Qwen物理GPU0，RoboReward物理GPU1。GPU2未使用。
- 视频→文字，8帧，max_pixels=50176，BF16，batch1，k32；Qwen last_frame、RoboReward all_frames，沿用已冻结代表配置。
- 从validation按视频SHA排序等距取12个不同视频组，每组第一条输入；每条件重复3次，共36次计时。相同样本的baseline/target交替先后运行。
- 预热模型和PNG/ROI轨迹缓存；CUDA同步计时。下述结果是小规模局部性能测量，不是线上负载、尾延迟或闭环机器人测试。

## 已有训练耗时

来自两个阶段各模型final checkpoint的elapsed_seconds，包含固定优化循环；不包含模型加载、前期head profiling、grounding、方法搜索和完整评估。

| 模型 | 第32轮门控 | 第33轮矩阵 | 两阶段合计 |
| --- | ---: | ---: | ---: |
| Qwen | 940.985秒 / 15.68分钟 | 962.174秒 / 16.04分钟 | 1903.158秒 / 31.72分钟 |
| RoboReward | 899.887秒 / 15.00分钟 | 924.259秒 / 15.40分钟 | 1824.145秒 / 30.40分钟 |

每模型70条训练输入、28视频组；每阶段4200次前后向、525次更新。两模型合计约1.04 GPU小时的优化循环，双卡分别训练两模型时可并行；完整重训流程还需准备、加载与检查时间。使用现有冻结权重无需重新训练。上述成本不代表Robometer/SOLE的训练成本。

## 新测单请求耗时

单位毫秒，中位数。模型与读出包括研究实现中的诊断开销；请求时间另外包括预处理和传输。

| 模型 | 原模型：模型与读出 | 最终方案：模型与读出 | 原模型：请求 | 最终方案：请求 |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 69.18 | 79.48 | 101.16 | 111.74 |
| RoboReward | 69.78 | 80.30 | 102.11 | 112.76 |

相对原模型，模型与读出中位数增加约10.3/10.5毫秒，约14.9%/15.1%；含预处理的请求中位数增加约10.6毫秒，约10.5%/10.4%。这不是FLOPs或功耗测量，不外推至其他k、布局、分辨率、GPU或并发负载。

最终方案请求p95为Qwen113.04毫秒、RoboReward114.15毫秒；每条件仅12个不同视频、36次观测，不能当作生产环境p95保证。

实际每请求仍是一个完整模型前向。额外运算包括选中heads的QK重算、softmax、差分乘V与低秩矩阵乘法；参数少不等于这些运算免费。

新增28672个矩阵参数加固定1792个门控，FP32数值载荷共121856字节，约119KiB。完整8B主模型及激活仍占主要显存。本测试PyTorch峰值allocated约16.4GiB；该数值不含全部CUDA上下文/外部分配，baseline测量也已经加载适配参数。该输入长度下全程峰值几乎不变，不能据此声称任意长度的额外激活显存为零。

## 在线应用边界

固定权重可以用于请求式视频评分，无需每条视频再训练。输入需要当前任务文本及可得的ROI。当前补测从已提取PNG帧和预先生成的ROI轨迹开始，不包含视频采集/解码、目标检测/跟踪、网络、排队或服务调度。

现实现对每个片段完整前向，use_cache=False，没有实现流式视频状态复用。约0.11秒的本地暖请求不能直接推出完整系统可以稳定9Hz运行。若实时使用当前历史片段，还需验证只依赖截至当前时刻的信息；本实验对完整轨迹的终点评分效果不能证明中途奖励准确。

## 可复核材料

结果根目录为`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911/`。

- 脚本：[benchmark_head_delta_latency_20260913.py](benchmark_head_delta_latency_20260913.py)。
- [Qwen逐次计时与汇总](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911/audit/head_delta_latency_qwen_20260913.json)。
- [RoboReward逐次计时与汇总](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911/audit/head_delta_latency_roboreward_20260913.json)。
- 两模型所有重复前向的五类logits逐值一致；配对input_ids哈希一致；适配器与门控文件SHA在测量前后保持。
- 两个进程均正常退出（exit0）；结束后GPU0/1占用归零。
