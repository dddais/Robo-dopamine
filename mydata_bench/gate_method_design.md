# gate最小可行方案

## Material Passport

- 状态：`DESIGNED`，尚未实现和运行新实验
- 适用模型：RoboReward-8B、Qwen3-VL-8B
- 目标：利用 target-region 与 matched wrong-region 和baseline的概率响应差异，过滤 attention steering 的非特异扰动，只在证据足够时采用 target-steered 结果
- 原则：不允许根据数据集本身的特定设置一些类似作弊的hard coding，比如端点判断等

## 1. 方法原理

对同一个样本运行三种条件：

- `baseline`：不做 steering；
- `target`：在 instruction 对应物体区域做 steering；
- `wrong`：用相同 heads、bias 和区域大小，在错误区域做 steering。

考虑这三者内部/结果可能存在的关系，怎么合理利用使得最终的模型效果表现最好