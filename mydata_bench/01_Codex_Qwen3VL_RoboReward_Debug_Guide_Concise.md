# Qwen3-VL / RoboReward 注意力干预排查指南

# 基本原则（必须遵守）

- 尽量不修改现有代码库，如果需要修改，进行增量式修改，比如增加可选配置项等；
- 不允许进行git 操作本地已有的仓库，只能git clone开源仓库进行参考；
- 不允许对本地数据，结果等进行删除修改等操作，只能新增；
- 不用担心耗时，进行充分的调研、思考、理论分析，提出有道理的优雅的方案，严禁作弊的方法

### 实验基础设置

增量式修改：代码修改不要影响到之前的实验运行，尽量以增量式的形式增加代码，比如加可选参数配置之类的

数据集 ：/home/dais/workspace/data/mydata_v2/new ;/home/dais/workspace/Robo-Dopamine/results/mydata_bench/cohorts/auto_grounded_v2 (认为这就是正确的，不需要人工审核)

config 放在：/home/dais/workspace/Robo-Dopamine/mydata_bench/configs/v2_debug

输出 在：/home/dais/workspace/Robo-Dopamine/results/mydata_bench/experiments_v2_debug

conda环境：sam3:rewardbench-sam3 ；其它实验：robo-dopamine

可用GPU：0，1，2

vpn : proxy_on


## 目标

排查 gaze-head 在 Qwen3-VL 和 RoboReward 上效果弱，究竟是代码问题还是模型机制差异。

请先阅读现有仓库，复用已有模型加载、预处理、hook、评分和评测代码；不要按本文创建固定目录或重构工程。开始修改前，先简要说明实际调用链和准备修改的入口。

---

## 1. 确认真实推理路径

梳理一次评测实际经过的流程：

```text
样本 → 图像/视频预处理 → prompt/token
→ model forward/generate → attention 干预
→ raw logits → reward/value → 指标
```

确认以下信息：

- 实际 attention backend；
- hook 挂载的模块是否真的被调用；
- 干预覆盖 prefill、首个评分 token、decode 中的哪些阶段；
- 图像、视频和多视角 token 的排列；
- 最终分数来自 raw logits 还是生成文本解析。

---

## 2. 验证干预确实生效

### 零干预测试

将 bias 设为 0，结果应与 baseline 一致。

### 强干预测试

选一个样本、一个 head 和一个明确区域，施加较强正 bias。计算目标区域注意力质量：

\[
A(R)=\sum_{k\in R}A(q,k).
\]

强干预后，\(A(R)\) 应明显上升。若 attention 改变但 reward 不变，说明 hook 已生效，问题位于对象理解或 reward 读出。

### 推理阶段对照

分别测试：

1. baseline；
2. prefill-only；
3. decode-only；
4. prefill + decode。

短答案 reward 模型应重点检查 prefill 和第一个评分 token。

---

## 3. 验证视觉区域到 token 的映射

不要直接把单图 image-token 范围逻辑用于视频。使用 processor 输出的媒体 token、`grid_thw` 或仓库中的等价元数据，确认：

- image token 与 video token 没有混用；
- 多图、多视频和多视角的边界正确；
- 文本、时间标记和分隔符未被当成视觉 token；
- resize、padding、patch、spatial merge、temporal patch 均被计入；
- 视频帧顺序和视角顺序正确。

为若干样本生成 overlay：把最终选中的 token 投影回原始帧，检查是否覆盖目标物体。

物体移动时使用逐帧 bbox、mask 或 object tube，不要把初始 bbox 复制到整段视频。

---

## 4. 验证 head 与 query 的选择

### Head

确认 layer/head 编号属于当前 checkpoint，并检查 query-head 与 KV-head 的映射。RoboReward 微调后不要默认沿用基础 Qwen3-VL 的 head 排名。

加入数量和层分布匹配的 random heads 作为对照。

### Query

至少比较：

- 最后一个 prompt token；
- 指令属性词；
- 指令物体类别词；
- reward anchor；
- 第一个评分 token。

只干预最后一个 prompt token 可能适合图像描述，但未必适合 reward 判断。

---

## 5. 验证 reward 读取方式

不要只看最终生成的整数。直接读取候选答案的完整序列 log-prob。

RoboReward 1-5 分至少记录：

\[
p(r)=\operatorname{softmax}(\ell_1,\ldots,\ell_5),
\]

\[
E[r]=\sum_{r=1}^{5}r\,p(r),
\]

\[
m=\operatorname{LSE}(\ell_4,\ell_5)-\operatorname{LSE}(\ell_1,\ell_2).
\]

同时检查：

- 多 token 候选是否被完整评分；
- chat template 是否改变评分 token 位置；
- GRM 多次 forward 是否都受到干预；
- 多分支融合是否抵消单分支变化。

---

## 6. 做三层诊断

对同一批成功/错误对象配对样本，同时记录：

1. **Attention**：目标区域 attention mass 是否提高；
2. **Object probe**：模型能否识别指令目标、实际抓取对象及二者是否一致；
3. **Reward**：成功与错误对象轨迹的 margin 是否拉开。

定义：

\[
C_{attn}=A^{int}(q,R)-A^{base}(q,R),
\]

\[
C_{reward}=
[m_{succ}-m_{cf}]_{int}
-
[m_{succ}-m_{cf}]_{base}.
\]

| 结果 | 主要结论 |
|---|---|
| Attention 不变 | backend、hook、mask 或索引有误 |
| Attention 变，object probe 不变 | query、区域或 heads 选择不对 |
| Object probe 变，reward 不变 | reward readout 未使用该绑定信息 |
| 三者都改善 | 当前干预链路有效 |

---

## 7. 检查 target-vs-all 的副作用

增加一个 evidence-preserving 对照：

- 增强 instructed target；
- 不强压制 rival object；
- 保留夹爪、接触区、对象运动和 destination。

若该版本优于“压制所有非目标 token”，说明原方法删除了识别错误操作对象所需的证据。

---

## 8. 最小实验矩阵

对每个模型运行：

- baseline / zero-bias / strong-bias；
- random heads / gaze heads；
- prefill-only / decode-only / both；
- target-vs-all / evidence-preserving。

每个条件保存 attention mass、object probe、raw reward margin、期望分数、离散分数和成功样本性能。

---

## Codex 输出

最终提交：

1. 实际调用链与修改位置；
2. 发现的 bug、修复和对应测试；
3. Qwen3-VL、RoboReward、GRM 的三层诊断表；
4. 对效果弱原因的结论：实现错误、区域/评分错误，或 reward-causal leverage 较弱。

完成标准：能明确回答“干预是否生效、区域是否对齐、对象理解是否改变、reward 是否使用了这一变化”。
