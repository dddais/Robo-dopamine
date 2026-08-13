# 概率空间反事实门控（PSCG）最小可行方案

## Material Passport

- 状态：`DESIGNED`，尚未实现和运行新实验
- 适用模型：RoboReward-8B、Qwen3-VL-8B
- 目标：利用 target-region 与 matched wrong-region 的概率响应差异，过滤 attention steering 的非特异扰动，只在证据足够时采用 target-steered 结果
- 原则：门控只使用模型概率，不读取当前样本标签；标签只用于离线校准和评估

## 1. 方法原理

对同一个样本运行三种条件：

- `baseline`：不做 steering；
- `target`：在 instruction 对应物体区域做 steering；
- `wrong`：用相同 heads、bias 和区域大小，在错误区域做 steering。

如果 target 和 wrong 都让模型朝同一个答案变化，说明变化可能来自“施加 attention bias”这一通用扰动，而不一定来自 instruction 对应物体。反之，如果只有 target 明显增强某个答案，而 wrong 没有复现该增强，就说明这个变化具有空间特异性，更值得接受。

因此，wrong 的作用不是判断答案对错，而是估计并扣除 steering 的非特异影响。

## 2. 从离散输出改为候选概率

当前旧 gate 只比较最终输出的 1–5 标签，会丢失大量信息。例如 target 和 wrong 的概率只差 0.01，也可能因为 argmax 不同而被旧 gate 当成强证据。

最小改造是：不再让模型自由生成最多 128 个 token，而是在输出前缀 `ANSWER:` 后直接读取分数 `1,2,3,4,5` 的 logits，并在这五个合法答案内做 softmax：

```text
p_c(k) = softmax(logits_c[1:5])[k]

c ∈ {baseline, target, wrong}
k ∈ {1,2,3,4,5}
```

这样一次分支只需要 prompt prefill 和一个答案位置的 logits，不需要完整自回归生成。当前两个模型的数字 1–5 都是单 token，可以直接实现；启动时再检查一次 tokenizer 即可。

## 3. 特异增益分数

令 target 分支最可能的答案为：

```text
e = argmax_k p_target(k)
```

先把某个答案的概率转换为 log-odds：

```text
z_c(e) = log[p_c(e) / (1 - p_c(e))]
```

target 和 wrong 相对 baseline 的变化分别为：

```text
Δ_target = z_target(e) - z_baseline(e)
Δ_wrong  = z_wrong(e)  - z_baseline(e)
```

最终只使用一个核心分数：

```text
specific_gain = Δ_target - max(0, Δ_wrong)
```

含义很直接：

- target 增强答案 `e`，`Δ_target` 越大越好；
- wrong 也增强同一个答案，说明存在非特异扰动，因此扣除；
- wrong 朝反方向变化时不额外奖励 target，所以使用 `max(0, Δ_wrong)`，避免把一个异常 wrong 分支变成虚假的强证据。

例子：

```text
Δ_target=1.5, Δ_wrong=1.2  → specific_gain=0.3  # 大部分是共同扰动
Δ_target=1.5, Δ_wrong=0.1  → specific_gain=1.4  # target 特异性明显
Δ_target=1.5, Δ_wrong=-1.0 → specific_gain=1.5  # 不因 wrong 反向而额外加分
```

## 4. 最终 gate 规则

只保留三个条件：

```text
1. e ∈ {1,5}                       # 与当前 suc=5 / fail=1 数据契约一致
2. p_target(e) ≥ τ_conf            # target 自身有足够概率置信度
3. specific_gain ≥ τ_gain          # 扣除 wrong 干扰后仍有足够增益
```

全部满足时采用 target 预测，否则回退 baseline：

```python
def pscg(p_baseline, p_target, p_wrong, tau_conf, tau_gain):
    baseline = argmax(p_baseline)
    target = argmax(p_target)

    if target not in {1, 5}:
        return baseline

    target_gain = logodds(p_target[target]) - logodds(p_baseline[target])
    wrong_gain = logodds(p_wrong[target]) - logodds(p_baseline[target])
    specific_gain = target_gain - max(0.0, wrong_gain)

    if p_target[target] >= tau_conf and specific_gain >= tau_gain:
        return target
    return baseline
```

若 target 与 baseline 本来就输出相同标签，直接返回该标签即可，不影响结果。

这比旧 SC-EG 合理的地方在于：

- 不再把“输出端点”直接当成高置信，而是检查真实概率；
- 不再只判断 `target != wrong`，而是连续衡量二者差异大小；
- wrong 只扣除与 target 同方向的共同变化，正好对应“消除非特异干扰”的目标；
- 规则只有两个待校准阈值，容易实现和解释。

## 5. 可行性分析

### 5.1 原理上可行

target 与 wrong 使用同一个输入、同一组 heads、相同 bias 和相同 token 数，唯一主要区别是 steering 区域。因此，同一样本内的 target–wrong 差异可以抵消一部分模型本身的置信偏差和 attention bias 的通用影响。

该方法识别的是“模型响应是否依赖正确物体所在区域”，不是证明 target 一定等于真实标签。最终是否值得接受 target，由离线数据上的两个阈值校准。

### 5.2 现有结果支持继续尝试

已有实验中，target steering 明显优于 wrong-region control，旧离散 SC-EG 也能减少一部分 fixed steering 对 suc 样本的损害。这说明 target 与 wrong 的差异包含有效信息。新版只是把离散差异升级为连续概率差异，理论上会比 `target != wrong` 更稳定。

### 5.3 最低实现要求

wrong region 只需满足：

- 与 target 不重叠；
- token 数相同；
- 尽量位于同一帧或同一视觉 span；
- 使用完全相同的 heads、bias、query scope 和 negative scope。

若无法构造等大小 wrong region，该样本直接使用 baseline。无需增加更多控制分支。

## 6. 阈值校准

每个模型分别校准 `τ_conf` 和 `τ_gain`，因为 RoboReward 与 Qwen 的概率尺度不同。

在一个不参与 head ranking 的 calibration split 上进行小网格搜索：

```text
τ_conf ∈ {0.4, 0.5, 0.6, 0.7, 0.8}
τ_gain ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
```

选择规则：

1. suc accuracy 不低于 baseline；
2. MAE 低于 baseline；
3. 在满足前两项的组合中选择总体 accuracy 最高者。

按 `video_sha256` 划分 calibration/test，保证同一视频的 suc 与 fail instruction 不跨集合。阈值确定后冻结，再在 test split 上评估。在线推理不使用标签。

若现有数据暂时无法重新划分，可先在现有 cohort 上做探索性验证，但结果只能用于选参数，不能当作最终确认结果。

## 7. 在线实时部署优化

### 7.1 首要优化：用一次候选打分代替自由生成

当前每个条件调用一次 `model.generate(max_new_tokens=128)`。PSCG 只需要五个答案 token 的 logits，因此应新增 `score_candidates()`：

```text
完整生成：prompt prefill + 多步 autoregressive decode
候选打分：prompt prefill + 读取一个位置的 5 个 logits
```

这会直接去掉绝大多数 decode 延时，也是收益最大、实现最简单的优化。

### 7.2 推荐在线方案：三分支并行

延时优先时，baseline、target、wrong 同时运行，而不是串行运行。

有两种实现：

**方案 A：三张 GPU，空间换时间。** 每张 GPU 放一个模型副本，分别执行 baseline、target、wrong。三路同时开始，最终延时接近最慢的一路。

- 优点：实现最简单，对现有 hook 改动小；
- 代价：模型权重显存约为三倍；
- 适合：当前已有 GPU 0、1、2 的实验和低延时验证。

**方案 B：单 GPU batch=3。** 将相同输入复制成三个 batch row，为每行分别使用 zero/target/wrong attention bias，一次 batched forward 得到三组概率。

- 优点：只保存一份模型权重，GPU 并行度更高；
- 代价：需要让 attention hook 支持 per-row bias，activation 显存增加；
- 适合：单卡部署或高吞吐服务。

第一版建议先实现方案 A 验证 gate 效果，再实现方案 B。

### 7.3 复用计算

三个分支输入完全相同，可以复用：

1. 视频解码、抽帧、resize 和 processor 输出；
2. target/wrong token position 映射；
3. vision encoder 输出。

attention steering 发生在语言模型 self-attention 中，不改变视觉编码器。因此 vision embedding 可以只计算一次，再复制给三个语言分支。这能避免最明显的重复视觉计算。

进一步可选优化是：找到所选 steering heads 中最早的 layer `L`，语言模型的 `0...L-1` 层在三个分支中完全相同，可以只计算一次，到第 `L` 层再分成三路。该优化改动较大，不作为第一版必须项。

### 7.4 可选平均算力优化：baseline early exit

如果更关心平均算力而非最坏延时，可以先计算 baseline：

```text
若 baseline 为端点且 max probability ≥ τ_skip：直接返回 baseline
否则：并行计算 target + wrong，再执行 PSCG
```

`τ_skip` 在 calibration split 上选择，使 early-exit 子集的准确率足够高。它能减少平均分支数，但未 early-exit 样本会经历两阶段推理，因此 P95/P99 可能高于三分支直接并行。第一版实时部署可暂不启用。

## 8. 最小实验与验收

只需要比较四组：

1. baseline；
2. fixed target steering；
3. target-only probability gate：不使用 wrong 扣除；
4. 完整 PSCG。

第 3 与第 4 组的差异直接检验 wrong-region 是否真的消除了非特异干扰。

主要指标沿用现有定义：总体/suc/fail accuracy、MAE、pairwise suc−fail 差值。PSCG 至少应满足：

- MAE 低于 baseline；
- 总体和 fail accuracy 高于 baseline；
- suc accuracy 不低于 baseline；
- 完整 PSCG 优于 target-only gate，证明 wrong 分支有实际价值。

延时只需测三种实现：

1. 三次自由生成串行；
2. 三分支候选打分串行；
3. 三分支候选打分并行。

报告端到端 P50、P95、吞吐和峰值显存即可。

## 9. 推荐落地顺序

1. 新增 `score_candidates()`，验证它与严格 `ANSWER: <1-5>` 输出基本一致；
2. 用现有冻结 b/t/w 配置补跑五候选概率；
3. 在 calibration split 上确定两个阈值；
4. 对比 baseline、fixed target、target-only gate 和 PSCG；
5. 效果成立后，先用三 GPU 并行部署；
6. 再实现单 GPU batch=3 和 vision embedding 复用。

最终在线规则保持为：**target 概率足够高，且扣除 wrong 同方向变化后仍有明显增益，才采用 target；否则使用 baseline。**
