# Codex 实验任务：Counterfactual Binding Heads 小规模验证
# 基本原则（必须遵守）

- 尽量不修改现有代码库，如果需要修改，进行增量式修改，比如增加可选配置项等；
- 不允许进行git 操作本地已有的仓库，只能git clone开源仓库进行参考；
- 不允许对本地数据，结果等进行删除修改等操作，只能新增；
- 不用担心耗时，进行充分的调研、思考、理论分析，提出有道理的优雅的方案，严禁作弊的方法

### 实验基础设置

增量式修改：代码修改不要影响到之前的实验运行，尽量以增量式的形式增加代码，比如加可选参数配置之类的

数据集 ：/home/dais/workspace/data/mydata_v2/new ;/home/dais/workspace/Robo-Dopamine/results/mydata_bench/cohorts/auto_grounded_v2 (认为这就是正确的，不需要人工审核)

config 放在：/home/dais/workspace/Robo-Dopamine/mydata_bench/configs/v2_cbh

输出 在：/home/dais/workspace/Robo-Dopamine/results/mydata_bench/experiments_v2_cbh

conda环境：sam3:rewardbench-sam3 ；其它实验：robo-dopamine

可用GPU：0，1，2

vpn : proxy_on

## 1. 本轮目标

在已完成的 debug 基础上，验证以下假设：

> 相比按目标区域 attention mass 选择的 gaze heads，按“正确指令—反事实指令”的 reward 因果差异选择 Counterfactual Binding Heads（CBH），能更稳定地降低错误对象轨迹的 reward，同时保留成功轨迹分数和对象理解能力。

本轮只完成 **CBH head discovery、局部干预和 held-out pilot**。暂不实现完整 Instruction × Focus 矩阵和最终 binding gate。

---

## 2. 已知结论

实现时直接基于以下事实继续，不再重复大规模排错：

- attention hook、zero-bias、prefill/decode 调用和视觉 token span 已验证有效；
- Qwen3-VL / RoboReward 的因果对照必须统一使用 `eager` backend；
- 当前低分辨率视觉网格过粗，需要提高分辨率后重新发现 heads；
- instruction token 位于图像前，不能直接关注后置视觉 token；
- Qwen 的全 decode 强干预会把实际对象改写成指令对象；
- RoboReward 已能识别实际对象，但 reward 没有充分使用该信息；
- raw gaze ranking 只能找到“看目标”的 head，不能保证其控制 reward binding。

---

## 3. 统一实验读出

复用仓库现有推理、processor、attention runtime、候选评分和 object probe，实现以下统一设置：

1. Qwen3-VL 和 RoboReward 的 baseline、head discovery、intervention 全部显式使用 `eager`。
2. baseline 与 intervention 使用相同的 prompt、媒体输入、processor 参数和 cache mode。
3. 主 reward 读出使用 cached-autoregressive 的完整候选序列 log-prob，保存：
   - `p(r=1...5)`
   - `E[r]`
   - high-vs-low margin
   - 最终生成整数
4. 保留 zero-bias 检查，结果必须与 baseline 完全一致。
5. 每条记录同时保存：
   - instructed object
   - manipulated object probe
   - match 判断
   - 目标区域 attention mass
   - reward 变化

定义：

\[
m(I,\tau)=
\operatorname{LSE}(z_4,z_5)
-
\operatorname{LSE}(z_1,z_2)
\]

其中 \(z_r\) 是候选分数 \(r\) 的完整序列 log-prob。

---

## 4. 数据划分

按独立 source video 划分，不能让同一视频出现在 discovery 和 evaluation 中。

建议使用 40 个独立视频：

- **Discovery split：20 个成功视频**
- **Evaluation split：20 个 held-out 视频及其成功/错误对象配对轨迹**

尽量覆盖不同任务、对象类别、颜色和场景。

Discovery split 只使用成功轨迹。对每条成功轨迹构造：

- `I_clean`：与实际成功对象一致的原指令；
- `I_swap`：将目标替换为同场景中的另一个候选对象，视频保持不变。

优先只替换一个属性或对象词，避免同时改变动作和目标位置。

---

## 5. 提高视觉区域精度

将视觉输入提高到 `max_pixels=200704`，然后重新执行所有 head discovery，不复用低分辨率 ranking。

区域映射改为面积加权：

\[
w_k=
\frac{\operatorname{Area}(B\cap C_k)}
{\operatorname{Area}(C_k)}
\]

其中 \(B\) 是逐帧 bbox 或 mask，\(C_k\) 是视觉 token cell。

实际 bias 为：

\[
b_k=\delta w_k
\]

视频使用逐帧 object tube。默认只对目标区域施加正 bias，不压制：

- rival object
- gripper
- contact region
- destination
- 其他动作证据

保留原 target-vs-all 方式，仅作为对照条件。

---

## 6. Query 位置

在所有图像之后、输出分数之前加入一条固定的 post-visual anchor。所有 baseline 和 intervention 使用完全相同的 anchor，例如：

```text
Check whether the manipulated object matches the instructed object before scoring.
ANSWER:
```

主要干预位置：

1. post-visual anchor 对应的 token；
2. 第一个 score token 的 decode query。

不要默认干预全部生成 token。全 decode 只在少量 sanity 样本中作为对照。

---

## 7. 发现 Counterfactual Binding Heads

两个 checkpoint 分别发现 heads，不能共用 ranking。

### 7.1 反事实 reward 差异

对成功视频计算：

\[
\Delta_{\mathrm{cf}}
=
m(I_{\mathrm{clean}},\tau)
-
m(I_{\mathrm{swap}},\tau)
\]

### 7.2 Head 粗筛

在 attention head 输出进入 `o_proj` 之前，为每个 query head 增加一个只用于分析的标量 gate \(g_h\)，初始值为 1。

只在 post-visual anchor 和 first-score query 上应用 gate，计算：

\[
S_h=
\frac{\partial \Delta_{\mathrm{cf}}}{\partial g_h}
\]

在 discovery split 上平均 signed score，保留对 clean-vs-swap 区分有正贡献的前 64 个 heads。

这里不更新参数，只读取梯度。

### 7.3 精确单 head 验证

对粗筛后的 heads 逐个做 head-output ablation：

\[
C_h=
\Delta_{\mathrm{cf}}
-
\Delta_{\mathrm{cf}}^{(-h)}
\]

按 discovery split 上的平均 \(C_h\) 排序，选择 top-8 作为 CBH。

同时保存每个 head 的：

- layer / head index
- 粗筛 score
- 精确 ablation score
- 跨视频正向比例

---

## 8. Pilot 实验

### 8.1 小剂量 sanity

先在 4 个不属于 discovery 的视频上测试：

- bias：2、4、6、8
- query：anchor + first-score
- head：CBH top-8
- 区域：area-weighted positive-only

选择满足以下条件的最低剂量：

- 目标区域 attention 有明显提升但不接近饱和；
- Qwen object probe 不出现目标吸附式退化；
- reward 候选分布发生可测变化。

### 8.2 Held-out evaluation

在 20 个 held-out 视频上固定剂量和 top-8，比较：

1. eager baseline
2. generic gaze heads top-8
3. CBH top-8
4. layer-matched random heads top-8
5. CBH + wrong-region
6. CBH + target-vs-all

所有 head set、剂量和 query scope 在正式评测前冻结。

---

## 9. 主要指标

按独立视频聚合，分别报告 success 和 wrong-object failure。

\[
\Delta_{\mathrm{pair}}
=
\mathbb{E}[E[r]_{\mathrm{success}}]
-
\mathbb{E}[E[r]_{\mathrm{failure}}]
\]

\[
C_{\mathrm{reward}}
=
\Delta_{\mathrm{pair}}^{\mathrm{intervention}}
-
\Delta_{\mathrm{pair}}^{\mathrm{baseline}}
\]

同时报告：

\[
\Delta_s=
\mathbb{E}[E[r]^{\mathrm{int}}_{\mathrm{success}}
-
E[r]^{\mathrm{base}}_{\mathrm{success}}]
\]

\[
\Delta_f=
\mathbb{E}[E[r]^{\mathrm{int}}_{\mathrm{failure}}
-
E[r]^{\mathrm{base}}_{\mathrm{failure}}]
\]

还需要报告：

- mean score shift
- high-vs-low margin
- object probe accuracy
- manipulated-object 文本保真率
- target attention mass
- CBH 相对 gaze / random / wrong-region 的差值
- 每个任务和对象类别的分层结果

理想结果是：

- `C_reward > 0`
- `Δ_s` 接近 0
- `Δ_f < 0`
- CBH 明显优于 matched random 和 wrong-region
- object probe 不退化
- 改善不依赖整体统一降分

---

## 10. 结果判断

### 可以进入完整方法

满足以下现象时，下一轮实现 Instruction × Focus 矩阵和 binding gate：

- CBH 在 held-out 数据上优于 generic gaze 和 matched random；
- 成功轨迹 reward 基本保持；
- 错误对象轨迹 reward 稳定下降；
- Qwen object probe 不再出现目标吸附；
- wrong-region 无法得到相近收益。

### Qwen 的备选路径

若 attention bias 仍导致 manipulated object 被改写为 instructed object：

- 保留同一 CBH ranking；
- 停止扩大 spatial bias；
- 改为在 anchor / first-score query 上做 head-output patching 或 residual intervention；
- 用 object probe 作为必要验收项。

### RoboReward 的备选路径

若 object probe 始终正确，但 CBH 对 reward 仍无明显作用：

- 说明对象信息存在但 reward readout 未使用；
- 下一轮直接实现显式 counterfactual binding score；
- 再用 binding gate 与 base progress 组合。

---

## 11. Codex 最终产出

在现有仓库结构中增量实现，不要求新建固定目录。最终提交：

1. 可复现的 discovery 与 evaluation 配置；
2. 两个 checkpoint 各自的 CBH top-8 head list；
3. 每条样本的 raw reward、probe、attention 和条件信息；
4. 汇总表：baseline / gaze / CBH / random / wrong-region；
5. 一份 `auto_cbh_pilot.md`，说明：
   - 实际 backend、分辨率、query scope 和剂量；
   - discovery/evaluation 视频划分；
   - CBH ranking 方法与 head 列表；
   - success、failure、pair margin 和 object probe 结果；
   - 是否进入 Instruction × Focus 与 binding gate 阶段。
