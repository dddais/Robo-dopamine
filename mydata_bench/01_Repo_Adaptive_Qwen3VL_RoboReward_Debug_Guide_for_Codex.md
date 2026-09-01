# Qwen3-VL / RoboReward Gaze-Head 方案错误排查指南

> 适用对象：在**已有机器人 value/reward/progress 研究仓库**中工作的 Codex。  
> 目标：判断 Qwen3-VL 与 RoboReward 上效果较弱，究竟来自实现错误、输入与 token 映射错误、评分错误，还是来自真实的模型机制差异。  
> 使用方式：先阅读并理解现有仓库，再把本指南中的“能力、观测量与验收条件”映射到仓库已有实现。  
> 重要约束：本指南**不要求任何固定目录、文件名、类名、函数名、配置系统或命令行形式**。

---

## 基本原则（必须遵守）

- 尽量不修改现有代码库，如果需要修改，进行增量式修改，比如增加可选配置项等；
- 不允许进行git 操作本地已有的仓库，只能git clone开源仓库进行参考；
- 不允许对本地数据，结果等进行删除修改等操作，只能新增；
- 不用担心耗时，进行充分的调研、思考、理论分析，提出有道理的优雅的方案，严禁作弊的方法

### 实验基础设置

增量式修改：代码修改不要影响到之前的实验运行，尽量以增量式的形式增加代码，比如加可选参数配置之类的

数据集 ：/home/dais/workspace/data/mydata_v2/new ;/home/dais/workspace/Robo-Dopamine/results/mydata_bench/cohorts/auto_grounded_v2 (认为这就是正确的，不需要人工审核)

config 放在：/home/dais/workspace/Robo-Dopamine/mydata_bench/configs/v2_crossmodel


输出 在：/home/dais/workspace/Robo-Dopamine/results/mydata_bench/experiments_v2_corssmodel/auto_research

conda环境：sam3:rewardbench-sam3 ；其它实验：robo-dopamine

可用GPU：0，1，2

vpn : proxy_on


## 0. 给 Codex 的总指令

### 0.1 先适配仓库，不先设计新框架

开始修改前，必须先定位并复用仓库中的现有能力：

- 模型加载与 checkpoint 配置；
- processor、chat template、图像/视频预处理；
- dataset、sample/episode 表示与 dataloader；
- Qwen3-VL、RoboReward、GRM 的推理入口；
- attention hook 或 gaze-head 干预实现；
- reward/value/progress 的评分与解析；
- 日志、实验配置、缓存、测试和指标实现。

除非现有抽象无法承载诊断需求，否则不要平行复制一套 model wrapper、dataset、scorer、CLI 或实验框架。优先做局部扩展、可关闭的 instrumentation、现有接口上的可选参数，以及与仓库风格一致的测试。

### 0.2 修改前先提交“集成映射”

Codex 在编码前应先给出一张仓库集成表。它不要求保存到某个固定位置，但至少应包含：


| 逻辑能力      | 仓库中现有入口/符号 | 当前行为                 | 计划插入的诊断或修复        | 是否改变基线行为 |
| --------- | ---------- | -------------------- | ----------------- | -------- |
| 模型加载      | 实际位置       | backend、dtype、device | 记录真实 backend      | 否        |
| 多模态预处理    | 实际位置       | resize、抽帧、token 化    | 暴露 token metadata | 否        |
| gaze 干预   | 实际位置       | hook 层级与 phase       | 增加 trace/sentinel | 默认否      |
| reward 评分 | 实际位置       | 文本解析或 logits         | 增加候选序列评分          | 默认否      |
| 评测        | 实际位置       | 指标与聚合                | 记录三层诊断量           | 否        |


如果仓库存在多个推理路径，例如 Hugging Face、vLLM、自定义 runner，必须明确本次实验实际经过哪一条路径。不能只检查“看起来相关”的实现。

### 0.3 最小改动与可逆性

所有诊断和修复应满足：

1. 默认关闭时，与修改前基线等价；
2. 可在单个实验配置中显式开启；
3. hook 在每次调用后可靠卸载，不跨样本污染；
4. 不修改模型权重；
5. 不改变数据划分、帧采样或 prompt 来制造效果；
6. 发现 bug 时，先留下能复现 bug 的测试，再修复；
7. 任何新缓存都必须带模型、processor、输入和配置指纹，避免错误复用。



### 0.4 本文档的完成标准

最终必须能够回答：

1. 干预是否进入模型**真实执行的 attention 计算路径**？
2. 干预是否覆盖了正确的 prefill、首个评分 token 和 decode 阶段？
3. 图像、视频、多视角媒体 token 是否定位正确？
4. bbox/mask/object tube 是否正确映射到 resize、patch、merge 后的视觉 token？
5. layer/head 编号以及 query heads/KV heads 是否正确？
6. reward 分数是否使用正确的候选 token 序列和 raw logits，而非不稳定文本解析？
7. 干预是否改变了注意力，却没有改变对象判断？
8. 干预是否改变了对象判断，却没有改变 reward readout？
9. 原始 target-vs-all 抑制是否删除了错误对象、夹爪或接触证据？
10. 只有上述实现问题被排除后，是否仍可得出“Qwen3-VL/RoboReward 的 reward-causal leverage 较弱”的结论？

---



## 1. 诊断原则：把问题拆成三层

不要只看最终 accuracy。对同一批样本同时记录以下三层结果。

### 1.1 Attention controllability

干预是否真的把指定 head、指定 query 的注意力质量移向预期 key 区域：


C_{attn}=A^{int}(q,R)-A^{base}(q,R).


### 1.2 Object understanding / binding probe

注意力改变后，模型是否更准确地识别：

- 指令要求的对象；
- 实际被夹取、抬起或放置的对象；
- 两者是否一致。

该层应使用受限候选 logits 或稳定的 forced-choice probe，不只依赖自由文本描述。

### 1.3 Reward-causal leverage

对象理解改变后，reward/value/progress margin 是否发生预期变化：

## 
C_{reward}=
[m_{succ}-m_{cf}]_{int}

[m_{succ}-m_{cf}]_{base}.


这三层的组合决定根因：


| Attention 改变 | 对象判断改变 | Reward 改变 | 主要解释                             |
| ------------ | ------ | --------- | -------------------------------- |
| 否            | 否      | 否         | hook/backend/mask/head 索引错误      |
| 是            | 否      | 否         | query/key 选错、区域映射错或 head 不承载目标语义 |
| 是            | 是      | 否         | gaze 可控，但 reward readout 不使用该信息  |
| 是            | 是      | 是         | 干预链路有效，继续比较方法设计                  |
| 是            | 否或退化   | 异常波动      | bias 过强、上下文被删除或 causal mask 被破坏  |


---



## 2. 冻结可复现基线



### 2.1 记录运行指纹

沿用仓库现有日志系统，至少记录：

- Git commit 与未提交修改状态；
- Python、PyTorch、Transformers、CUDA、attention backend；
- 模型 checkpoint/revision、processor/tokenizer revision；
- dtype、量化、device map；
- 模型层数、query head 数、KV head 数、head dimension；
- image/video 特殊 token ID；
- processor 的 resize、patch、spatial merge、temporal patch 参数；
- 视频抽帧规则、帧索引、fps/timestamp；
- prompt/chat template 的最终文本或 token IDs；
- 输入媒体与处理后 tensor 的哈希；
- 干预 head 集、query/key 集、bias 模式、强度和 phase；
- 随机种子与确定性设置。

不要只记录用户配置中的 backend；必须记录模型实际初始化后、真实运行时采用的 backend。

### 2.2 建立最小诊断集

从开发数据中固定 8–16 个样本，不使用最终测试集调试。至少包含：

- 单图、多图、视频和多视角输入；
- 同类别不同颜色或属性的多个候选物体；
- 小目标；
- 目标移动；
- 抓取期间遮挡；
- 目标与 distractor 距离很近；
- 成功轨迹与已知目标物体错误轨迹；
- 一个 GRM 上已知干预明显有效的正对照。

每个诊断样本必须固定媒体、帧序、视角、指令、候选对象、目标区域和评分方式。

### 2.3 基线需要保存的结果

对每个模型和样本保存：

- raw generation；
- 候选评分标签的 token IDs；
- 每个候选标签的 sequence log-prob；
- 归一化 progress/value；
- reward margin；
- entropy；
- 解析后的离散结果；
- 如适用，GRM 每个比较分支及融合后的结果。

后续任何干预条件都必须使用完全相同的准备后输入，除非实验明确只改变某一个因素。

---



## 3. 仓库与真实调用链审计



### 3.1 查找内容而不是假设结构

使用仓库搜索、调用栈、断点或运行 trace 查明：

- 真正被评测调用的 model loader；
- processor/chat template 的真实入口；
- attention 模块实际类型；
- hook 注册的位置和时机；
- `forward`、`generate` 或外部推理引擎的调用关系；
- score 生成、文本解析和指标聚合路径；
- GRM 是否对一个 episode 做多次 forward；
- 多视角/多状态输入在 token 序列中的排列方式。

可搜索的关键词包括但不限于：模型名、`from_pretrained`、`attn_implementation`、`register_forward_*hook`、`attention_mask`、`image_token_id`、`video_token_id`、`grid_thw`、`generate`、`logits`、`ANSWER`、`reward`、`progress`、`frame_indices`、`spatial_merge`、`temporal_patch`。

### 3.2 输出真实调用链

用仓库中的实际符号描述：

```text
实验入口
  → dataset/episode 读取
  → 帧采样与多视角组合
  → processor/chat template
  → hook 注册
  → model.forward / model.generate / 推理引擎
  → raw logits 或文本
  → score adapter/parser
  → 多分支聚合
  → 最终指标
```

必须标出 gaze 干预究竟插在这条链的哪一步，以及是否对所有实际 forward 都生效。

---



## 4. P0：证明 hook 进入真实 attention 路径



### 4.1 检查实际 attention backend

重点排查：

- 配置声称 eager，但运行时切换成 SDPA/FlashAttention；
- hook 挂在 Python 模块上，但真实计算由融合 kernel 或另一实现完成；
- 推理走 vLLM/其他引擎，PyTorch hook 根本不在执行路径；
- hook 修改的是普通 `attention_mask`，模型内部随后又重建或覆盖 mask；
- 返回 attention 权重的 debug 路径与正常评分路径不同。

如果当前生产评测路径不能访问逐 head logits，允许建立一个**仅用于正确性审计**的 eager 等价路径；但必须验证输入、权重和评分逻辑与正式路径一致。不要未经验证就把 eager 结果当作生产结果。

### 4.2 Hook trace

对少量样本记录每次 hook 调用：

- 模型、layer、模块类型；
- 当前序列长度与 past-key-value 长度；
- prefill/decode 阶段；
- batch size；
- attention mask 原 shape、dtype、范围；
- 修改后 shape、dtype、范围；
- 被选中的 head/query/key 数量；
- 干预强度；
- 调用次数；
- hook 注册和卸载状态。

硬性要求：一次基线运行不得残留 hook；重复运行不得出现调用次数递增。

### 4.3 No-op equivalence

实现一种“经过同一干预代码路径但数值为零”的条件。它与完全不注册干预相比应满足：

- 输入哈希一致；
- logits 在允许的数值误差内一致；
- score/margin 一致；
- generation 在确定性配置下相同；
- hook 调用后完全卸载。

如果 no-op 都改变结果，先修复生命周期、复制、dtype、mask 合并或缓存污染问题。

### 4.4 强扰动 sentinel

选择：

- 单个明确 layer/head；
- 单个或很小的 query 集；
- 单个可验证视觉 token 区域；
- 极强但有限的测试 bias。

同时保存该位置干预前后的 attention distribution。sentinel 只用于证明链路，不作为正式方法参数。

至少计算：


\Delta A_{target}=A^{int}(q,R)-A^{base}(q,R),



L_1=\sum_k|A^{int}(q,k)-A^{base}(q,k)|,


以及 attention entropy、NaN/Inf 和 causal-invalid mass。

通过条件：

- 指定 head/query 的 attention 明显变化；
- 未选 head 在数值误差内基本不变；
- causal mask 禁止的位置仍为零；
- 没有 NaN/Inf；
- 反向 bias 能产生相反方向变化；
- 取消 hook 后恢复基线。

若 sentinel 不通过，不允许继续解释模型机制。

---



## 5. P0：区分 prefill、首个评分 token 与 decode



### 5.1 必做条件

在同一输入和同一 head/region 上运行：

1. 无干预；
2. 仅 prefill；
3. 仅首个评分 token 对应的 causal step；
4. 仅后续 decode；
5. prefill + 首个评分 token；
6. prefill + 全部 decode。

对于只输出一个短标签或一个数字的 reward 模型，“仅 decode”很可能几乎无效，因为决定性表征在 prefill 或第一个 score token 已形成。

### 5.2 使用 teacher-forced scoring 做主诊断

不要仅靠自由生成。对每个候选 reward 标签执行完整序列的 teacher-forced log-prob 评分，确保：

- 能精确定位每个评分 token 的 query step；
- 不受 early token 不同导致的后续分支差异干扰；
- 能观察 argmax 未改变时的概率变化。



### 5.3 阶段标注

phase 不能仅通过“序列长度是否为 1”粗略判断。结合实际 `past_key_values`、输入长度和 score prefix 明确标注：

- initial prefill；
- candidate label 第一步；
- candidate label 后续步骤；
- free-generation decode。



### 5.4 解释规则

- prefill 有效、decode 无效：原实现可能漏掉 prefill；
- 首个评分 token 有效：需要把干预覆盖到 teacher-forced score step；
- attention 有效但 reward 不变：继续做对象 probe 与 reward readout 诊断；
- 所有阶段均无 attention 变化：回到 backend/hook/mask 排查。

---



## 6. P0：媒体 token span 审计



### 6.1 禁止的通用假设

不要把“第一个视觉 token 到最后一个视觉 token”当作单一连续媒体区间，特别是在以下情况：

- 多张图；
- 多视角；
- video token；
- 媒体之间插入 timestamp、separator 或文本；
- 不同媒体具有不同 grid；
- processor 使用 placeholder 展开后再 merge。



### 6.2 从 processor 结果建立真实映射

优先使用 processor/model 已提供的 metadata，例如：

- image/video placeholder 的位置；
- image/video grid；
- media 顺序；
- frame/view 分组；
- merge 前后 token 数；
- 特殊 token 类型或 multimodal token type IDs。

对每个媒体块，至少得到：

- 媒体类型；
- 对应 view/frame 范围；
- LM 序列中的确切 key positions；
- 二维或三维 token grid；
- 与原始像素/帧的变换参数。



### 6.3 必须打印并人工检查一个完整样本

输出 token 序列的分段摘要：

```text
text span
media placeholder / media block 1
separator or timestamp
media block 2
...
score prefix
```

同时验证每个媒体块的 token 数与 processor grid 推导一致。

### 6.4 视频专项

检查：

- 使用的是 image token 还是 video token；
- temporal patch 是否把多个原始帧合并为一个时间 token；
- 抽帧后索引与 annotation 帧索引是否同一坐标系；
- frame reorder、padding、重复末帧是否发生；
- object tube 映射时是否使用处理后的时间网格，而非原视频总帧数。



### 6.5 多视角专项

必须保留 view 到媒体 token block 的一一对应。不能把 front、wrist 等视角拼成一个平面后仍使用单视角坐标。

---



## 7. P0：bbox/mask/object tube 到 token 的映射



### 7.1 明确全部坐标系

至少区分：

1. 原始图像/视频像素坐标；
2. processor resize/crop/pad 后坐标；
3. vision patch grid；
4. spatial merge 后视觉 token grid；
5. temporal patch 后视频 token grid；
6. LM 序列中的绝对 key position。

映射函数必须显式接收或能恢复所有变换参数，不能只按原图宽高做线性缩放。

### 7.2 区域离散化原则

对 bbox 或 mask 映射时：

- 优先使用 patch 与 mask 的覆盖比例；
- 记录 `any-overlap`、中心点落入、覆盖率阈值等策略；
- 小对象至少保证非空 token 集；
- 映射为空时不得静默回退到全图；
- 区域跨越 padding/crop 边界时必须截断；
- 多媒体块之间不允许索引串位。



### 7.3 视频必须使用 object tube

移动对象不能把初始 bbox 复制到所有帧。object tube 应表示：


R_j=R_j(t,v),


其中包含每个采样时间和视角的 bbox/mask、可见度及置信度。

首轮机制诊断优先使用 GT tube；自动 tracker 只作为后续实用性实验。

### 7.4 合成正确性测试

构造无需模型语义的合成输入：

- 固定色块位于图像四角与中心；
- 色块跨 patch 边界；
- 色块小于一个 merge 后 token；
- 视频中色块随时间移动；
- 两个视角色块位于不同坐标；
- resize/crop/pad 参数不同。

验证映射 token 投影回处理后媒体时，与色块位置一致。合成测试应先于真实样本。

### 7.5 可视化验收

对每个诊断样本生成或显示：

- processor 实际输入帧；
- GT bbox/mask；
- 被选 token 的栅格 overlay；
- target、rival、gripper/contact、destination 使用不同标签；
- 每帧/每视角被选 token 数；
- 原始到处理后坐标的变换摘要。

至少人工检查：首帧、接触前、夹爪闭合、抬起、搬运、终态。

---



## 8. P1：layer/head 编号与 checkpoint 一致性



### 8.1 Query heads 与 KV heads

Qwen 系列可能采用 grouped-query attention。干预 attention output 时通常应按 `num_attention_heads` 切分，而不是按 `num_key_value_heads`。

必须从当前 checkpoint config 和运行 tensor shape 验证：


hiddensize=numqueryheads\times headdim.


若修改的是 Q/K/V 或缓存，还需按模型内部 KV 复制逻辑核对。

### 8.2 Head ID 映射

无论仓库使用 `(layer, head)` 还是 global ID，都必须保证：

- 映射可逆；
- 不越界；
- 不重复；
- top-K 排序没有 0/1-based 偏移；
- checkpoint 的层数/head 数与 head list 一致；
- 保存和加载后含义不变。



### 8.3 Head identity test

逐次只干预一个 head，验证：

- 只有该 head 的 attention/head output 改变；
- 相邻 head 不变；
- layer 位置正确；
- 使用另一个 head ID 时变化位置随之移动。



### 8.4 不默认跨 checkpoint 共享 heads

必须分别报告：

- 原始 generic gaze heads；
- 在当前 checkpoint 上重新发现的 gaze heads；
- 当前 reward checkpoint 上的 reward-causal heads（后续 CBH 文档）。

如果 RoboReward 直接复用 base Qwen 的 head list，应把这点作为独立实验变量，而不是默认正确。

---



## 9. P1：reward/value/progress 评分正确性



### 9.1 使用完整候选序列 log-prob

对 RoboReward 的 1–5 分或 Qwen3-VL 的离散 rubric，不能假设每个标签只对应一个 token。应对实际 tokenizer 产生的完整候选序列计算：


\ell(r)=\sum_s\log p(y_s^{(r)}\mid x,y_{<s}^{(r)}).


然后：


p(r)=\operatorname{softmax}(\ell(1),\ldots,\ell(5)),



E[r]=\sum_r rp(r),
\qquad
v=\frac{E[r]-1}{4},



m=\operatorname{LSE}(\ell_4,\ell_5)-\operatorname{LSE}(\ell_1,\ell_2).


需要记录 tokenization，防止 `"1"`、`" 1"`、换行后 `1` 等前缀不一致。

### 9.2 Greedy parser 只作为兼容输出

同时保留：

- raw candidate log-probs；
- expected score；
- margin；
- entropy；
- greedy/free generation；
- parser 结果。

如果概率明显变化但 argmax 不变，应判定为“离散输出掩盖”，不是“干预无效”。

### 9.3 Qwen3-VL 通用评分

若采用 yes/no、success/failure 或其他 rubric，同样以候选序列 log-prob 为主。需要定义一个方向明确的 margin，例如：


m=\ell_{success}-\ell_{failure}.


### 9.4 GRM

保持仓库现有 incremental、forward-anchored、backward-anchored 等比较与融合逻辑不变，并额外记录每次 forward 的原始候选分布和 hop/progress。确认 hook 在每个实际 forward 都被注册，而不是只影响其中一个分支。

### 9.5 评分实现测试

至少验证：

- 候选序列批量评分与逐个评分一致；
- 关闭干预时与原评分在容差内一致；
- score prefix 相同；
- padding 不进入 label log-prob；
- candidate 长度差异处理正确；
- cache 与非 cache scoring 一致；
- batch size 变化不改变单样本结果。

---



## 10. P1：输入与 prompt parity

对 baseline 与 intervention 条件逐字段比较：

- 原始媒体路径/字节哈希；
- 抽帧索引、顺序、timestamp；
- resize/crop/pad 后 tensor；
- chat template；
- input IDs 与 attention mask；
- media grid；
- dtype/device；
- score prefix 与候选标签；
- generation/scoring 参数；
- cache 使用方式。

除干预参数外，任何差异都必须解释。尤其排查：

- 为了拿 attention 而走了不同 processor；
- 干预条件重跑随机抽帧；
- baseline 使用 vLLM，干预使用 HF，但 prompt/tokenization 不同；
- 多视角顺序在两个路径中不同；
- hook 版本无意中关闭了 DeepStack、cache 或某些视觉输入。

---



## 11. P1：干预 mask 与数值稳定性



### 11.1 必测干预条件

在固定少量样本上比较：

- 无干预；
- no-op；
- target boost only；
- target boost + all-nontarget suppress；
- target boost + background-only suppress；
- 保留 rival/contact/destination 的 evidence-preserving 条件；
- 随机 matched heads；
- shuffled region；
- 相反方向 bias；
- 多个有限强度。



### 11.2 检查 mask 合并

保证：

- 原 causal mask/padding mask 永远保留；
- intervention bias 只加在合法位置；
- dtype 与 attention logits 兼容；
- 不因半精度极值产生 NaN；
- query 维没有意外广播到所有 token；
- batch 中每个样本使用各自区域，不互相串位；
- GQA 情况下 head 维广播符合预期。



### 11.3 记录数值

记录：

- target/rival/contact/background attention mass；
- entropy；
- 最大/最小 attention logit 或 bias；
- output hidden-state norm；
- score margin；
- 格式失败率；
- NaN/Inf；
- 达到目标 attention mass 所需强度。



### 11.4 解释 target-vs-all 的风险

在目标物体错误中，rival object、夹爪接触和对象运动是失败证据。若 target-vs-all 抑制让目标识别改善却让错误检测不变或更差，应归因于方法设计删除证据，而不是简单归因于 Qwen3-VL/RoboReward 不可干预。

---



## 12. 对象与绑定 probe



### 12.1 Probe 设计

在不改变媒体的前提下，对同一轨迹提出受限问题，读取候选 logits：

- “指令要求操作哪个对象？”
- “夹爪实际接触/抬起了哪个对象？”
- “实际对象与指令对象是否相同？”

候选集合来自场景中真实对象。不要用开放式 caption 作为唯一判断。

### 12.2 Probe 运行条件

比较：

- baseline；
- generic gaze heads；
- checkpoint-specific gaze heads；
- random matched heads；
- target-vs-all；
- evidence-preserving。



### 12.3 诊断结论

- 目标对象问答改善，但实际操纵对象问答退化：非目标证据被压制；
- 两类问答都改善，reward 不变：reward readout 与对象表征解耦；
- 两类问答都不变，attention 明显变化：选中的 heads 只改变空间 gaze，不承载所需语义；
- probe 与 reward 都改善：当前方案实现和机制均有效。

---



## 13. 模型专项检查



### 13.1 Qwen3-VL

重点检查：

- 本地模型 revision 是否与 head 发现时一致；
- eager 路径是否真的返回逐 head attention；
- image/video media span 是否正确区分；
- DeepStack 或多层视觉注入是否在 debug 路径中保留；
- query 位置是否是 reward anchor，而非只用原论文中的 final prompt token；
- generic gaze head ranking 是否适用于当前 prompt 与任务；
- attention 改变后对象 forced-choice 是否变化。



### 13.2 RoboReward

重点检查：

- video token ID 与 image token ID 没有混用；
- object tube 对齐 temporal patch 后的时间网格；
- score 只输出短标签时，prefill 和首个 score token 是否覆盖；
- 1–5 的完整 candidate sequence 评分正确；
- reward 微调后的 checkpoint 是否重新发现 heads；
- 全视频被大量无信息帧稀释时，contact/lift event window 是否更有效；
- 静态 bbox 是否导致抓取后区域错位。



### 13.3 Robo-Dopamine GRM

作为正对照重点确认：

- 同一干预实现是否真的应用在 GRM 的每个比较 forward；
- 各分支单独的 margin 如何变化；
- GRM 的输入是静态状态、多图还是其他排列；
- GRM 上的强效果是否来自正确 token 映射，而另外两个模型映射错误；
- 不要因为 GRM 有效就跳过它的 no-op、sentinel 与评分审计。

---



## 14. 必须加入现有测试体系的正确性测试

不规定测试文件位置或框架；应使用仓库已有测试惯例。至少覆盖：

### 14.1 Hook 与计算路径

- no-op 等价；
- 注册/卸载无泄漏；
- 强 sentinel 改变指定 head；
- 未选 head 基本不变；
- prefill/decode phase 判定正确；
- cache 与非 cache 的 phase 行为一致。



### 14.2 Media span 与区域映射

- 单图；
- 多图；
- video；
- 多视角；
- temporal patch；
- spatial merge；
- crop/pad；
- 移动物体 tube；
- 小对象非空映射；
- 空区域显式报错或标记，不静默回退。



### 14.3 Head 索引

- layer/head/global ID 可逆；
- query/KV head 不混淆；
- 单 head identity；
- checkpoint shape 不匹配时立即失败。



### 14.4 Scoring

- 完整候选序列 log-prob；
- padding mask；
- batch/逐个一致；
- parser 与 raw logits 都保存；
- GRM 多分支覆盖；
- score 方向和归一化单调。



### 14.5 输入 parity 与复现

- baseline/intervention 除干预外完全一致；
- 同 seed 重复结果一致；
- 缓存 key 包含完整指纹；
- 两次连续实验不相互污染。

---



## 15. 实验运行与结果记录契约

不要求特定 CLI 或产物路径，但每次运行必须可追溯，并能导出以下语义信息。

### 15.1 每次运行记录

- run ID；
- 模型与 processor 指纹；
- Git 状态；
- 样本 ID 与输入哈希；
- prompt/token IDs 哈希；
- attention backend；
- 干预 phase；
- layer/head/query/key；
- 区域来源与映射统计；
- attention 前后指标；
- object probe 前后指标；
- reward logits/margin 前后指标；
- 最终预测；
- 异常与格式失败。



### 15.2 可视化记录

至少保留：

- token overlay；
- target/rival/contact 的 attention mass 对比；
- prefill/decode 条件图；
- raw reward distribution；
- 典型成功、典型改善、典型退化与不变样本。



### 15.3 Bug 记录

每个确认的 bug 应包含：

- 现象；
- 最小复现；
- 根因；
- 修复位置（使用仓库实际符号）；
- 修复前失败测试；
- 修复后通过证据；
- 是否会影响过去结果；
- 是否需要重跑历史实验。

---



## 16. 推荐执行顺序与停止条件

按以下顺序执行，前一阶段失败时不要跳到后一阶段调参。

### Gate A：基线冻结

通过条件：输入、环境、raw logits、score 和样本均可复现。

### Gate B：真实计算路径

通过条件：no-op 等价；sentinel 明确改变指定 attention；hook 无泄漏。

### Gate C：phase 正确

通过条件：prefill/score/decode 可独立控制并记录；短标签 scoring 覆盖决定性 step。

### Gate D：媒体与区域映射

通过条件：合成测试通过；真实 overlay 人工检查正确；video tube 与视角无串位。

### Gate E：head 与 score 正确

通过条件：head identity 通过；完整候选序列评分通过；GRM 多分支均被覆盖。

### Gate F：三层诊断完成

通过条件：每个模型都有 attention、object probe、reward margin 三层结论。

只有 Gate A–F 全部通过，才进入 Counterfactual Binding Heads 方法实验。

---



## 17. 根因分类与最终报告模板

最终结论至少归入以下一类，可多选：

- **A. 干预没有进入真实 attention 计算路径**
- **B. prefill / 首个 score token / decode 阶段覆盖错误**
- **C. image/video/multiview media span 错误**
- **D. bbox/mask/object tube 到 token 的映射错误**
- **E. layer/head/query/KV 索引错误**
- **F. reward/value/progress 评分或解析错误**
- **G. 输入、prompt、抽帧或推理路径不等价**
- **H. bias 数值不稳定或破坏 causal/padding mask**
- **I. target-vs-all 删除了 rival/contact 等失败证据**
- **J. attention 和对象理解可控，但 reward readout 不敏感**
- **K. checkpoint-specific head 功能迁移**
- **L. 仍未定位；列明未排除假设和缺失证据**

建议最终报告按以下逻辑组织，不要求固定文件名：

```markdown
# Gaze-Head Debug Report

## 1. Executive conclusion
- Qwen3-VL 根因：
- RoboReward 根因：
- GRM 对照结论：
- 是否允许进入 CBH 实验：

## 2. Existing-repository integration map
- 实际推理入口：
- 实际 attention 路径：
- 实际评分路径：
- 本次最小改动：

## 3. Correctness gates
| Gate | Qwen3-VL | RoboReward | GRM | Evidence |

## 4. Bugs and fixes
### Bug 1
- 现象：
- 根因：
- 修复：
- 回归测试：
- 对历史结果影响：

## 5. Three-layer diagnosis
| Model | Attention controllability | Object probe | Reward leverage |

## 6. Phase and mapping results
- prefill / score token / decode：
- image/video span：
- tube/overlay：

## 7. Remaining uncertainties

## 8. Handoff to Counterfactual Binding Heads
- 可复用的现有仓库能力：
- 不应复用的错误假设：
- 冻结的 scoring/token mapping 配置：
```

---



## 18. Definition of Done

本排查任务只有同时满足以下条件才算完成：

- [ ] 已基于仓库实际代码画出完整调用链；
- [ ] 没有引入与现有框架重复的平行实现，或已说明不可避免原因；
- [ ] no-op 与 baseline 等价；
- [ ] sentinel 证明指定 attention 真实改变；
- [ ] hook 生命周期无泄漏；
- [ ] prefill、首个 score token、decode 已分别验证；
- [ ] image/video/multiview token span 已验证；
- [ ] bbox/mask/tube overlay 已人工检查；
- [ ] query/KV heads 与 ID 映射已验证；
- [ ] RoboReward/Qwen 的完整候选序列 scoring 已验证；
- [ ] GRM 的全部实际评分分支均覆盖；
- [ ] baseline/intervention 输入 parity 已验证；
- [ ] target-vs-all 与 evidence-preserving control 已比较；
- [ ] 三层诊断结果已分别报告；
- [ ] 每个确认 bug 都有修复前后测试；
- [ ] 最终结论没有仅凭 accuracy 或单个 top-K 得出；
- [ ] 已明确哪些实现与配置可以冻结并交给下一阶段 CBH 实验。