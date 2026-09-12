# 一手文献核验与候选机制（2026-09-11）

## Material Passport

模式：相关工作核验与实验假设构建。15 篇 arXiv 的摘要页、HTML 正文已实际获取，原文与 SHA256 清单保存在 `results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911/references/`。未上传本地数据。这里的论文结论均是作者在各自任务上的报告，不是本数据集的实验结论；会议信息未独立核验，故不沿用计划中的会议信息。

| 来源 | 已核实的机制 | 对当前研究的启发与限制 |
| --- | --- | --- |
| [PASTA, 2311.02262](https://arxiv.org/abs/2311.02262) | 对选中 head 的非强调 token 权重乘 α，再全局归一化；通过小数据集逐 head 干预效果 profiling 选择 head，并取多任务高排名交集 | head 应以功能效应验证；raw localization mass 只是代理。原文不保持视觉总量。与计划最早 PAI 链接写反的问题一致 |
| [PAI, 2407.21771](https://arxiv.org/abs/2407.21771) | 图像 key 的 pre-softmax 分数加 α·绝对值；配合去除纯文本分布先验的 logit 对比，减轻 text inertia | 同时处理模态权重与输出先验；并非简单统一 +bias，也未在机器人进度上验证 |
| [ASCD, 2506.14766](https://arxiv.org/abs/2506.14766) | 用 text/visual attention ratio 的跨样本投票识别 text-centric heads；正向增强视觉、负向压制关键视觉 token，再组合两分支 logits，带 plausibility 截断 | 支持用同一样本反事实注意力分支去除先验，但两分支成本和解码路径需独立测量 |
| [CAST, 2605.04641](https://arxiv.org/abs/2605.04641) | 利用 caption query 的注意力激活模式识别敏感 heads，并估计其输出 steering directions | 不局限于注意力权重，也可干预 head 输出；caption 目标与机器人指令完成判断仍有差异 |
| [Gaze Heads, 2606.14703](https://arxiv.org/abs/2606.14703) | 在漫画叙述中利用注意力与当前描述区域的关联识别 gaze heads；选定 head 的区域 attention-mask 干预改变描述对象，支持生成中动态切换 | 指向哪个物体与正确判断指令是否完成不是同一能力；文中 all-head 干预会破坏生成，小 k 的选择标准差异大 |
| [HAS, 2607.17994](https://arxiv.org/abs/2607.17994) | 使用连续帧重要性分布引导视频总结，保留未高亮帧的上下文 | 提醒全帧统一强度可能丢失动作顺序和背景关系；时间权重应平滑而非直接删除帧 |
| [Arbitration Failure, 2604.09364](https://arxiv.org/abs/2604.09364) | 区分视觉证据已编码与最终选择；MAC/logit lens、全序列 activation patching、线性和 SAE 残差 steering | 定位提升不能保证最终读出采用该证据；只改最后 token 可能漏掉视觉表征传播路径 |
| [VLA Driving Attention Steering, 2608.17095](https://arxiv.org/abs/2608.17095) | 在 Qwen3-VL backbone 上对检测到的交通对象加有界 attention bias，记录每次实际注入 | 轨迹响应不代表推理文本响应；必须验证 hook 真正到达当前推理分支，本研究逐层保存诊断 |
| [Attention is Case-Sensitive, 2608.03711](https://arxiv.org/abs/2608.03711) | 大小写影响文字注意力；VLM 中同时改变图文总量与视觉内区域集中度 | 两个变化需要拆开；注意力集中与任务性能可能相背，直接支持独立保持模态总量的对照 |
| [Localization Heads, 2503.06287](https://arxiv.org/abs/2503.06287) | 少量冻结 heads 的 text-to-image 注意力可以定位对象；筛选考虑视觉注意力和空间集中程度 | 区域定位 head 不是完成度判别 head；排名不能只解释为通用 reward 重要性 |
| [Attention-Guided Safety Filter, 2606.09749](https://arxiv.org/abs/2606.09749) | 从 VLA attention 获得当前目标，再结合 tracker 与 CBF 避障 | 是下游显式控制器利用定位信号的例子，不证明单独 attention steering 足以改进 reward |
| [Voita et al., 1905.09418](https://arxiv.org/abs/1905.09418) | 专门化 head 承担主要功能；用随机门控与 L0 松弛裁剪大量 head | 支持稀疏干预；文本机器翻译中的发现不能直接外推到多模态奖励模型 |
| [VCD, 2311.16922](https://arxiv.org/abs/2311.16922) | 对原图与受扰图输出分布做对比，抑制语言先验和统计偏差 | 可把 wrong-region/目标抑制分支用于样本内证据对比；必须保留五档候选，不得人为压成 1/5 |
| [Robometer, 2603.02115](https://arxiv.org/abs/2603.02115) | 帧级进度监督加轨迹间偏好监督；成功、进度 head 的训练目标不同 | progress 与 success 必须各自匹配 baseline，不能混换输出并将收益全部归于 steering |
| [SOLE-R1, 2603.28730](https://arxiv.org/abs/2603.28730) | 视频时空 CoT 与连续逐时刻进度；用于在线机器人 RL | official 的首帧/前帧/当前帧拼图与预测递推必须一致，最终输出是绝对进度 |

## 原因分析与研究顺序

原 ±6 bias 有两个效应：目标/背景赔率乘 exp(12)≈162755，且视觉域总注意力变化。由于 fail 多于 suc，整体分数下移可能看似改善 MAE 和总准确率。若背景包含盘子、参照物或其它杯子，压低背景还可能破坏“放入盘中”“最左边”“距离某物最近”等关系判断。仅保留目标对象本身不充分。

第一轮隔离模态总量因素：对 D 域的权重按 exp(λ·1[T]) 倾斜，在 D 内重新归一化并恢复原域质量。固定 Q/K 时域外权重逐项不变；最后帧设置仅保持该时域的质量，全帧设置保持所有视觉质量。通过对原 SDPA 输出添加精确权重差乘 V 的残差实现，未选 head 保留原始实现，避免用新的矩阵乘法重建全部注意力。

如果该候选仍牺牲 suc，后续依证据选择：①背景保留的关系邻域 soft steering；②只根据当前注意力的饱和度调节剂量，避免已集中 head 再过度集中；③按指令差异产生的目标注意力对比排序，而不是只按 suc raw mass；④ASCD/VCD 风格分支对比。以上都只能访问视频与当前指令，不得根据标签、任务 ID、文件路径或配对排名推断奖励。

本研究只将这些作为机制上合理的假设，是否达到两模型×三输入的验收由完整实验决定。

## 本地机制诊断补充

`analysis/roi_quantization_v1.json` 对 846 条已接受的输入及 543 对同视频样本进行了掩码比较。普通图像协议只有 384 个视觉 key，末帧目标中位数为 2 个 key，476 条的目标只有 1–2 个 key；官方 Robometer 有 2400 个视觉 key，目标中位数为 6 个 key。然而两种分辨率的全帧相同掩码对数均为 183，末帧均为 185。它们对应不同颜色或空间顺序的任务指令，说明提高分辨率并没有自动提供额外的区域干预区别。该结果支持进一步检验 PASTA 式指令强调与视觉 steering 的结合，不能据此宣称 grounding 的人工真值已重新审查。

## 固定掩码与完整验证的描述性分层

进一步将已完成的 660 条 validation 结果按同视频 suc/fail 是否有相同 ROI 分层，其中 421 对可完整匹配：144 对相同 ROI、277 对不同 ROI。RoboReward interleaved、质量重分配 λ8、k32 的归一化 suc−fail 平均分差，在相同 ROI 组变化 −0.0382，在不同 ROI 组变化 +0.2175；text→image、λ4、k32 分别为 −0.0260 与 +0.1913。相邻 k 的方向一致。

这说明当前方案的配对区分收益主要出现在区域本身提供差异的样本上，支持继续检验任务绑定。它是事后描述性机制检查：分组同时可能包含不同任务难度；相同 mask 也不意味着相同模型激活，因为指令仍然不同。因此不能将该结果写成 grounding 的因果质量评估，更不能据分组改写推理分数。来源：`analysis/paired_mask_20260911_131852.json`，原评分检查点 `checkpoint_20260911_130601_validation`。

## 证据对比的真实分支消融

Qwen text→video 的证据对比在完整 validation 首次出现明显的双类改善。λ8、最后帧、k48 时，baseline 的 MAE/总准确率为 1.415/15.00%；只读取实际保存的正向 attention 分支为 1.132/19.24%；按预定 α1 合成后为 1.062/28.79%。正分支的 suc/fail 准确率为 48.33%/5.76%，合成为 55.98%/16.19%；baseline 为 43.06%/2.00%。负向分支的原生预测也完整报告，未利用标签拟合新 α。

因此该条件的增益不只来自更换 head 排名或五类 likelihood 读出：在相同输入、排名、读出位置和正向 attention 下，加入目标抑制分支的对比有额外贡献。这支持 ASCD/VCD 所强调的样本内证据对比，但尚不能外推为所有输入都有效。来源：`analysis/branch_ablation_20260911_135037.json`，仅从已经记录的完整原生 logits 做分支读出，不添加新推理、不压缩为端点。


## 2026-09-12 00:16 新增一手来源：Context-aware Decoding

[Trusting Your Evidence: Hallucinate Less with Context-aware Decoding, 2305.14739](https://arxiv.org/abs/2305.14739) 的摘要页与HTML正文已获取，读取第2.2节公式、超参数讨论与结论。CAD用含context和不含context的同模型分布构造 `softmax((1+α)z_context−αz_no_context)`，也可写成原分布乘条件PMI比例。论文是在摘要忠实性、知识冲突任务上检验；不能将作者报告的收益转移成机器人完成度效能。论文在摘要任务用α=.5、知识冲突任务用α1，这也不能当作本研究已选参数。

与已实现ASCD/VCD分支的区别是反事实条件：可进一步问当前模型对“视频看起来像已完成”的先验是否高于对当前具体指令的敏感性。如果以后把具体指令视为context，必须实际构造或严格验证指令内容不再传播的负分支；少数head的−4抑制仍保留许多指令传播路径，不能直接称无指令条件。保留所有原生类别，不使用固定类别校准offset、不以predicted class决定规则。这个启发尚未实施或评分。

来源与SHA256：`references/2305.14739_abs.html`、`references/2305.14739_html.html`及对应txt，`references/context_aware_decoding_manifest_20260912_001440.json`。仅下载公开论文，未上传本地数据；原15篇清单保留，本次作为第16篇补充。


## 2026-09-12 01:34 新增一手来源：V* 与局部视觉细节

[V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs, 2312.14135v2](https://arxiv.org/abs/2312.14135)（Penghao Wu、Saining Xie）摘要与HTML正文已取得。已读取3.1视觉工作记忆、3.2训练数据、3.3视觉搜索及附录A.3/A.4实现。SEAL保留全局图像、问题、找到的目标裁剪与坐标；搜索根据目标/场景线索递归查看子区域，最终把局部观察送回VQA模型。机制动机是缩小后的全局视觉特征可能没有保留需要的细节，单纯增加某些已编码token的注意力不能创造原先丢失的像素证据。

原论文**有专门训练**：VQA侧有投影模块/LLM指令微调，搜索侧有定位decoder与LoRA等训练；不是在任意冻结Qwen/RoboReward上加裁剪就必然有效的零样本方法。工作记忆还包含目标名字与坐标，其提示/视觉token数量都变了。论文5.3表5在部分通用基准出现退化，不能说视觉搜索普遍增益；也没有提供本机器人数据的效能证据。

对当前研究只形成后备假设：应区分“原低分辨率视觉信息不足”与“信息存在但未被任务读出利用”。已有ROI量化审计显示低分辨率目标经常只有1–2个视觉key，但提高分辨率并未自动消除同ROI指令配对；因此不能预先认定分辨率是主要原因。若研究视觉细节，必须保留全局关系上下文、明确额外token/计算预算、对照相同高分辨率baseline与attention分支，继续使用已接受的grounding，而不能裁掉所有参照物或引入标签控制的观察选择。

来源与SHA：`references/visual_search_vstar_manifest_20260912_013044.json`及所列摘要/正文HTML、txt，作为第17篇补充。本次只读取公开论文，没有访问作者邮箱、调用外部VQA模型或上传本地数据；此时没有据该文登记第29轮或运行高分辨率/crop实验。


## 2026-09-12 12:56 原有一手正文复读与第30轮机制边界

复读CAST正文4.1–4.4（本地已核验`references/2605.04641_html.txt`）：其公式10是caption/非caption的跨样本平均head输出差，公式11把预计算方向加到被选head的输出；head选择来自caption类型probe分类准确率。本研究第30轮改用同一样本当前QKV下三种局部attention输出构造方向，并固定原head输出范数，未训练caption probe，也没有使用论文的预计算跨样本方向；因此准确定位为受head输出干预启发的独立候选，**不是CAST原方法复现**。

复读Voita等的门控段落（`references/1905.09418_html.txt`）：head-specific Hard Concrete门控、可微L0罚、重参数化，作者从收敛模型继续联合优化模型θ和门控ϕ。这可以支持继续思考head组合的端到端选择，但不能直接证明只学习attention控制标量且冻结机器人reward权重有效。理论备选9仅登记了思路，尚未实现或产生新效能。两篇均为已核验的原15篇之一，本次复读不增加文献篇数。


## 2026-09-12 14:01 新增一手来源：ReFT与可学习表示干预

[ReFT: Representation Finetuning for Language Models, 2404.03592v3](https://arxiv.org/abs/2404.03592)（Zhengxuan Wu、Aryaman Arora等）的摘要与HTML正文已获取，读取3.1–3.3、5及附录F.1/F.2。LoReFT在冻结LM中使用`h′=h+Rᵀ(Wh+b−Rh)`，R的行正交，训练R/W/b；DiReFT去掉正交及差分约束，成为`h′=h+W₂ᵀ(W₁h+b)`。这支持把低维可学习干预放在内部表示，而不是直接改变输出类别阈值。注意论文分类任务另训练classification head，不能声称其所有实验都冻结读出；本研究若借鉴仍须保持原输出embedding和全部五类。

论文5节明确主要研究LLaMA系列，把视觉语言模型列为后续方向；不能据此认定对当前Qwen/RoboReward有效。其评估实践段反对在test上爬坡，附录F还表明很小的rank-1干预能记忆长文本或许多任意输入输出映射。因此“参数少”“模型主体冻结”不构成泛化证据，当前70条/28视频组再代入必须和未参与梯度的validation区分，且validation已反复查看，仍不能称独立确认。

可能后备方向是对attention steering产生的**内部增量**学习低秩变换，而不是对整个残差流加任意偏置。这样在无steering增量时可严格返回原网络路径，便于检验新参数是否实际依赖attention证据。这是受ReFT启发的受限新假设，并不是LoReFT/DiReFT公式原样复现；当前第32轮仍按已登记协议训练，没有新增第33轮policy或GPU实验。

来源与SHA256：`references/reft_manifest_20260912_135858.json`及所列公开摘要/正文HTML和txt，作为第18篇补充；没有上传本地数据或调用外部模型。
