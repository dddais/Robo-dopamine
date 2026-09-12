# 均匀指令注意力注入与原生证据对比：可复现实验说明

## Material Passport

- 目标目录：Robo-Dopamine；数据与结果根目录：session_20260911。
- 状态：候选方法已实现，数学和实际前向已核验；完整有效性验证仍进行中。
- 数据：固定846 cohort；discovery70包含28任务的完整视频组；validation660和ranking视频组116保持原划分。
- 本文说明实现与已观察证据，不宣称总验收已通过。反复查看validation后的结论属于数据集内自适应探索。

## 机制和原方案的区别

空间ROI相同不等于指令含义相同。原比例绑定把一半非视觉prompt质量移向任务span，但任务内部仍按旧注意力比例分配。如果颜色、方位或计数词原本只得到很少注意力，这一规则可以继续忽视它们。均匀注入向当前query能看到的全部任务token分配相同份额；不人工挑词、不改prompt、不访问标签或视频配对。

该假设结合PASTA的文本注意力干预与ASCD/VCD的推理时证据对比思想，均匀分配公式是本研究自行提出并实现的候选，不把它表述为上述论文的原公式。相关一手来源与机制对比见LITERATURE_20260911.md。

## 精确的域与公式

对每个被选head、每个query，p为施加原causal/padding mask后的attention概率。D_visual由已接受映射中当前scope的target与negative key并集构成。D_text由有效、未列入visual映射的原prompt token构成：其中也包含有效的格式和特殊token，不能把它误称为仅自然语言词。T_task是实际任务字符串覆盖的token，利用fast-tokenizer offsets定位，并要求input IDs精确round-trip、任务字符串只出现一次、task token均属于D_text。

先在D_visual中进行原视觉质量重分配，目标core乘exp(lambda)，再缩放到原视觉域总质量；域外保持原p。随后对D_text执行：

    p_new[text] = (1 - eta) * p[text] + eta * mass(text) * Uniform(visible task tokens)

主设置eta=0.5，lambda=4。可见性来自原布尔或加性mask，能够区别概率下溢为0的合法key与被mask遮挡的key；若当前query看不到任何任务token，则整行文字干预为恒等。两个域不相交。

因此，对固定当前层Q/K，有以下不变量：D_visual质量、D_text质量以及所有域外key权重分别保持；未选head沿用原SDPA；causal/padding位置不会获得质量。每个非task文字key保留一半原权重。每个可见task key获得至少0.5*mass(text)/可见task token数的份额。这些是逐层局部性质，不意味着后续层或最终输出保持不变。

默认all-query，覆盖所有可见query，未伪装成只修改最终readout。实现计算被选head的float32 attention变化乘V，再加回原SDPA结果；不覆盖原baseline或其它实验配置。

## 两个真实前向分支与读出

正分支：上述视觉质量重分配和均匀任务注入。

负分支：同输入、同scope、同head索引，仅对视觉core做lambda=-4的质量重分配，不施加文字绑定。

    z = (1 + alpha) * z_positive - alpha * z_negative

主设置alpha=1；第十六轮在预测前另行冻结alpha=0.5/2敏感性。所有原生类别使用同一个alpha，未加入类别偏置、类别特定权重或拟合阈值。Qwen/RoboReward在ANSWER格式前缀后读取五个原生奖励token的logits，全部五类一起softmax和argmax。raw_output为null，明确为likelihood读出，不伪装为自由生成文本。无干预baseline也使用完全相同的likelihood输入与读出。

一次干预预测需要正负两个整网前向；基线另算。与原单次attention steering相比有额外推理成本，不能只报告head预算。相邻k使用同一排名前缀，未为每个k重选head。k4/12等非8倍数会取层内固定顺序的部分head，应区别于原8-head组的监督profiling单位。

## Head选择和监督边界

均匀方法复用stage8已冻结的功能排名，尚未为均匀分配重新profiling。原profiling用discovery70标签选择层组：候选为第8–35层、每层8个视觉/文字双列表head，共224个；按两类NLL变化中较坏者排序，平局比较两类均值与层号。模型权重冻结。Qwen/RoboReward的NLL使用全部五类，Robometer profiling使用其原生success logit。

所以该路线包含少样本监督head选择，不能称零样本。新推理只读冻结head索引；不读label文件、任务编号、文件名或同视频对手。均匀分配本身不增加新的监督，也没有按这轮validation更改排名。

## 已核验与尚待核验的证据

数学和旧行为回归19项通过，包含因果遮罩、布尔/负无穷/有限最小值mask、可见零概率token、prefill/decode和分组query/key heads。原alpha1两模型64条实际smoke与比例绑定版本的baseline、negative logits逐值一致，positive确实改变，概率重建最大误差1.321e-7。审计见audit/uniform_binding_actual_20260911_163307.json。

全部五输入discovery的独立比较已进一步匹配4200条预测、60个条件；两个版本input IDs、prompt、baseline与negative均逐值一致。结果保存于analysis/uniform_vs_proportional_20260911_171350.json。举例：Qwen text_image last/k8，两种绑定MAE同为1.543，但suc准确率由比例版本44%变为均匀版本96%，总准确率37.14%变为50%；相同baseline为88% suc和31.43%总准确率。这支持均匀指令质量分配改变了类别间的取舍，仍不证明完整数据有效。

另一个反例必须一起保留：Qwen interleaved all/k64，均匀方法的suc为100%，高于比例版本92%，但总准确率64.29%低于65.71%，MAE1.429也差于1.343。因此不能称每个指标都优于比例方法。discovery仅25条suc，baseline此输入已96%，均匀新增suc只是一条。

第十六轮原alpha1的4200条概率重建最大误差1.192e-7，五个类别的同分支恒等均通过；新alpha候选仍须实际前向核验后才进入完整验证，派生结果不作GPU实测或最终效果。

## 完整验证与验收

原alpha1已冻结：Qwen text_image last k4/8/12、interleaved all k48/64/80、text_video all k48/64/80；RoboReward image_text all k24/32/40、text_image all k48/64/80、interleaved all k24/32/40、text_video all k24/32/40。其余输入的负结果保留。选择基于每输入最低MAE的discovery通过点，完整参数与源SHA保存在selection_uniform_binding_*_full_v1.json。

最终需要完整846、旧730和validation660，MAE、两类和总准确率、任务分布、同视频pairwise、head统计及匹配原方法/区域/低排名对照。总准确率门槛采用至少增加10个百分点。两套阈值、多个k、多个head不能重复计作输入或模型。不同attention机制的零散通过点也不能拼接成同一方法完成。当前只允许物理GPU0/1。


### 17:31 状态追加

第十六轮Qwen α.5/2、RR α2共96条实际smoke已全部通过，源分支逐值不变、概率重建最大1.192e-7；相应六个完整验证任务在通过后才注册。统一operator family的主候选、α/scope/k选择及完整结果前的覆盖检查另冻结在selection_uniform_family_primary_v1.json；该主选择目前尚无完整有效性结论。


### 17:40 同ROI配对的反例：机制解释必须收窄

以实际输入映射的visual-relative target key集合分层，discovery中相同ROI为20对/11视频组，不同ROI为22对/14组。uniform相对proportional的reward配对间隔在Qwen text_image last/k8的相同ROI组只增加.05，95%视频组bootstrap区间[-.632,.773]；不同ROI组增加.818，区间[.333,1.273]。RR text_image all/k64的相同ROI组反而−.05、区间[-.381,.300]，不同ROI组+.455、区间[.071,.833]。Qwen interleaved all/k64相同ROI间隔由proportional1.0降为uniform.8。

因此现有探索证据不支持“均匀任务注入已经解决同ROI下的指令区分”。总体suc变化不能替代这种条件配对检验。它仍是有数值候选的attention方法，但绑定机制解释需保留不确定性；所有60条件、120分层、5000视频组bootstrap及原始档位计数在analysis/uniform_pair_mechanism_20260911_173952.json。已核对每个直方图总数等于有效pair数；这是推理后分层关联，不是随机化mask因果实验，未改推理。


## 19:03 完整验证边界

原selection_uniform_family_primary_v1.json主方案已无法满足每模型三输入覆盖：Qwen text_image/interleaved失败、RR image_text/text_image失败。Qwen image_text α2的k48/64虽通过846/660描述门槛，但不能把单输入成功扩写成统一方法达标。详见PROGRESS_20260911_1845.md。

新增三分支方案属于独立冻结variant uniform_factorized_evidence_a1，使用相同正分支、原视觉负分支和实际任务抑制负分支；两模型各三输入仅在discovery通过，完整验证尚在运行。另一新候选uniform_evidence_robustprofile_a1仍保持两分支原uniform算子，改用真实读出的逐层功能profiling与保守NLL排序；目前在profiling阶段。不得将这些新候选的未来成功追溯为原uniform主方案成功。
