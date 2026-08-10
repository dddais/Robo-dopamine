需要注意检查的事项：
attention mask相关：
1.attention mask是否广播到ranking head的所有query行，而不只是last_prompt token，decode token（这个是用来ranking的）
2.attention mask是否正确施加到target obj对应的img_token的key 列。因为视频经过video processor之类的video span可能存在语义错位？需要仔细检查
3.ranking head是否排除了前n层
4.ranking head用的指标是 raw mass还是什么
5.ranking head用的是last_prompt token还是 decode token
6.wrong region是怎么选取的
7.+bias施加的target img token是最后一帧的bbox区域还是所有帧的bbox区域，-bias的施加的target img token是最后一帧的非bbox区域还是所有帧的非bbox区域

baseline相关：
1.text video的输入顺序
2.GRM 8张图的构造，有三视角就正常用，只有单视角的话，需要填充
3.各个模型的输入prompt
4.incremental模式是否构造正确，比如输入构造是否正确，score累计计算是否正确

grounding相关：
1.sam3能否区分“离A物体最近的杯子”这类空间关系
2.sam3的视频tracking能力
3.sam3的实时性
4.sam3的环境是否配好

数据处理相关：
1.采集顺序是否正确，需要人工审核
2.

结果相关：
1.检查结果总结的数据是否写对，有无错位情况