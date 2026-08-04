# LJX/LFZ 人工审核 N×K matrix 协议补充

## 结论边界

本轮是 **human-reviewed exploratory robustness rerun**。未经人工审核的
N×K 结果已经被观察，因此本轮不能标为 confirmatory、formal 或预注册验证。
它只回答：在冻结输入和相同 N×K 方法下，改用人工审核 target 后，探索性现象
是否仍然存在。

## 冻结的自动来源

- requests：grounding_auto_v2_low015_top40/requests.jsonl
- requests SHA-256：
  b07b284aed78b3a078b093d6ddcf3b55296b291b19bf5585da16cf1470c18977
- proposals：grounding_auto_v2_low015_top40/proposals.jsonl
- proposals SHA-256：
  0f92ef98a486c1a152aaf38129caa7f9cf56ea509b15bfccad50cb3ba26e5db9
- requests 固定为 755 条；proposals 固定为 755×3=2265 条。

人工产物只能写入
artifacts/my_dataset/ljx_lfz_cf_v1/grounding_reviewed_v1/，不得回写上述自动
来源目录。

## 审核规则与盲法

- 审核者可以看 instruction、冻结 terminal image，以及叠加在图上的 SAM3 candidate
  bbox 和候选 query/label/score 元数据；当前 UI 不读取或渲染 SAM3 mask，因此不能
  表述为审核者看过 proposals 的 mask；
- 审核时不得打开 reward 标签、baseline/steering 输出、head ranking 或旧 N×K
  分数；
- 同一个 review session 必须使用固定且非空的 reviewer ID，同一 output-dir
  禁止混用 reviewer；eligible 的三模型 target 均合法，target 图像必须与冻结
  request 的 terminal slot 和 SHA 一致，bbox 必须在图像边界内；
- RoboReward/Qwen 的页面只展示一个冻结 terminal PNG。若 processor 的 temporal
  merge 把多个采样帧组成同一个 token group（例如固定 8 帧时，末组可能同时包含
  倒数两帧），确认末帧 PNG 上的 bbox 不等价于也人工确认了该末组中的前一采样帧。
  因而本轮只能称为 terminal-image grounding 已审核；若要解释整个末 temporal token
  group，后续还需逐采样帧/token-group 可视化 preflight；
- ineligible 必须填写原因；禁止为凑数伪造 bbox；
- ground-audit 必须通过，且 audit 中的 request/review SHA 必须与
  attention-prepare 实际读取文件一致。

## ineligible、完整 group 与 S20

- attention_reviewed_v1/<model>/all.jsonl 保存所有逐条 eligible 样本，仅供
  固定 S20 ranking source 取数；
- complete_groups.jsonl 只保存组内所有 counterfactual examples 均 eligible
  的完整 group，只允许该文件进入 matrix evaluation 和成组指标；
- S20 仍使用原 label-free、嵌套的固定顺序，不因审核结果重选。任一固定 S20
  source 自身不是 eligible 时，ranking-cohort 必须硬失败。只有在确认是漏审或
  人工框错误时才允许按审核历史修正；若该 source 确实 ineligible，则原 S20
  协议本轮不能运行；
- 不允许静默用第21名以后的 source 替换。若确需替换 cohort，必须另立协议版本
  和输出目录，并把它报告为新的 cohort，而不是原冻结 S20 的复跑。

## 固定实验与输出隔离

四个配置为：

- exploratory_matrix_reviewed_roboreward_text_then_video.yaml
- exploratory_matrix_reviewed_roboreward_video_then_text.yaml
- exploratory_matrix_reviewed_qwen_video_then_text_attention8.yaml
- exploratory_matrix_reviewed_grm.yaml

共同方法仍为 excess-mass、skip8、N={5,10,20}、K={8,32,64}、bias=6、
all-query。输出只能写入
outputs/my_dataset/ljx_lfz_cf_v1/reviewed_matrix_v1/。

该 N×K matrix 只使用 target bbox；虽然审核表收集 wrong-region bbox，但本轮并未
执行 wrong-target、low-rank 或 layer-matched-random controls。wrong bbox 也尚未
通过 processor-token 等量且不相交的 causal-control preflight，不能声称这些 control
已经可运行。

## 比较口径

若人工审核导致样本减少，不得直接把 reviewed 子集分数减去旧 755 条总分。必须：

1. 用 complete_groups.jsonl 过滤旧 unreviewed records，在完全相同 ID population
   上重新评分；
2. reviewed score 使用同 variant 的旧 records 作为 --reference-records，逐例
   要求 shared baseline prediction/progress 完全一致；不一致即视为 runtime/code
   drift，禁止解释 steering 差异。
