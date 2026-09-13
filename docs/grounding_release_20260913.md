# Grounding 全量处理最终交付（2026-09-13）

已完成全部 1213 条评测样本与独立 ranking 清单的全部 36 条 grounding。全量首轮后，重试全部不可用样本，再完成类别复核、多点跟踪和形态描述修复。最终目录保留每条记录的阶段来源；历史结果未覆盖。

| 数据集 | 总数 | 最终双端点可用 | 自动可用率 | 全帧有框 | 仍不可用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| evaluation | 1213 | 1116 | 92.00% | 357 | 97 |
| ranking | 36 | 33 | 91.67% | 25 | 3 |

这些比例是自动双端点可用结果经定向误定位剔除后的覆盖率，不是人工验证的定位准确率。双端点可用允许中间缺帧；全帧有框也不能排除目标漂移或 mask 扩张。

## 实际运行阶段

| 阶段 | 评测可用 | ranking 可用 | 报告 |
| --- | ---: | ---: | --- |
| 首轮全量 | 991 / 1213 | 28 / 36 | [全量报告](grounding_full_run_20260912.md) |
| 全部残留恢复 | 1100 / 1213 | 32 / 36 | [残留恢复](grounding_recovery_20260913.md) |
| 北寄贝类别复核与空响应放大 | 1104 / 1213 | 32 / 36 | [类别复核](grounding_final_20260913.md) |
| 多点跟踪 | 1117 / 1213 | 33 / 36 | [多点跟踪](grounding_refined_20260913.md) |
| 朝向、茶杯类别描述与已知误定位剔除 | 1116 / 1213 | 33 / 36 | 本报告 |

最后一阶段覆盖四条虾寿司序数候选不足与一条杯子参照关系失败。类别描述包含翻转虾的棕色腹面和尾扇，以及深色、小碗形茶杯；仍使用原始类别查询和确定性几何选择。纯鲑鱼和只有笔、托盘的负例保持无候选。模型未读取成功/失败标签或机器人实际抓取对象来决定目标。

对全部 15 条评测北寄贝样本的首末帧做助手视觉复核时，发现 `fail/ljx_lfz_task_1_2/2` 仍框中了虾；第二种形态描述也未纠正它。最终将这条样本的两个端点标为 `rejected`，并从两个可用 cohort 剔除。原框和原轨迹仅保留作诊断，详见 `quality_exclusions.json`。其余 14 对端点类别目视一致；这不是人工全轨迹审核，也不是准确率估计。

多点恢复样本 `suc/ljx_lfz_task_5_8/5` 的末帧 mask 实际分割了托盘，并把寿司排除为孔洞；其中间帧还出现过红积木漂移。此记录同样标为 `rejected` 并剔除。因此，本表已扣除两条已知质量问题，不能直接采用多点阶段的自动有框数量。

形态描述恢复的 `fail/ljx_lfz_task_3_9/32` 首末框覆盖第三个虾寿司，但部分中间采样框扩张到托盘；它仅满足双端点口径，不能视为任意中间帧可直接评分的正确轨迹。茶杯案例 `suc/ljx_lfz_task_5_2/8` 的首帧候选恢复为五个，但末帧身份核验仍未通过，继续保留不可用。

## 产物

- 评测 grounding：`results/mydata_bench/grounding_v2_release/sam3/grounding.jsonl`。
- 独立 ranking grounding：`results/mydata_bench/ranking_grounding_v2_release/sam3/grounding.jsonl`。
- 双端点 cohort：`results/mydata_bench/cohorts/auto_grounded_v2_release/`。
- 全帧 cohort：`results/mydata_bench/cohorts/auto_grounded_v2_release_full_frames/`。
- 每个结果根目录包含 `processing_audit.json`、`example_statuses.json`、`still_unavailable_examples.json`、`tracking_geometry_warnings.json`。
- 最后两阶段的配置、输入选择、日志、源码哈希与 ZIP 快照保存在 `grounding_v2_multipoint_trials/` 和 `grounding_v2_category_trials/`。

合并文件引用各阶段的原始 track、mask 和预览文件，使用或迁移时需保留这些来源目录。每条样本同时替换两个端点，不拼接不同尝试的轨迹。

## 仍不可用的样本

### evaluation

| 原因 | 数量 |
| --- | ---: |
| `ambiguous_target_candidates_with_visual_proposals_then_sam3_box_tracking` | 56 |
| `reverse_identity_not_verified` | 26 |
| `insufficient_instances_for_ordinal_with_visual_proposals_then_sam3_box_tracking` | 7 |
| `ambiguous_ordinal_geometry_with_visual_proposals_then_sam3_box_tracking` | 4 |
| `terminal_target_not_detected` | 2 |
| `assistant_visual_review_category_mismatch` | 1 |
| `assistant_visual_review_terminal_mask_drift` | 1 |

### ranking

| 原因 | 数量 |
| --- | ---: |
| `ambiguous_target_candidates_with_visual_proposals_then_sam3_box_tracking` | 3 |

双胡萝卜且指令只说“the carrot”的样本缺少唯一实例证据；四条“第三支笔”的首帧实际只有两支笔。部分序数候选横坐标近似并列；还有类别漏检及无法唯一返回首帧目标的轨迹。没有用任意挑选、降低几何门槛或切换目标 ID 强行填满。

## 验证与适用范围

- 126 项相关测试通过；最后一阶段只使用已有类别描述配置，没有改动推理源码。
- 两份全集清单、端点配对、视频身份和输入指纹、同帧 bbox、单轨迹 ID、mask 合法性及覆盖统计通过程序核验。
- 反向恢复仅接受唯一回到原始目标、首帧 IoU ≥ 0.5 的轨迹。
- cohort 的 ID 列表、episode manifest 和统计数量逐项一致。
- 类别和跟踪做过定向视觉抽查，未做随机人工准确率估计或全量逐帧人工标注。
- `tracking_geometry_warnings.json` 的四倍面积规则只提示复核，正常尺度变化也可能触发，不能当作正确率标签。

此次完成 grounding 和数据集合处理；attention head ranking 与 reward/attention 评分未重跑。历史 attention 结果仍对应旧 cohort，不能当作新集合的实验结果。
