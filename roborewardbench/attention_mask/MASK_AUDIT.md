# Attention-mask 实现审计

审计日期：2026-07-23。对象是当前 `roborewardbench/attention_mask` 与
Robo-Dopamine-GRM-2.0-8B-Preview（Qwen3-VL，Transformers 4.57.0）。

## 结论

当前默认 `boost_suppress` 的 layer/head 选择、bbox→绝对 token 映射以及 prefill/decode
施加位置均正确，没有发现 off-by-one、camera-span 漏施加、reference/text 污染或 decode
位置漂移。审计中补了两个 fail-fast guard：

1. 八图协议的 image-span labels 必须完整、唯一且不能出现额外 label；
2. eager attention mask 的 head 维必须是 1 或 `num_query_heads`，否则拒绝广播。

## 静态调用链

```text
prepare_inputs
  -> processor 生成 input_ids + image_grid_thw
  -> infer_image_spans 按 image_token_id 连续区间定位八张图
  -> target_position_set 将 before/after bbox 映射成绝对 LM key index
  -> intervention_positions 生成 target(+bias) 与 other endpoint image(-bias)
  -> registered_mask_hooks 按 layer 分组并只注册指定 query heads
  -> Qwen3VLTextAttention.forward 的 eager causal mask
  -> softmax(QK^T / sqrt(d) + causal_mask + intervention_bias)
```

head ranking 和 experiment 都使用模型的 `num_attention_heads`（query heads），不是
`num_key_value_heads`。Qwen eager attention 会先把 KV heads repeat 到 query-head 数，随后将
`[batch, heads, query, key]` mask 加到 logits，因此这里的 head axis 与 ranking 中记录的 head
index 一致。

## token/span 核对

- 八图顺序固定为 reference start/end、before 三 camera、after 三 camera；
- `target_role=both` 只包含 6 个 endpoint spans，不包含两张 reference；
- 单视角数据复制到三个 camera slots，bbox 也映射到对应三个 token spans；
- bbox 映射先按 processor 的 `image_grid_thw / spatial_merge_size` 校验 span 长度，再返回
  `span.start + local_grid_index`；
- `intersection` 策略会保留所有与 bbox 相交的 grid cells，避免小物体刚好落在 cell center
  之间而得到空集合；代价是 4×4 等粗 grid 上的实际干预区域可能明显大于像素 bbox；
- `target` 与 `other_image` 不重叠，二者并集严格等于所选 role 的 endpoint image tokens；
- wrong-region control 每个 span token 数匹配且不与真 bbox 相交；无法满足时明确缺失。

## hook/broadcast 核对

每层构造的 intervention tensor 是 `[1, num_query_heads, 1, base_key_len]`：

- 只有 ranking 选中的 head rows 被写入；
- 所有 query rows 使用相同的 key-side spatial intervention；
- batch 维广播；
- prefill key 较短时只截取已存在的 keys；
- decode key 变长时右侧补 0，prompt 内的绝对 image key index 不变；
- `decode_only=True` 跳过 `query_len>1` 的 prefill；
- `swap_bias=0` 返回严格 no-op，不复制/改写 mask。

## 实际模型 generate trace

对 frozen evaluation 的第一条样本、正式 excess-mass top-64、bias=4，在 GPU 上给 mask hook
之后追加只读 audit hook。prompt 长度为 892，抽查 ranking 第一名 L19H10；同层 H1 未被选中。
目标 key=194，other endpoint key=196：

| phase | observed mask shape | L19H10 target | L19H10 other | L19H1 target | newest key |
|---|---|---:|---:|---:|---:|
| prefill | `[1,32,892,892]` | +4 | -4 | 0 | 0 |
| decode 1 | `[1,32,1,893]` | +4 | -4 | 0 | 0 |
| decode 2 | `[1,32,1,894]` | +4 | -4 | 0 | 0 |

这直接证明当前 Transformers/Qwen runtime 中：选中 head 正确、未选 head 不变、prefill 与
decode 都命中原图像 keys、增长的 decode keys 没有被误加 bias。

heatmap smoke 对同一例子的 64-head 平均也符合预期：

| endpoint | baseline bbox mass | candidate bbox mass | candidate span mass |
|---|---:|---:|---:|
| before high | 0.001660 | 0.036741 | 0.036755 |
| after high | 0.004639 | 0.066931 | 0.066949 |

说明 intervention 不只是 tensor shape 正确，post-softmax attention 也实际集中到 target bbox。

## 自动测试证据

完整 `python -m unittest discover -s tests -v`：54/54 通过。其中 attention 专项覆盖：

- before/after/both span 集合和 reference 排除；
- bbox 三 camera 复制与小 bbox intersection；
- wrong region token-count match/disjoint；
- selected-head-only、batch/query broadcast；
- prefill、短/等长/扩展 key，decode 新 key 为 0；
- heatmap 按绝对 span、layer/head 聚合；
- dose-response curve 的 paired shift；
- discovery/evaluation 不重叠、人工 fingerprint、reward 不进入模型 item。

既有完整实验中 bias=0 的 125 个 intervention records 与 baseline 逐值完全相同，也提供了
端到端 no-op sanity check。

## 仍需保留的解释边界

- `suppress_image` 这个历史名字实际只对“非 target 的所选 endpoint image tokens”加负
  bias，不会压低 target；默认正式实验使用语义更清晰的 `boost_suppress`。
- 同一单视角图被复制到三个 GRM camera slots，因而 intervention 也施加三份。这与标准
  Robo-Dopamine benchmark adapter 一致，但不等价于真实三相机观测。
- heatmap 是最后 prompt query 的 endpoint attention，不代表所有 decode queries，也不是
  原视频逐帧 tracking。
- 实现正确不等于因果结论成立；结论仍要求 target 同时超过 wrong-region 和 low-ranked-head
  controls，并报告 paired coverage。
