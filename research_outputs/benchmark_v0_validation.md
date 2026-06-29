# Benchmark v0 Validation / Benchmark v0 校验

## Summary / 总结

- Status: **PASS**
- Episodes: 1
- Errors: 0
- Warnings: 0
- Expected current non-Markovian episodes: `xzx_radio_sub23`

This validator checks file existence, required fields, event order, non-Markovian event presence, negative latch types, and the current data-scope invariant that only `xzx_radio_sub23` is a verified non-Markovian episode.

该校验器检查文件存在性、必需字段、事件顺序、非马尔可夫事件是否存在、负事件 latch 类型，以及当前数据边界：只有 `xzx_radio_sub23` 是已核验非马尔可夫 episode。

## Episodes / Episodes

| Episode | Valid | Events | Order OK | Non-Markovian events | Mean confidence | Issues |
|---|---:|---:|---:|---|---:|---:|
| `xzx_radio_sub23` | true | 6 | true | `button_press, indicator_green` | 0.86 | 0 errors / 0 warnings |

## Reproducibility / 可复现性

```bash
conda run -n robo-dopamine python research/validate_benchmark_v0.py
```
