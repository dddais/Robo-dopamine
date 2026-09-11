"""Build a review candidate from the completed, audited matrix only.

This reporting supplement does not execute inference or change any estimand.
It is descriptive and was written after intermediate results were available.
Outputs use exclusive creation; the reviewed final document is a separate step.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .prepare import OUT, ROOT
from .write_records import create_text, table


THRESHOLDS = ['0.125/0.875', '0.2/0.8']
CONDITIONS = [f'{s}:target:{k}' for s in ['last_frame', 'all_frames'] for k in [8, 32, 64]]
PROTOCOLS = ['video_text', 'text_video', 'image_text', 'text_image', 'interleaved', 'official']


def num(v):
    return 'NA' if v is None else f'{v:.4f}'


def pct(v):
    return 'NA' if v is None else f'{v:.2%}'


def acc(d, threshold=THRESHOLDS[0]):
    return ' / '.join(pct(d.get('accuracy', {}).get(threshold, {}).get(s, {}).get('rate_all_expected'))
                      for s in ['all', 'suc', 'fail'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', default='report_candidate_v1.md')
    args = parser.parse_args()
    audit = json.loads((OUT / 'completion_audit_v1.json').read_text())
    index = json.loads((OUT / 'analysis_v1/index.json').read_text())
    digest = json.loads((OUT / 'interpretation_v1.json').read_text())
    assert audit['complete'] and not audit['issues']
    assert audit['evidence_checks']['predictions_observed'] == 197292
    assert len(index['experiments']) == 12 and index['rows'] == 468
    names = [f'{model}_{p}' for model in ['meter', 'sole'] for p in PROTOCOLS]
    data = {n: json.loads((OUT / 'analysis_v1' / f'{n}.json').read_text()) for n in names}
    holm = json.loads((OUT / 'analysis_v1/holm.json').read_text())
    rankings = json.loads((OUT / 'head_overlap_v1/rankings.json').read_text())
    assert all(holm[p]['family_size'] == 72 for p in ['cohort', 'holdout'])
    assert all(len(data[n]['conditions']) == 19 for n in names)
    # Links are relative to the final destination mydata_bench/*.md.
    prefix = '../results/mydata_bench/experiments_v2_addbase/'
    def link(label, path):
        return f'[{label}]({prefix}{path})'
    lines = []
    def prose(s):
        lines.extend([s.strip(), ''])
    def tab(headers, rows):
        lines.extend(table(headers, rows) + [''])
    def summary(n, c='baseline', pop='cohort'):
        return data[n]['conditions'][c][pop]

    prose('# Robometer-4B 与 SOLE-R1-8B：新增 baseline 与 attention steering 研究总结')
    prose('## Material Passport')
    prose('- Origin Skill: academic-research-suite / experiment-agent\n'
          '- Origin Mode: run + validate\n'
          '- Origin Date: 2026-09-10\n'
          '- Verification Status: ANALYZED\n'
          '- Version Label: exp_plan_addbase_summary_v1\n'
          '- 工作目录：`/mnt/public1/dais/workspace/Robo-Dopamine`，与计划中的 `/home/dais/workspace/Robo-Dopamine` 指向同一仓库。\n'
          '- 范围：执行根目录 [exp_plan_addbase.md](../exp_plan_addbase.md) 的全部主线要求。矩阵完整性和逐步证据已审计；未做整矩阵第二次推理复现。')
    prose('## 1. 结论与完成范围')
    prose('已完成两模型各五种输入顺序和额外官方输入，共 **12 个配置、228 个 baseline/干预条件**。'
          '每个配置包含全量 baseline，以及两个时域 × 三个 k × target/wrong-region/low-rank 的 18 个干预。'
          '按配置、条件、example_id 取追加日志最新记录，共 **197,292 条终点尝试记录**；该数量包含明确报告的无效输出，不能称为全部有效预测。')
    prose('其中有效输出 **196,797 条**，答案格式失败 **315 条**，wrong-region 不可构造 **180 条**，'
          '其它未分类无效记录为 0。SOLE 官方输入的全部 16,441 条终点均有效。')
    rows = []
    for model in ['meter', 'sole']:
        for pop in ['cohort', 'holdout']:
            a = digest['aggregate'][f'{pop}/{model}']
            direction = a['complete_mae_directions']
            higher_significant = sum(
                r['paired_mae_change'].get('complete', False) and
                r['paired_mae_change'].get('mae_delta', 0) > 0 and r['holm_p'] < .05
                for n, e in digest['experiments'].items() if n.startswith(model + '_')
                for r in e['populations'][pop]['targets'].values())
            rows.append([model, pop, f"{a['complete_target_comparisons']}/{a['target_conditions']}",
                         '/'.join(str(direction.get(k, 0)) for k in ['lower', 'equal', 'higher']),
                         len(a['complete_lower_mae_holm_lt_05']),
                         higher_significant,
                         len(a['descriptive_target_lower_than_both_controls']),
                         '/'.join(str(len(a['complete_lower_mae_and_both_classes_strictly_higher'][t])) for t in THRESHOLDS)])
    tab(['模型', '样本集', '完整 target 比较', 'MAE 降/平/升（完整比较）', '下降且 Holm p<.05', '上升且 Holm p<.05',
         'target MAE 小于两控制（共同有效样本）', 'MAE 下降且 suc/fail 都严格提高：严格/宽阈值'], rows)
    prose('**Robometer 的 MAE 收益较一致，但端点准确率存在类别权衡。** '
          'cohort 的 36 个 target 都降低 MAE，也都在四条件共同有效样本上优于两控制；'
          '然而两套阈值均没有一个条件同时严格提高 suc 与 fail 准确率。'
          '因此本轮支持特定离线误差和区分度收益，不能写成成功、失败判断全面改善。')
    prose('**SOLE 的效果依赖输入协议、干预范围与 k。** 完整矩阵同时包含改善、退化和格式失败；'
          '官方输入与普通单次输入应分别解释。上表对不完整比较保留在 36 项总数中，'
          '下文完整列出六个 target，避免只报告最低 MAE 点。')
    prose('SOLE 官方输入的 cohort baseline MAE 为 1.3121；last-frame k8/32/64 为 '
          '1.3132/1.2199/1.2317，all-frames k8/32/64 为 1.3369/1.3972/1.3298。'
          '因此官方输入下只有 last-frame k32、k64 降低 MAE，全帧三种 k 均上升；'
          '“更多帧受到干预”没有带来一致收益。每个条件的统计不确定性与控制见后表。')
    prose('SOLE 的 22 个完整比较中，MAE 下降与上升各 11 个；没有 MAE 改善通过 Holm 校正。'
          '官方六个 target 的 cohort 95% CI 均包含 0、Holm p 均为 1；'
          '例如 last-frame/k32 的 ΔMAE=−0.0922，95% CI [−0.2019, 0.0164]，未校正 p=0.1014。'
          '因此这些官方改善点仍是描述性候选，不能写成已证实的稳定收益，也不能用未显著来证明没有效应。')
    prose('另有一个明确的 cohort 负结果：SOLE interleaved/last_frame/k64 的 ΔMAE=+0.1501，'
          '95% CI [0.0674, 0.2303]，Holm p=0.0190。留出同样呈上升方向（+0.1342），'
          '但 Holm p=0.1482，未通过校正；两样本集有重叠，不能视为两次独立检验。')
    prose('这组结果没有确立跨模型通用的最优输入顺序、k 或时域。历史 GRM 的较大收益与本轮新增结果'
          '共同表明需要区分“输出分数改变”“指令区分改善”和“双类端点判断改善”。'
          '模型训练、读出、分辨率和官方递推均不同，本轮不能单独识别是哪一因素造成模型间收益差异。')
    tab(['计划步骤', '交付'], [
        ['step1 新增入口', '[meter_eval](meter_eval/__main__.py)、[top_eval](top_eval/__main__.py)、[共享实现](addbase_eval/README.md)；top_eval 的模型是 SOLE-R1-8B'],
        ['step2 check.md 核对', '第 3 节实现核对、13 个契约检查、官方 readout 局部 parity、全矩阵证据审计'],
        ['step3 全部实验', '[12 个配置](configs/v2_crossmodel_addbase)、' + link('完整数值表', 'analysis_v1/full_tables.md')],
        ['step4 五类指标与结论', '本文；每配置 exp_record、task CSV、配对分档、具体 top-8 与 top-8/32/64 重合度']])

    prose('## 2. 样本、指标与统计口径')
    tab(['样本集', '总量', 'suc / fail', 'task 子集', '用途'], [
        ['full', 1213, '407 / 806', 34, '每个 baseline 的完整样本'],
        ['cohort', 846, '268 / 578', 28, '用户认可的 auto_grounded_v2；所有干预与同组 baseline 比较'],
        ['holdout', 730, '234 / 496', 28, '排除原 ranking 清单全部 36 个视频组；与 cohort 重叠']])
    prose('实际有效 ranking 为 34 条；留出排除使用全部 36 个候选视频组，包含未进入有效 ranking 的两组。'
          'full 中的 task3_5、task3_9、task4_4、task4_5、task5_5、task5_9 完全不在 cohort；'
          'attention 结果覆盖的是 28 个 task 子集。留出仍来自相同任务域，并非外部任务集，也不是一次独立复现。')
    prose('固定标签为 suc=5、fail=1。主 MAE 使用五档奖励 `r(p)=1+Σ[p≥t]`，'
          '`t∈{.125,.375,.625,.875}`，计算 `mean(|r(p)−label|)`。'
          '它衡量这批指令一致性标签上的误差，不是有逐帧真值的物理完成百分比回归。'
          '另存连续映射 `1+4·clip(p,0,1)` 的 MAE，但不替换主指标。')
    prose('端点准确率独立采用两套 `(low,high)=(.125,.875)、(.2,.8)`：'
          'fail 仅在 `p≤low` 时正确，suc 仅在 `p≥high` 时正确，中间值均不正确。'
          '本文准确率以固定期望样本数为分母，无效输出不计正确；分析 JSON 同时保存有效分母的比率。'
          'MAE 表示有效输出的描述性均值，并另给无效项误差在 [0,4] 下的全体上下界。')
    prose('最终 `metrics.csv` 保留旧 `acc_*`（有效分母），并新增含义明确的 `acc_valid_*`、'
          '`acc_fixed_*`、`correct_*`、`expected_*` 列；`index.json` 记录列口径。'
          '本文和完整 Markdown 表使用 `acc_fixed_*` 对应口径，不能把旧 CSV 的 `acc_*` 当成固定分母准确率。')
    prose('预测分布同时保存奖励档与等宽进度区间（边界 `.2/.4/.6/.8`）；不能互换，'
          '更不能把某分布最低档数量直接当作两套端点准确率的正确数。SOLE 支持负百分比，'
          '归一化保留负进度，并按既定解析规则将越界值裁剪到 [−1,1]；负值落最低档。'
          '连续配对差使用这一归一化、范围裁剪后的进度，仍可超过 1；未裁剪的原百分比另存，不能与该差值混称。')
    prose('同视频配对按 `source_suc_id` 连接并核验 SHA256。完整 cohort 最多 543 对，'
          '另有 35 个 fail 的来源 suc 未进入 cohort；543 对并非 543 个独立视频。'
          '报告连续 suc−fail、每 10% 差值分布及离散差 `<0/0/1/2/3/4`，'
          '并在相同有效配对上比较变化、补充视频等权均值。')
    prose('ΔMAE=干预−baseline，负值表示误差降低。采用视频组 bootstrap 5000 次给出点对点 95% CI，'
          '视频组 sign flip 10000 次，seed=20260909；检验依赖视频组可交换性/对称性假设。'
          'cohort 和 holdout 各固定 72 个 target 比较做 Holm，不完整比较以 p=1 保留在 family 中。'
          '控制、阈值、task 和最佳 k 的描述不额外声称经过这一校正；CI 也不是同时置信区间。'
          '描述性补充在部分结果产生后编写，本研究不声称正式预注册。')
    constants = json.loads((OUT / 'analysis_v1/constant_reference_metrics.json').read_text())
    tab(['恒定输出 p=0：始终失败', 'MAE', '准确率 all / suc / fail'],
        [[pop, num(constants['0.0'][pop]['mae']), acc(constants['0.0'][pop])] for pop in ['full', 'cohort', 'holdout']])
    prose('cohort 中 fail 占 68.32%，始终判失败也能取得该总准确率，但 suc 准确率为 0，'
          'balanced accuracy 为 .5。这是解释模型总准确率所需的类别基率参考。')

    prose('## 3. 官方输入与 check.md 实现核对')
    prose('Robometer 使用 checkpoint 训练过的 prog_token 和 progress MLP；10-bin logits 经 softmax 后'
          '对 [0,1] 等距 bin center 求期望。738 个权重张量严格加载，success probability 单独保存。'
          '官方输入为 instruction 后接 image/prog_token 交错序列，使用原生图像尺寸；'
          '普通 image_text/video_text 把最终 prog_token 放在指令之后，保证最终读出能读取指令。')
    prose('SOLE 官方输入直接取 RewardGen 原文 system/user prompt 与首/前/当前帧拼图函数，'
          '八个采样时刻对应七次模型预测，首时刻为 0。每个干预条件用自身上一时刻绝对进度递推；'
          '**绝对进度不累加**。全部 SOLE 使用 greedy、512 新 token 上限；官方 RewardGen 的'
          'temperature=1、top_p=.9、top_k=50、max_tokens=200 与本轮不同，因此这里是官方输入构造，'
          '不是官方随机采样结果或论文全部实验的逐参数复现。')
    prose('SOLE official 的 ranking 在第七步未干预 prompt 上计算，前驱取 baseline 自身第六步预测；'
          '34 个发现样本均有有效前驱。得到的固定 head 集合用于各步干预，不在每一步重排，也不使用标签作为进度上下文。')
    prose('五种普通输入固定八个采样时刻、正面单视角、图像 max_pixels=50176；'
          'interleaved 在图像间插入 Robometer prog_token 或 SOLE 帧次文字。'
          'native video 将八源帧编码为四个 tubelet。官方 Robometer 保留 640×480 原生大小，'
          'SOLE 保留官方拼图上下文，故 official 对普通输入的差异同时涉及分辨率或时序上下文。'
          '本轮不改变旧 GRM 三视角填充实现；SOLE 选官方 external-only 分支。')
    tab(['check.md 检查项', '实现与验证口径'], [
        ['all-query 广播', 'B×H×1×K 的 key bias 广播至所有 prefill/decode query，保留原 causal/padding mask；生成文字 key bias 为 0'],
        ['视觉 key 对齐', '逐项核对真实 token span、grid_thw、源帧、tracking 帧与 bbox；native video 使用两源帧 bbox 并集'],
        ['ranking', '未干预 mean raw mass，排除零基 L0–L7；Robometer 最终 prog_token，SOLE 最后 prompt token'],
        ['wrong-region', '在相同时域按网格距离选远处非目标单元，等 token 数且不相交；不足时仅该控制不可用'],
        ['low-rank', '同一排名最后 k 个合格 head，仍排除前八层；未逐层匹配，不能视作完全相同的层分布'],
        ['±bias 时域', '+6 目标 key；−6 同选定视觉域的非目标 key。last 为末图/末 tubelet；all 为全部输入图域'],
        ['SOLE 官方 last/all', '每一步 last 为与当前帧内容相交的 key 域，all 为该步拼图三时刻；包含自身前驱反馈'],
        ['输入与递推', '逐样本保存完整 prompt、token 哈希、各步原文、progress_curve；标签文件与模型输入分开'],
        ['结果对齐', '按 example_id 与 condition 审计，无缺失/额外 ID；同视频配对另核验哈希']])
    prose('**几何限制：矩形相交不是像素级隔离。** 确定性选取第一条 ranking 样本的网格核查中，'
          'SOLE official 当前域有 130 单元，目标 6 单元；34 个域单元不完全位于当前内容内部，'
          '其中 10 个也与前一帧内容相交。因此早期协议中“黑边/未选时刻不受负 bias”的绝对措辞过强，'
          '应以 '+link('几何审计与可视化', 'research/geometry_audit_v1.md')+' 为准。'
          '本次没有据此改 mask 或重算预测。视觉编码器此前的上下文混合也意味着目标位置 token 不等于纯目标语义。')

    prose('## 4. 完整 baseline：两套阈值均报告')
    tab(['配置', '有效/1213', 'MAE', '全体 MAE 界', '严格 all / suc / fail', '宽 all / suc / fail'], [
        [n, summary(n, pop='full')['n'], num(summary(n, pop='full').get('mae')),
         '–'.join(num(x) for x in summary(n, pop='full')['mae_all_expected_bounds']),
         acc(summary(n, pop='full')), acc(summary(n, pop='full'), THRESHOLDS[1])] for n in names])
    prose('Robometer 的非官方 image_text、video_text 明显退化：两套端点准确率均为 0。'
          'video_text 的全部 1213 条输出均落奖励档 3；image_text 为 496 条档 3、717 条档 4。'
          'text_video 虽有更低的 MAE，其严格 suc/fail 端点准确率仍很低。'
          '这说明顺序影响读出与校准，不能仅据 MAE 排名声称指令判断可靠。')
    prose('SOLE 官方输入的全量 MAE 为 1.4295，严格 all/suc/fail 为 42.46%/23.34%/52.11%，'
          '宽阈值为 54.16%/26.78%/67.99%；在本轮六个 SOLE baseline 中，MAE 更低且两类端点准确率均更高。'
          '这支持保留模型特有的官方上下文进行评估；但官方拼图、分辨率、七步推理和自身前驱同时改变，'
          '不能把差异单独归因于输入顺序，也不等于在数据外普遍优于其它模型。')
    prose('### Robometer 次要 success head')
    tab(['配置', 'success head MAE', '严格 all / suc / fail', '宽 all / suc / fail'], [
        [n, num(data[n]['secondary_success_head']['baseline']['full']['mae']),
         acc(data[n]['secondary_success_head']['baseline']['full']),
         acc(data[n]['secondary_success_head']['baseline']['full'], THRESHOLDS[1])]
        for n in names if n.startswith('meter_')])
    prose('success 与 progress 回答不同问题，故次要头可能呈现不同的端点准确率。'
          '这里将 success 概率沿用同一分档和两套阈值作辅助描述，并非官方 success 指标复现。'
          '全部干预的 success 统计另列各 exp_record 和分析 JSON，不据此事后替换主指标。'
          'Robometer 论文 E-5 的失败检测还组合 success 与进度曲线时间相关性，不能把本轮最终 progress 阈值结果当成论文 E-5 复现。')

    prose('## 5. 全部 target：不隐去 k、时域或缺失比较')
    for pop in ['cohort', 'holdout']:
        prose(f'### {pop} 的有效输出 MAE')
        rows = []
        for n in names:
            row = [n, num(summary(n, pop=pop).get('mae'))]
            for c in CONDITIONS:
                d = summary(n, c, pop)
                row.append(num(d.get('mae')) + ('†' if not d['paired_change'].get('complete') else ''))
            rows.append(row)
        tab(['配置', '同组 baseline', 'last k8', 'last k32', 'last k64', 'all k8', 'all k32', 'all k64'], rows)
    prose('† 表示 baseline/target 配对覆盖不完整，该有效输出 MAE 不能与不同有效样本集直接作总体因果比较。'
          '所有配对 ΔMAE、95% CI、有效/期望数、固定分母双类准确率和 Holm p 见 '+
          link('全矩阵数值表', 'analysis_v1/full_tables.md')+'；缺失误差界和共同样本比较见对应 JSON。')
    prose('### 两模型官方输入的全部六个 target')
    rows = []
    for n in ['meter_official', 'sole_official']:
        for c in CONDITIONS:
            d = summary(n, c)
            change = d['paired_change']
            rows.append([n, c, f"{d['n']}/{d['expected']}", num(change['mae_delta']),
                         '[' + ', '.join(num(x) for x in change['mae_delta_ci95']) + ']',
                         num(holm['cohort']['adjusted_p'][f'{n}/{c}']), acc(d), acc(d, THRESHOLDS[1])])
    tab(['配置', 'target', '有效/期望', '配对 ΔMAE', '95% CI', 'Holm p', '严格 all / suc / fail', '宽 all / suc / fail'], rows)
    prose('### 官方输入的区域/head 控制（四条件共同有效样本）')
    rows = []
    for n in ['meter_official', 'sole_official']:
        for c in CONDITIONS:
            scope, _, k = c.split(':')
            m = data[n]['matched_controls'][f'{scope}:{k}']['cohort']
            keys = ['baseline'] + [f'{scope}:{kind}:{k}' for kind in ['target', 'wrong_region', 'low_rank']]
            rows.append([n, f'{scope}/{k}', m['common_n']] + [num(m['conditions'][key].get('mae')) for key in keys])
    tab(['配置', '范围/k', '共同 n', 'baseline', 'target', 'wrong-region', 'low-rank'], rows)
    prose('控制只匹配 token 数/时域或低排名 head，没有完全匹配语义对象、形状或层分布。'
          'target 优于控制支持当前计算设置下的选择性，不能证明收益全部来自正确目标语义；'
          '控制本身改善也不应被隐去。全部普通输入控制见完整数值表。')
    prose('例如 SOLE official/all_frames/k64 的 target 与 wrong-region 均有完整的同一 846 条有效输出：'
          'target MAE 为 1.3298，wrong-region 为 1.2742；baseline 为 1.3121。'
          '正确区域 target 并未优于该空间控制，这限制了将该设置解释为稳定增强正确目标证据的主张。')
    prose('### 解释用例：误差、阈值与控制回答不同问题')
    examples = [('meter_text_image', 'all_frames:target:8'),
                ('meter_official', 'all_frames:target:32'),
                ('sole_official', 'last_frame:target:32'),
                ('sole_official', 'last_frame:target:64'),
                ('sole_official', 'all_frames:target:8'),
                ('sole_text_video', 'last_frame:target:64'),
                ('sole_text_video', 'all_frames:target:32')]
    tab(['配置 / target', '有效输出 MAE：baseline→target', '严格 all/suc/fail：baseline→target',
         '宽 all/suc/fail：baseline→target'], [
        [n+' / '+c, num(summary(n)['mae'])+' → '+num(summary(n,c)['mae']),
         acc(summary(n))+' → '+acc(summary(n,c)),
         acc(summary(n),THRESHOLDS[1])+' → '+acc(summary(n,c),THRESHOLDS[1])] for n,c in examples])
    prose('这些用例用于解释完整矩阵，属于观察后选例，并非独立确认的最优配置。'
          'Robometer text_image/all_frames/k8 是本轮 36 个 Robometer target 中观察到的最低 cohort MAE：'
          '1.8889→0.9823；严格 suc 83.58%→67.16%，fail 3.11%→24.57%，'
          '宽阈值总准确率虽达 70.09%，suc 仍由 92.54% 降至 71.64%。'
          'Robometer official/all_frames/k32 则连严格总准确率也下降，说明降低 MAE 不保证改善端点决策。')
    prose('两模型官方输入的全部六个条件亦画为 '+
          link('双类端点准确率变化图', 'analysis_v1/official_endpoint_tradeoffs_v2.png')+'（'+
          link('SVG', 'analysis_v1/official_endpoint_tradeoffs_v2.svg')+'）：横轴为 fail 准确率变化，纵轴为 suc 变化，'
          '只有右上象限表示两类都严格提高。该图是固定分母的描述性点估计，不表示准确率变化通过显著性检验。')
    prose('SOLE official/last_frame/k32 的严格 suc/fail 同时提高，但宽阈值 suc 下降；'
          'k64 在宽阈值 suc 与 baseline 相等。official/all_frames/k8 的严格双类准确率都提高，'
          'MAE 却上升。因此不能把“低 MAE”和“更好的双类端点判断”互相替代，阈值选择也会改变判断。')
    n='sole_text_video';c='last_frame:target:64'
    b=summary(n);d=summary(n,c)
    db=digest['experiments'][n]['populations']['cohort']['targets'][c]
    prose('SOLE 普通 text_video/last_frame/k64 的有效输出 MAE 为 '+num(b['mae'])+'→'+num(d['mae'])+
          '，全体 MAE 界为 ['+', '.join(num(x) for x in b['mae_all_expected_bounds'])+']→['+
          ', '.join(num(x) for x in d['mae_all_expected_bounds'])+']。'
          'target 上界仍低于 baseline 下界，故误差改善并非只能依靠删掉无效输出才能成立。'
          '但其严格 suc 准确率仅 4.48%，且 baseline/target 配对覆盖不完整；Holm p=1 是保守缺失规则，'
          '不能解释为证明没有效果。该协议的 last-frame target 在共同样本上优于两控制，'
          'all-frames/k32 的 wrong-region 反而比 target MAE 更低，选择性随时域改变。')
    assert db['lower_mae_for_all_missing_error_assignments']
    prose('### 全部协议的 task 异质性')
    rows = []
    for n in names:
        for scope in ['last_frame', 'all_frames']:
            counts = []
            for k in [8, 32, 64]:
                d = digest['experiments'][n]['populations']['cohort']['targets'][f'{scope}:target:{k}']
                dirs = d['complete_task_mae_directions']
                counts.append('/'.join(str(dirs.get(v, 0)) for v in ['lower', 'equal', 'higher']) + f" ({d['complete_task_count']}/28)")
            rows.append([n, scope] + counts)
    tab(['配置', '时域', 'k8：task MAE 降/平/升（完整 task 数）', 'k32', 'k64'], rows)
    prose('task 方向为描述性结果，既不代表每条样本改善，也不代表两个类别都改善。'
          '不完整 task 没有强行归入方向统计，其固定分母、无效数和准确率仍完整保留。'
          '逐 task 的 full/cohort/holdout 指标和两种分布分别见 '+
          link('13,176 行 task 指标', 'analysis_v1/task_metrics_all_populations.csv')+' 与 '+
          link('79,056 行 task 分布', 'analysis_v1/task_distributions_all_populations.csv')+'。')

    prose('## 6. 预测分布与同视频指令区分')
    prose('下表是 full baseline 的奖励档分布，顺序为 `[1,2,3,4,5]`；无效输出不被填入任何档。'
          '干预的全部 suc/fail/task 分布与原始连续差分档保存在分析 JSON 和 task CSV。')
    tab(['配置', 'suc 档 1–5 计数', 'fail 档 1–5 计数'], [
        [n] + [', '.join(str(summary(n, pop='full')['ordinal_prediction_distributions'][s]['counts'][str(k)]) for k in range(1, 6))
               for s in ['suc', 'fail']] for n in names])
    prose('SOLE official 全量 baseline 仍将 176/407 条 suc 放入最低奖励档，'
          '而 Robometer official 将 430/806 条 fail 放入最高奖励档；两者的主要偏差不同。'
          '下面给出前述解释用例在完整 cohort 上的分布变化，全部条件的分布仍完整保存在分析文件。')
    rows=[]
    for n,c in [('meter_text_image','all_frames:target:8'),('meter_official','all_frames:target:32'),
                ('sole_official','last_frame:target:32')]:
        for split in ['suc','fail']:
            values=[]
            for condition in ['baseline',c]:
                dist=summary(n,condition)['ordinal_prediction_distributions'][split]
                values.append(', '.join(str(dist['counts'][str(k)]) for k in range(1,6)))
            rows.append([n+' / '+c,split]+values)
    tab(['解释用例（cohort）','类别','baseline 奖励档 1–5','target 奖励档 1–5'],rows)
    rows = []
    for n in names:
        p = summary(n)['pairwise']
        rows.append([n, p['n'], p['unique_suc_videos'], num(p['continuous_mean_delta']), pct(p['continuous_negative_rate']),
                     ', '.join(str(p['ordinal_difference_counts'][k]) for k in ['<0', '0', '1', '2', '3', '4'])])
    tab(['cohort baseline', '有效对', '来源 suc 视频', '平均连续差', '连续负差率', '离散差 <0,0,1,2,3,4 计数'], rows)
    prose('### 官方 target 的共同有效配对变化')
    rows = []
    for n in ['meter_official', 'sole_official']:
        for c in CONDITIONS:
            d = digest['experiments'][n]['populations']['cohort']['targets'][c]
            p = d['common_pair_change']
            rows.append([n, c, p['common_pairs'], num(p['pair_weighted_mean_delta_change']), num(p['equal_video_mean_delta_change']),
                         pct(d['pairwise']['continuous_negative_rate']),
                         ', '.join(str(d['pairwise']['ordinal_difference_counts'][k]) for k in ['<0', '0', '1', '2', '3', '4'])])
    tab(['配置', 'target', '共同对', '平均连续差的变化：配对等权', '视频等权', 'target 连续负差率', 'target 离散差六档计数'], rows)
    prose('配对等权会使拥有更多 fail 指令的视频权重更高，因此补充视频等权；二者都在相同有效配对 ID 上作差。'
          '幅度增加与负差率降低是不同指标，不能替代端点准确率。跨模型比较时，SOLE 的负进度使连续差的范围也不同。')
    prose('Robometer text_image/all_frames/k8 的 543 对平均连续差从 0.2540 增至 0.5071，'
          '连续负差率从 20.07% 降至 13.81%，差≥.5 的比例从 25.05% 增至 66.85%。'
          '但 Robometer official/all_frames/k32 虽将平均连续差从 0.3312 提高到 0.4701，'
          '离散负差数却由 10 增至 29；量化后的平局/反序与连续均值也不能互相替代。'
          'SOLE official/last_frame/k32 的均值仅从 0.2468 增至 0.2704，连续负差率从 37.75% 降至 30.20%，'
          '仍有大量反序对。这些是配对分布描述，没有另作配对排序指标的显著性声明。')

    prose('## 7. 具体 top-8 与 top-8/32/64 重合度')
    tab(['配置/时域', '具体 top-8（零基 L/H）'], [
        [f'{n}/{scope}', ', '.join(f'L{l}H{h}' for l, h in rankings[f'{n}/{scope}']['pairs'][:8])]
        for n in names for scope in ['last_frame', 'all_frames']])
    rows = []
    for protocol in PROTOCOLS:
        for scope in ['last_frame', 'all_frames']:
            a = rankings[f'meter_{protocol}/{scope}']['pairs']
            b = rankings[f'sole_{protocol}/{scope}']['pairs']
            rows.append([f'{protocol}/{scope}'] + [f'{len(set(map(tuple,a[:k])) & set(map(tuple,b[:k])))}/{k}' for k in [8, 32, 64]])
    tab(['Robometer 与 SOLE 同名协议/时域', 'top-8 交集', 'top-32 交集', 'top-64 交集'], rows)
    prose('完整导出有 54 个 ranking 集、2,988 条重合度记录，包含新增 24 集与历史 Qwen/RoboReward/GRM 及 GRM raw-mass 重排。'
          'GRM 历史发布顺序以 excess mass 优先，额外 `/reranked_raw` 仅从历史值只读重排，没有改写历史结果。'
          '具体历史 top-8、全部交集比例和 Jaccard 见 '+link('ranking 汇总', 'head_overlap_v1/ranking_overlap.md')+'、'+
          link('overlap.csv', 'head_overlap_v1/overlap.csv')+'。跨权重和协议的 L/H 编号重合仅作描述，不等价于同一功能 head，也未验证直接复用旧 head 的效果。')

    prose('## 8. 完整性、无效输出与复现边界')
    rows = []
    for n in names:
        records = audit['experiments'][n]
        total = sum(d['observed'] for d in records.values())
        good = sum(d['statuses'].get('ok', 0) for d in records.values())
        rows.append([n, total, good, sum(d['format_failure_records'] for d in records.values()),
                     sum(d['infeasible_control_records'] for d in records.values()),
                     sum(d['other_invalid_records'] for d in records.values()),
                     sum(d['negative_progress'] for d in records.values()),
                     sum(d['percentage_clipped'] for d in records.values()),
                     sum(d['generation_at_token_cap'] for d in records.values())])
    tab(['配置：baseline+18 干预', '终点尝试', '有效', '格式失败', '不可构造控制', '其它无效', '负进度', '越界裁剪', '终点生成达 token cap'], rows)
    checks = audit['evidence_checks']
    prose('审计 `complete=true`，无缺失/额外 ID 和未处理运行错误。SOLE official 共核对 '+
          f"**{checks['official_step_records_checked']:,} 个实际步骤**，覆盖同条件前驱、progress_curve、末步 prompt 哈希及实际生成步骤的 attention hooks。"+
          '无效条件仍保留终点记录；本轮 SOLE 官方终点全部有效且完成七步。token cap 计数仅描述终点实际生成，不能单凭达上限判断答案是否无效。')
    prose('格式解析失败保留原文，不能从损坏标签中猜数，也不补为 0/reward=1；wrong-region 不可构造时只影响该控制。'
          'SOLE 合法数字若超出 [−100,100]% 则按既定解析规则裁剪，原百分比与标志仍保存；'
          '普通输入中发现的 3 条为 117%、133%、133%，不能把它们描述为模型天然满足数值范围。'
          '不完整比较的缺失 [0,4] 误差界及对所有缺失赋值仍降低 MAE 的描述性判定见 '+
          link('完整解释 JSON', 'interpretation_v1.json')+'。四条件共同有效比较避免分母错配，但仍只代表该有效子集。')
    prose('最终有 15 条终点记录发生范围裁剪，其中 12 条来自 SOLE official。'
          '另对官方全部 115,087 步作数值检查：13 步发生裁剪（12 个终点、1 个中间步骤），'
          '越界原值为 101%、102%、105%、110% 或 171%；有 2 个负进度步骤，无生成达 token cap 的步骤。'
          '中间步骤裁剪后的值会进入该条件自身递推上下文；本轮没有重跑无裁剪版本。详见 '+
          link('官方全步骤数值审计', 'research/official_numeric_audit_v1.json')+'。')
    prose('13 个契约检查通过；Robometer 两条样本的 progress/success 与官方纯源码 readout 最大误差均为 0，'
          'ranking 观测前后输出误差也为 0。这是局部 readout 验证，**不是整矩阵二次推理复现**。'
          '全部 baseline 的新增奖励档分布复算确认原 MAE/准确率/配对统计未变（浮点容差 1e-12），该检查同样只验证统计。')
    prose('运行环境为 robo-dopamine，Python 3.10.20、PyTorch 2.8.0/CUDA build 12.8、Transformers 4.57.0；'
          '仅使用用户授权的 GPU 0、1、2。权重/processor 的 31 个文件指纹、实际 batch 与调度事件均保留。'
          '早期 meta buffer、不可行控制阻断和官方 ranking 前驱处理修正及本轮进程重启已留痕；'
          'raw 结果只追加，按每条件/ID 最新记录评分，pilot 独立保存。')
    prose('操作审计：首次读取计划前误执行过一次只读 `git status --short`，未执行版本写入；'
          '读到禁令后停止已有仓库 Git 操作，随后仅使用计划允许的新官方开源仓库 clone。'
          '未参考其它已有本地代码仓库；原始数据与历史结果没有删除或覆盖。')

    prose('## 9. 统计谬误检查：11/11')
    tab(['检查', '结论/限制'], [
        ['1 Simpson 悖论', '已比较总体、suc/fail 与逐 task；存在类别权衡和 task 异质性，不能把总体 MAE 下降称为各层都改善，也不把局部反向直接命名为严格 Simpson 悖论'],
        ['2 生态谬误', 'task 均值不推断每条样本；保留逐例记录和同视频配对'],
        ['3 Berkson 选择偏差', 'grounding cohort 为筛选子集；full baseline 与 cohort 分开，未据此外推全部 34 个 task'],
        ['4 Collider 偏差', '无基于模型分数的事后筛选；grounding 和有效输出选择仍可能限制推广，共同有效比较不能消除此风险'],
        ['5 基率忽视', '报告 suc/fail 固定分母和两套阈值，给出始终失败参考；总准确率不等价于平衡判断'],
        ['6 均值回归', '未按极差预测选择 episode；所有 k 完整披露，观察后最低 MAE 不当作独立确认，holdout 与 cohort 重叠'],
        ['7 生存者偏差', '格式失败/不可行控制公开；固定分母准确率、MAE 误差界、完整性门槛与共同 ID 比较并行报告'],
        ['8 多处寻找效应', '每 population 固定 72 项 Holm，未完整项 p=1；没有为探索性控制/阈值/task 结果冒称同一校正的显著性'],
        ['9 分析岔路', '参数、输入、发现集冻结，修复留痕；后加分布和解释导出明确为描述性，无正式预注册声明'],
        ['10 相关与因果', '固定视频上的 attention 计算干预可改变该模型输出；不推出机器人闭环成功率或训练机制因果归因'],
        ['11 反向因果', 'head raw mass 排名与结果关联不证明训练形成机制；跨模型编号相交不作因果或功能身份解释']])
    prose('总体解释等级为 **CAUTION**：已完成数值和完整性审计，但探索性设置、同域筛选、缺失输出与协议混杂仍限制结论。'
          '没有用“未显著”证明无效，也没有把显著的微小 MAE 变化解释成实际可用性。')

    prose('## 10. 机制解释与下一步边界')
    prose('对某个可见 query，原始注意力质量为目标 mT、受负 bias 域 mN、其它 mO，则施加 b 后：'
          '`mT′=exp(b)mT/[exp(b)mT+exp(−b)mN+mO]`。b=6 将目标对负域的相对 odds 乘以 exp(12)≈162755，'
          '对其它可见域乘以 exp(6)≈403；causal 禁止 key 仍不可见。'
          '这说明相同数值 bias 并非跨模型等效剂量，目标面积、原 mass、层位置和最终读出均改变作用。')
    prose('Robometer 通过专门训练的 prog_token 和分类头读进度；SOLE 在最后 prompt token 排名后还生成 reasoning/answer，'
          '官方输入又包含逐步自反馈。普通图像先于指令时，较早视觉 token 在因果语言层不能读取后续指令；'
          '这些差异提供了协议敏感性的合理解释，但本轮没有用独立消融识别其相对贡献。'
          'ViT 混合、tubelet 并集和拼图边界也限制“纯目标语义增强”的机制主张。')
    prose('若继续独立研究，优先在新视频组/新任务上固定候选协议与 k，同时检查双类准确率、共同控制和配对；'
          '再单独消融 bias 强度、ranking query、层分布与官方递推，避免同时改变多个因素。'
          '这些是后续假设，未算作本轮已执行的实验；本轮计划要求的矩阵与五类指标已完成。')

    prose('## 11. 证据与复查入口')
    tab(['材料', '位置'], [
        ['完整性与终止状态', link('completion audit', 'completion_audit_v1.json')+'；'+link('scheduler completion', 'research/scheduler_completion.json')],
        ['完整数值与统计', link('full_tables.md', 'analysis_v1/full_tables.md')+'；'+link('analysis index', 'analysis_v1/index.json')+'；'+link('Holm', 'analysis_v1/holm.json')],
        ['全矩阵科学图', link('ΔMAE PNG', 'analysis_v1/paired_mae_changes.png')+'；'+link('可导出 SVG', 'analysis_v1/paired_mae_changes.svg')],
        ['逐 task 全样本集', link('metrics CSV', 'analysis_v1/task_metrics_all_populations.csv')+'；'+link('distributions CSV', 'analysis_v1/task_distributions_all_populations.csv')],
        ['契约与局部官方一致性', link('13 项检查', 'research/contracts_v9.log')+'；'+link('官方 readout parity', 'research/official_readout_parity.json')],
        ['统计定义与几何', link('分布复算审计', 'research/distribution_definition_audit_v1.json')+'；'+link('几何审计', 'research/geometry_audit_v1.md')],
        ['环境与来源', link('环境', 'research/runtime_environment_v2.json')+'；'+link('模型指纹', 'research/model_fingerprints.json')+'；'+link('输入/配置/官方源码指纹', 'research/source_provenance_v1.json')+'；'+link('文献和估计目标', 'research/literature_and_estimands_v2.md')]])
    tab(['配置', '实验记录'], [[n, link('exp_record.md', f'{n}/exp_record.md')] for n in names])
    prose('一手来源：[Robometer 论文](https://arxiv.org/html/2603.02115)、'
          '[官方 Robometer 实现](addbase_eval/references/robometer/robometer/models/rbm.py)、'
          '[SOLE-R1 论文](https://arxiv.org/html/2603.28730v2)、'
          '[官方 RewardGen SOLE 实现](addbase_eval/references/rewardgen/rewardgen/sole.py)。'
          '历史背景为 [GRM 总结](exp_plan_GRM_summary.md) 和 [跨模型总结](exp_plan_crossmodel_summary.md)；'
          '本轮没有重跑这些历史模型，也没有将不同协议的结果混称同一受控比较。')
    create_text(OUT / 'research' / args.output_name, '\n'.join(lines))
    print(OUT / 'research' / args.output_name)


if __name__ == '__main__':
    main()
