"""Add reviewed scientific interpretation to the existing axis-value drafts.

Formatting/edit provenance only: a separate actual bilingual review is required.
No model, data, scoring, training or automatic acceptance is performed.
"""
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/foodnutrigpt_v9_research/report_snapshot_r9_axisvalue_v1'
VERSION = ROOT / 'experiments/foodnutrigpt_v9_research/r9/axisvalue_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def paragraph(text, prefix, replacement):
    parts = text.split('\n\n')
    matches = [i for i, part in enumerate(parts) if part.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError('Expected one paragraph: ' + prefix)
    parts[matches[0]] = replacement
    return '\n\n'.join(parts)


INTERPRETATION = {
    'ZH': {
        'scope': '版本范围：R0–R9完整阶段报告，含MAE三种子复验、MSE及轴×数值残差的单因素筛选。两个新增配方均未通过各自筛选，保留原MAE作为控制。第2–10节保留此前已审阅的历史证据，第11节记录最新完整版本；Transformer优于冻结RF的整体目标仍未达到。',
        'lead': '最新的轴×数值残差配方完成60轮、独立重放和11项比较。补全主误差0.187792，相对同种子MAE点退步2.309%，改善区间[−5.099%,+0.263%]；旧log-MAE点退步3.063%。三个预登记筛选条件均未满足，拒绝此固定配方作为补全升级，不追加其23/24种子。相对冻结RF虽有0.655%的点改善，但区间[−1.888%,+3.017%]跨零且仅有一个种子，不能宣称稳定优于RF。完整结果和归因见第11节。',
        'headline': '## 11. 营养轴×数值残差：筛选完成，拒绝此固定升级配方',
        'diagnosis': '共同分母分解精确重构主误差增加0.004238696：正值低估增加0.004090529，正值高估增加0.000921943，零值收益为−0.000773776。六面板曲线已实际查看，两个模型均选第60轮；本轮末10轮裁剪比例0.344902，低于控制0.350883，未支持“裁剪更频繁导致退步”的解释。全量1,828,536个训练目标的无来源推理主误差也从0.134937略增至0.135435；不能据此单独区分优化、容量或泛化机制。条件正值/零值宏平均有各自分母，即使两项条件均值较低，也不保证总体宏平均较低。',
        'axes': '相对同种子MAE，60/142轴点改善，14个逐轴区间支持改善、20个支持退步（均未作多重比较校正）；9轴验证支持不足30个食品组。7个训练支持少于100组的轴贡献+0.004304441，20个100–999组轴贡献+0.000573374，115个至少1000组轴贡献−0.000639120。稀疏轴中的Lignin和Isomeric linolenic acids贡献较大，支持仅17个验证食品组；后者区间跨零，不能把稀疏轴总体视为已确定的因果瓶颈。脂肪酸家族贡献+0.002138810、膳食纤维+0.001829756；FooDB来源分区贡献+0.004233540。来源、家族、支持度是同一误差的重叠分解，不能相加。',
        'cases': '已从保存预测重建并核对40个RF对比极端案例。20个退步案例均来自FooDB，其中19个为正值Cholesterol，预测至少比保留标签低500倍；这不证明单位错误，也不证明这些标签正确。20个改善案例含14个正值、6个零值，覆盖5个来源、13个食品候选组。极端案例并不代表总体分布或40个独立样本，不能据此修改标签或排除数据。逐条食物名称、浓度与预测仍只存本地数据目录。',
        'aux': '补全的142轴正值条件误差从0.302773增至0.306668，零值误差从0.092039降至0.085240。仅名称142轴主误差点改善1.623%，区间[−0.921%,+4.038%]跨零；其45轴和187轴误差反而变差。全可见/30%可见的R@10分别为0.008550/0.004341，均低于MAE；30%可见R@10差值区间为[−0.004658,−0.000073]，其余7个同种子检索指标差值区间均跨零。这些条件性区间未校正多指标选择，不能由某一辅助指标替代补全筛选。',
        'causal': '观察事实是：此固定干预在单种子下没有满足改善条件，零值收益被正值代价抵消，训练主误差也略高。对照支持拒绝当前完整配方，不能证明所有轴专属编码均无效。新增48,384参数、交互形式和优化轨迹一同改变，无法唯一归因；未验证另一学习率、正则或等容量控制。条件区间不包括训练种子总体、反复选型、标签有效性或外部迁移不确定性。数据、尺度、指标、门槛均未随结果修改。',
        'decision': '版本决定：拒绝当前轴×数值残差配方作为补全升级；保留全部产物，不追加其20260923/20260924种子，继续使用原MAE作为方法控制。区间跨零不证明等效或普遍有害；本决定遵循预先规定的接受条件。该结果不完成Transformer优于冻结RF的整体目标。详见[机器决定](../r9/axisvalue_v1/decision.json)。',
        'next': '下一项优先核验训练中的来源校准：在原MAE控制上单独比较source weight 1→0，保持按(1+w)归一化的损失尺度及其余训练设置，先验证零残差初始时共享参数梯度一致。该问题尚未登记或训练；已有来源偏置稳定性不证明其收益，当前来源分区差异也不构成校准的因果证据。后续长训练将根据既有约3小时耗时安排定时回访，停止逐epoch观察。数据和RF/XGB继续冻结，测试保持关闭，来源留出、少样本迁移及可靠foundation model能力仍未验证。',
    },
    'EN': {
        'scope': 'Version scope: completed R0–R9 stage report including three-seed MAE confirmation and single-factor MSE and axis-by-value screens. Both added recipes fail their registered screens; the original MAE model remains the control. Sections 2–10 retain previously reviewed historical evidence, while Section 11 records the latest complete version. The overall Transformer-over-frozen-RF objective remains unachieved.',
        'lead': 'The latest axis-by-value recipe completed 60 epochs, independent replay and 11 comparisons. Primary completion error is 0.187792: a 2.309% point regression against same-seed MAE, with an improvement interval of [−5.099%, +0.263%]; legacy log-MAE regresses by 3.063% at the point estimate. All three registered screening gates fail. This fixed recipe is rejected as a completion upgrade and will not receive seeds 23/24. Its 0.655% point gain over frozen RF has an interval of [−1.888%, +3.017%] and only one seed, so it does not establish stable superiority over RF. Section 11 gives the complete results and interpretation.',
        'headline': '## 11. Axis-by-value residual: screen complete, fixed upgrade recipe rejected',
        'diagnosis': 'The common-denominator decomposition reconstructs the primary increase of 0.004238696 exactly: positive underprediction adds 0.004090529, positive overprediction adds 0.000921943, and explicit-zero improvement contributes −0.000773776. All six curve panels were visually inspected; both models select epoch 60. The last-ten-epoch clipping fraction is 0.344902, below the control at 0.350883, which does not support increased clipping frequency as the explanation. Source-free inference over all 1,828,536 training targets also raises primary error slightly, from 0.134937 to 0.135435; this alone cannot distinguish optimization, capacity or generalization mechanisms. Conditional positive and zero macro averages have their own denominators, so lower values for both do not guarantee a lower overall macro average.',
        'axes': 'Against same-seed MAE, 60/142 axes improve at the point estimate, with 14 axis intervals supporting improvement and 20 supporting regression (without multiplicity correction); 9 axes have fewer than 30 validation food groups. The 7 axes with fewer than 100 training groups contribute +0.004304441, the 20 with 100–999 contribute +0.000573374, and the 115 with at least 1000 contribute −0.000639120. Lignin and Isomeric linolenic acids contribute large sparse-axis differences, each with only 17 validation groups; the latter interval crosses zero. This does not establish sparse-axis support as a causal bottleneck. Fatty acids contribute +0.002138810 and dietary fibre +0.001829756; the FooDB source partition contributes +0.004233540. Source, family and support partitions describe overlapping decompositions of the same error and must not be added together.',
        'cases': 'Forty extreme RF-comparison cases were reconstructed and checked against saved predictions. All 20 worsening cases come from FooDB; 19 are positive Cholesterol labels with predictions at least 500-fold below the retained label. This proves neither a unit error nor label correctness. The 20 improving cases contain 14 positive and 6 zero labels across 5 sources and 13 candidate food groups. These extremes are neither representative nor 40 independent samples, and do not justify relabeling or exclusion. Individual food names, concentrations and predictions remain in the local data directory.',
        'aux': 'For 142-axis completion, conditional positive error rises from 0.302773 to 0.306668 while zero error falls from 0.092039 to 0.085240. Name-only primary error improves by 1.623% at the point estimate, but its [−0.921%, +4.038%] interval crosses zero; its 45-axis and 187-axis errors worsen. Full/30%-visible R@10 are 0.008550/0.004341, both below MAE. The 30%-visible R@10 difference interval is [−0.004658, −0.000073]; the other seven same-seed retrieval intervals cross zero. These conditional intervals are not adjusted for multiple-metric selection, and an auxiliary metric cannot substitute for the completion screen.',
        'causal': 'Observed facts are that this fixed intervention fails the single-seed improvement gates, positive-label costs outweigh zero-label gains, and training primary error also rises slightly. The control supports rejecting this complete recipe, not a claim that all axis-specific encodings are ineffective. The additional 48,384 parameters, interaction form and optimization trajectory change together, preventing unique attribution. Another learning rate, regularization setting or matched-capacity control was not evaluated. Conditional intervals exclude seed-population, repeated-selection, label-validity and external-transfer uncertainty. Data, scales, metrics and gates were not changed in response to results.',
        'decision': 'Version decision: reject the current axis-by-value recipe as a completion upgrade, retain every artifact, do not add seeds 20260923/20260924, and continue using the original MAE model as the method control. Intervals crossing zero establish neither equivalence nor universal harm; this decision applies the preregistered acceptance rule. It does not complete the overall Transformer-over-frozen-RF objective. See the [machine-readable decision](../r9/axisvalue_v1/decision.json).',
        'next': 'The next priority is to test training-only source calibration: compare source weight 1→0 on the original MAE control, retaining loss normalization by (1+w) and all other training settings, and first verify matching shared-parameter gradients at zero residual initialization. This question is not yet registered or trained. Stable source offsets do not establish a benefit, and current source-partition differences do not establish a calibration mechanism. Future long runs will use estimated-duration follow-ups based on the existing approximately three-hour runtime, ending per-epoch observation. Data and RF/XGB remain frozen, test remains closed, and source-held-out transfer, few-shot transfer and a reliable foundation-model claim remain unverified.',
    },
}


def main():
    evidence = read(OUT / 'evidence.json')
    if (OUT / 'interpretation_edit.json').exists():
        raise FileExistsError('Interpretation already applied; inspect before further edits.')
    analysis = read(ROOT / 'reports/v9_r9_axisvalue_analysis_v1/summary.json')
    if any(analysis['screening_gates'].values()):
        raise ValueError('This fixed interpretation requires the observed failed screen.')
    for name, expected in evidence['document_hashes'].items():
        if sha(OUT / name) != expected:
            raise ValueError('Draft changed before interpretation: ' + name)
    final = {}
    for lang, words in INTERPRETATION.items():
        path = OUT / f'REPORT_{lang}.md'
        text = path.read_text(encoding='utf-8')
        text = paragraph(text, '草稿：' if lang == 'ZH' else 'DRAFT:', words['scope'])
        text = text.replace('本次新增的MSE单因素对照', '此前MSE单因素对照', 1) if lang == 'ZH' else text.replace('The added MSE contrast', 'The preceding MSE contrast', 1)
        marker = '## 2. '
        before, separator, after = text.partition(marker)
        if not separator:
            raise ValueError('Missing data section')
        text = before.rstrip() + '\n\n' + words['lead'] + '\n\n' + marker + after
        history, separator, section = text.partition('## 11. ')
        if not separator:
            raise ValueError('Missing new section')
        section = words['headline'] + '\n' + section.partition('\n')[2]
        section = paragraph(section, '低估、高估及零值三项' if lang == 'ZH' else 'The underprediction, overprediction and explicit-zero contributions', words['diagnosis'] + '\n\n' + words['axes'] + '\n\n' + words['cases'] + '\n\n' + words['aux'] + '\n\n[Visual review](../../../reports/v9_r9_axisvalue_analysis_v1/visual_review.json) · [Axis contrasts](../../../reports/v9_r9_axisvalue_axis_changes_v1/summary.json) · [Case review](../../../reports/v9_r9_axisvalue_cases_v1/summary.json)')
        section = paragraph(section, '本轮同时增加轴专属交互参数' if lang == 'ZH' else 'This intervention adds both', words['causal'])
        section = paragraph(section, '此处只呈现自动筛选结果' if lang == 'ZH' else 'This section presents the automatic screen only.', words['decision'])
        section = paragraph(section, '完整审阅后按登记规则决定' if lang == 'ZH' else 'After complete review, apply the registered rule', words['next'])
        commands = '\n\n```powershell\n.venv/Scripts/python.exe -X utf8 scripts/train_foodnutrigpt_v9_r9_axisvalue.py --candidate tf192_mae_axisvalue_lr3e4_60 --output-dir output/v9_r9_methods/tf192_mae_axisvalue_lr3e4_60\n.venv/Scripts/python.exe -X utf8 scripts/audit_foodnutrigpt_r9_axisvalue.py --run output/v9_r9_methods/tf192_mae_axisvalue_lr3e4_60 --output-dir reports/v9_r9_axisvalue_lr3e4_60_audit_v1\n.venv/Scripts/python.exe -X utf8 scripts/analyze_foodnutrigpt_r9_axisvalue.py --output-dir reports/v9_r9_axisvalue_analysis_v1\n.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_axisvalue_cases.py --output-dir reports/v9_r9_axisvalue_cases_v1\n.venv/Scripts/python.exe -X utf8 scripts/review_foodnutrigpt_r9_axisvalue_axes.py\n```\n\n'
        environment = ('环境：Windows、Python3.10.19、PyTorch2.7.1+cu128、RTX5070Ti16GB。上述命令已执行，已有目录拒绝覆盖；完整命令及依赖哈希见运行/分析清单。正式训练、重放及比较无失败；案例读取出现pandas混合类型提示，但已用精确值与元数据逐项核对通过。' if lang == 'ZH' else 'Environment: Windows, Python 3.10.19, PyTorch 2.7.1+cu128, RTX 5070 Ti 16GB. These commands have completed and existing directories reject overwrite; manifests retain complete commands and dependency hashes. Formal training, replay and comparisons had no failures. Case loading emitted a pandas mixed-type warning, but exact values and metadata passed the explicit checks.')
        section = section.replace('### 11.4.', commands + environment + '\n\n### 11.4.', 1)
        final[lang] = history + section
    version = final['ZH'].partition('## 11. ')[2]
    version = '# V9-R9 / ' + version
    version = re.sub(r'^### 11\.(\d)\.', r'## \1.', version, flags=re.M)
    def rebase(match):
        target = match[2]
        if re.match(r'^[a-z]+:|^#|^/', target):
            return match[0]
        path, separator, fragment = target.partition('#')
        target_path = (OUT / path).resolve()
        relative = Path(os.path.relpath(target_path, VERSION)).as_posix()
        return f'[{match[1]}]({relative}{separator}{fragment})'
    version = re.sub(r'\[([^\]\n]+)\]\(([^)\n]+)\)', rebase, version)
    version = version.replace('第6节', '综合报告第6节')
    version = version.replace('\n\n## 1.', '\n\n完整[中文报告](../../report_snapshot_r9_axisvalue_v1/REPORT_ZH.md)和[英文报告](../../report_snapshot_r9_axisvalue_v1/REPORT_EN.md)包含数据来源、此前迭代及冻结RF/XGB方法。\n\n## 1.', 1)
    for lang, content in final.items():
        (OUT / f'REPORT_{lang}.md').write_text(content, encoding='utf-8')
    (VERSION / 'README.md').write_text(version, encoding='utf-8')
    receipt = {'status': 'scientific_interpretation_added_actual_bilingual_review_required',
        'original_draft_hashes': evidence['document_hashes'],
        'document_hashes': {f'REPORT_{lang}.md': sha(OUT / f'REPORT_{lang}.md') for lang in final},
        'version_readme_sha256': sha(VERSION / 'README.md'), 'script_sha256': sha(Path(__file__)),
        'manual_bilingual_review_complete': False, 'goal_achieved': False,
        'training_performed': False, 'data_modified': False, 'baseline_refit': False, 'complete_test_opened': False}
    (OUT / 'interpretation_edit.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': receipt['status']}))


if __name__ == '__main__':
    main()
