"""Record actual reviewed capacity findings; final bilingual proofreading is separate."""
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/foodnutrigpt_v9_research/report_snapshot_r9_capacity256_v1'
VERSION = ROOT/'experiments/foodnutrigpt_v9_research/r9/capacity256_v1'
ANALYSIS = ROOT/'reports/v9_r9_capacity256_analysis_v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))

WORDS = {
  "ZH": {
    "scope": "版本范围：R0–R9完整阶段报告，包含MAE三种子复验，以及MSE、轴×数值残差、关闭来源校准和PDF容量组合的筛选。四个追加配方均未通过各自接受条件，当前方法控制仍为192维MAE、source weight=1。第2–12节保留此前已审阅的历史证据；第13节为最新完整容量实验，含运行中断与恢复。Transformer优于冻结RF的整体目标仍未达到。",
    "overview": "## 1. 研究目标与主要结论\n\n当前目标是在冻结数据和RF/XGBoost结果的条件下，通过有验证依据的单阶段Transformer方法改善营养补全，同时跟踪仅名称预测和营养到名称检索。每个版本保留假设、全部结果、失败和归因限制；不把不同模型的任务最佳成绩拼接成一个模型。\n\n原192维MAE的三种子补全主误差为0.189184 ± 0.006925，固定RF为0.189031；食品组改善区间跨零，旧log指标均值退步也超过2%保护条件，尚未获得稳定超越RF的证据。完整基线与历史方法见第2–12节。",
    "lead": "最新256维、8头、FF1024容量组合完成原定60轮，按验证主指标选中第59轮。补全主误差0.201520，较同种子192控制退步9.788%，改善区间[−12.943%,−6.495%]；旧log-MAE退步14.421%。相对冻结RF主误差退步6.607%，相对XGB退步15.493%。三个筛选条件均失败，拒绝这个固定容量升级配方，不追加其23/24种子。训练拟合也更差，不支持直接把失败解释为更大模型过拟合。详见第13节。",
    "title": "## 13. PDF容量组合：筛选完成，拒绝此固定升级配方",
    "diagnosis": "共同分母分解精确重构主误差增加0.017966898：正值低估+0.016194201、正值高估+0.002098269、显式零−0.000325572。主要代价来自正值低估。条件零值宏平均反而从0.092039升至0.094971；它与共同分母零贡献的方向不同，因为轴内分母和有效权重不同。不能把条件统计和主误差贡献混为一谈。对全部1,828,536个训练目标的无来源推理主误差由0.134937升至0.155628（差+0.020690469），条件正值与零值误差也都升高，未显示更大模型改善拟合。",
    "curves": "已实际查看六面板曲线。容量模型选第59轮，192控制选第60轮；容量模型的主误差、旧log和正值误差在后期仍更高，曲线趋缓但不证明全局收敛。末10轮裁剪比例为0.366907，对照为0.350883；平均梯度范数有波动，没有可见发散。较高裁剪率是描述性现象，参数量也不同，不能单独证明裁剪导致退步。第48轮中断及49–60轮恢复已单独审计，不把曲线平滑当作未中断轨迹等价的证明。",
    "axes": "13/142轴点改善，只有2个未校正逐轴区间支持改善，96个支持退步；9轴验证支持不足30个食品组。两个有改善区间的轴为Tocotrienol, gamma和Vitamin D2；20个氨基酸轴全部点退步。115个至少1000训练组的轴贡献+0.013257662，20个100–999组轴贡献+0.001768189，7个少于100组轴贡献+0.002941046，因此代价并非只集中于稀疏轴。脂肪酸、维生素、膳食纤维家族分别贡献+0.005717328、+0.004269019、+0.003190012。24个来源分区均点退步，FooDB贡献+0.006363348。最大单轴代价Lignin仅有17个验证组，区间宽；来源、家族与支持度分区不能相加，也不能据此认定唯一生物机制。",
    "cases": "已从冻结任务和保存预测重建并核对40个RF对比极端案例。20个退步案例均为FooDB正值，来自18个食品组，含15个Cholesterol和5个Biotin案例；其中15个Cholesterol预测均至少比保留标签低500倍。这既不确认单位错误，也不证明标签正确。20个改善案例含7个正值、13个零值，覆盖5个来源和18个食品组。尾部案例不是代表性独立样本，不据此更改数据；逐条名称、浓度和预测保留在本地。",
    "aux": "仅名称主误差0.432889，较同种子MAE点退步1.519%，改善区间[−4.296%,+1.364%]跨零；其旧log-MAE点改善4.593%，相应区间仍跨零。相对冻结RF/XGB，仅名称预测较好，但明显弱于名称KNN的0.262260。全可见/30%可见检索R@10为0.008510/0.004516。两种可见度下R@1略有点改善，MRR/R@5/R@10均点退步，8个差值区间全部跨零；不能宣称检索升级。45轴和187轴补全也点退步。全部三任务使用同一个第59轮检查点。",
    "causal": "已观察事实是：在这组固定训练设置下，容量组合增加后训练与验证主误差都更高，退步覆盖大多数营养轴和所有来源分区。对照支持拒绝直接替换为此256/8/FF1024配方，不能推出更大Transformer普遍无益。宽度、头数、前馈规模、参数量、初值形状和随机数消耗共同改变，相同种子不是相同初始化。训练拟合也变差，与单纯“训练更好、验证更差”的过拟合叙述不一致；优化设置与容量的相互作用仍是未验证假设。中断前48轮已保存，恢复状态和数值循环通过核验，但未中断GPU反事实不可获得。单种子食品组区间不覆盖种子总体、重复选型、标签有效性或外部迁移。",
    "decision": "版本决定：拒绝此固定容量配方作为补全升级，不追加20260923/20260924复验，保留原48轮、中断记录、恢复运行及全部结果。继续以192维MAE、source weight=1为方法控制；保留控制不等于证明其参数最优。整体RF目标尚未达到。[机器决定](../r9/capacity256_v1/decision.json)记录筛选依据。",
    "next": "下一轮回到现有192维控制，优先检验初始学习率3e-4→6e-4的优化假设，其余结构、损失、来源权重、采样和60轮日程保持不变。此前学习率筛选及本轮训练拟合退步说明应继续验证优化设置，不能仅靠扩容；这不预设更大学习率会获胜。该具体候选尚未登记或启动，必须另行登记、核验并先估时再回访。数据和RF/XGB继续冻结，测试关闭，来源留出与少样本迁移能力仍未验证。"
  },
  "EN": {
    "scope": "Version scope: completed R0–R9 stage report, including three-seed MAE confirmation and screens of MSE, an axis-by-value residual, source-calibration removal and the PDF capacity combination. All four added recipes fail their respective acceptance conditions; the method control remains width-192 MAE with source weight=1. Sections 2–12 preserve previously reviewed history. Section 13 is the latest complete capacity experiment, including interruption and recovery. The overall Transformer-over-frozen-RF objective remains unachieved.",
    "overview": "## 1. Research objective and principal findings\n\nThe current objective is to improve nutrition completion through validated single-stage Transformer methods while keeping data and previously trained RF/XGBoost results frozen. Name-only prediction and nutrition-to-name retrieval remain tracked tasks. Every version retains hypotheses, all results, failures and attribution limits; different models’ best task scores are not combined into one model.\n\nThe original width-192 MAE model has three-seed primary completion error of 0.189184 ± 0.006925, against 0.189031 for fixed RF. Its food-group improvement interval crosses zero, and mean legacy-log regression exceeds the 2% guard. Stable superiority over RF is therefore unestablished. Sections 2–12 provide complete baselines and historical methods.",
    "lead": "The latest width-256, eight-head, FF1024 capacity combination completed the registered 60 epochs and selected epoch 59 by validation primary error. Completion error is 0.201520, a 9.788% regression against the same-seed width-192 control, with an improvement interval of [−12.943%,−6.495%]; legacy log-MAE regresses by 14.421%. Primary error is 6.607% worse than frozen RF and 15.493% worse than XGBoost. All three screening gates fail, so this fixed capacity upgrade is rejected without seeds 23/24. Training fit is also worse, which does not support simply attributing failure to a larger model overfitting. Section 13 gives the evidence.",
    "title": "## 13. PDF capacity combination: screen complete, fixed upgrade recipe rejected",
    "diagnosis": "The common-denominator decomposition reconstructs the primary increase of 0.017966898: positive underprediction contributes +0.016194201, positive overprediction +0.002098269, and explicit zeros −0.000325572. Positive underprediction accounts for most of the cost. Conditional zero macro error instead rises from 0.092039 to 0.094971; its direction differs from the common-denominator zero contribution because within-axis denominators and effective weights differ. Conditional statistics and primary contributions must not be conflated. Source-free inference on all 1,828,536 training targets increases primary error from 0.134937 to 0.155628 (difference +0.020690469). Conditional positive and zero training errors also increase, providing no evidence of improved fitting from greater capacity.",
    "curves": "All six curve panels were visually inspected. The capacity model selects epoch 59 and the width-192 control selects epoch 60. Primary, legacy-log and positive errors remain higher for the larger model late in training; flattening does not establish global convergence. Last-ten-epoch clipping is 0.366907 versus 0.350883. Mean gradient norms fluctuate without visible divergence. Higher clipping frequency is descriptive, and parameter counts differ; it does not by itself establish clipping as the cause. The epoch-48 interruption and recovery through epochs 49–60 were audited separately. Smooth curves are not proof of equivalence to an uninterrupted trajectory.",
    "axes": "Only 13/142 axes improve at the point estimate; two unadjusted axis intervals support improvement and 96 support regression. Nine axes have fewer than 30 validation food groups. Tocotrienol, gamma and Vitamin D2 are the two axes with improvement intervals; all 20 amino-acid axes worsen at the point estimate. The 115 axes with at least 1,000 training groups contribute +0.013257662, the 20 with 100–999 contribute +0.001768189, and the seven below 100 contribute +0.002941046. Costs therefore are not confined to sparse axes. Fatty-acid, vitamin and dietary-fibre families contribute +0.005717328, +0.004269019 and +0.003190012. All 24 source partitions worsen at the point estimate; FooDB contributes +0.006363348. Lignin has the largest single-axis cost but only 17 validation groups and a wide interval. Source, family and support partitions overlap and do not identify a unique biological mechanism.",
    "cases": "Forty extreme RF-comparison cases were reconstructed and checked against frozen tasks and saved predictions. All 20 worsening cases are positive FooDB labels across 18 food groups: 15 Cholesterol and five Biotin cases. Every one of those 15 Cholesterol predictions is at least 500-fold below its retained label, confirming neither unit error nor label correctness. The 20 improving cases contain seven positive and 13 zero labels across five sources and 18 food groups. Extreme tails are not representative independent samples and do not justify changing data. Individual names, concentrations and predictions remain local.",
    "aux": "Name-only primary error is 0.432889, a 1.519% point regression against same-seed MAE, with an improvement interval of [−4.296%,+1.364%] crossing zero. Its legacy log-MAE improves by 4.593% at the point estimate, but that interval also crosses zero. Name-only prediction remains better than frozen RF/XGB and substantially worse than name kNN at 0.262260. Full/30%-visible retrieval R@10 is 0.008510/0.004516. R@1 improves slightly at both visibility levels, while MRR/R@5/R@10 worsen; all eight difference intervals cross zero, so a retrieval upgrade is unestablished. Completion on the 45-axis and 187-axis subsets also worsens. Every task uses the same epoch-59 checkpoint.",
    "causal": "Observed facts are that, under these fixed training settings, greater capacity raises both training and validation primary errors, with regression across most nutrition axes and every source partition. The control supports rejecting this direct 256/8/FF1024 replacement, not a universal claim that larger Transformers are ineffective. Width, head count, feedforward size, parameter count, initialization shapes and random consumption change together; the same seed does not match initialization. Worse training fit is inconsistent with the simple narrative of better training fit but worse validation. Interactions between optimization settings and capacity remain an untested hypothesis. The first 48 epochs were preserved and recovery state and numerical loops were verified, but an uninterrupted GPU counterfactual is unavailable. Single-seed food-group intervals exclude seed-population, repeated-selection, label-validity and external-transfer uncertainty.",
    "decision": "Version decision: reject this fixed capacity recipe as a completion upgrade and do not add seeds 20260923/20260924. Preserve the original 48 epochs, interruption record, recovered run and all results. Retain width-192 MAE with source weight=1 as the method control; retention does not establish optimal parameters. The overall RF objective remains unachieved. The [machine-readable decision](../r9/capacity256_v1/decision.json) records the screen.",
    "next": "The next priority returns to the width-192 control and tests initial learning rate 3e-4 to 6e-4, retaining architecture, loss, source weight, sampling and the 60-epoch schedule. Earlier learning-rate screening and this experiment’s worse training fit motivate further validation of optimization settings rather than assuming capacity alone will help. This does not predict that a higher rate must win. That concrete candidate is not yet registered or launched; it requires separate registration, checks, a runtime estimate and a scheduled follow-up. Data and RF/XGB stay frozen, the test stays closed, and source-held-out and few-shot transfer remain unverified."
  }
}


def paragraph(text, prefix, replacement):
    parts = text.split('\n\n')
    matches = [i for i, part in enumerate(parts) if part.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError('Expected exactly one paragraph: '+prefix)
    parts[matches[0]] = replacement
    return '\n\n'.join(parts)


def main():
    for path in [OUT/'interpretation_edit.json', VERSION/'decision.json', ANALYSIS/'visual_review.json']:
        if path.exists():
            raise FileExistsError(path)
    evidence = read(OUT/'evidence.json')
    analysis = read(ANALYSIS/'summary.json')
    if analysis['status'] != 'complete_registered_capacity256_analysis' or any(analysis['screening_gates'].values()):
        raise ValueError('This interpretation requires the observed failed capacity screen')
    for name, expected in evidence['document_hashes'].items():
        if sha(OUT/name) != expected:
            raise ValueError('Draft changed: '+name)
    documents = {}
    for lang, words in WORDS.items():
        content = (OUT/f'REPORT_{lang}.md').read_text(encoding='utf-8')
        content = paragraph(content, '草稿：' if lang=='ZH' else 'DRAFT:', words['scope'])
        header = content.partition('## 1. ')[0]
        _, boundary, after = content.partition('## 2. ')
        if not boundary:
            raise ValueError('Missing historical section boundary')
        history, boundary, section = ('## 2. '+after).partition('## 13. ')
        if not boundary:
            raise ValueError('Missing new capacity section')
        section = words['title']+'\n'+section.partition('\n')[2]
        section = paragraph(section,
            '正值低估、正值高估和零值贡献' if lang=='ZH' else 'Positive underprediction, positive overprediction and explicit-zero contributions',
            '\n\n'.join(words[k] for k in ['diagnosis','curves','axes','cases','aux'])+
            '\n\n[Visual review](../../../reports/v9_r9_capacity256_analysis_v1/visual_review.json) · [Axis contrasts](../../../reports/v9_r9_capacity256_axis_changes_v1/summary.json) · [Case review](../../../reports/v9_r9_capacity256_cases_v1/summary.json)')
        section = paragraph(section, '这是容量组合干预' if lang=='ZH' else 'This is a capacity-group intervention', words['causal'])
        section = paragraph(section, '这里只呈现自动筛选' if lang=='ZH' else 'This is the automatic screen only.', words['decision'])
        section = paragraph(section, '完成审阅后按登记条件' if lang=='ZH' else 'After review, apply the registered criteria', words['next'])
        note = ('恢复后完整60轮、原48轮历史、最优检查点、两份完整预测和检索结果的独立审计均通过，11项比较全部完成。恢复训练实际耗时2511.188秒，原保存耗时10908.641秒，合计13419.829秒；这不是包含未知丢失工作和停机时间的完整墙钟成本。训练拟合另耗时74.890秒，没有优化器更新。训练恢复前16项合成检查通过，真实状态重载与训练行前向核验通过。案例读取有pandas混合类型提示，但逐值/元数据验证成功。原中断原因仍不明确，不能省略该失败记录。' if lang=='ZH' else
            'Independent audits passed for all 60 epochs, the preserved 48-epoch prefix, the selected checkpoint, both complete prediction tables and retrieval outputs. All 11 comparisons completed. Resumed training took 2511.188 seconds, added to 10908.641 saved seconds for 13419.829 recorded seconds; this is not full wall time including unknown lost work and downtime. Training-fit inference added 74.890 seconds without optimizer updates. Sixteen synthetic recovery checks and real saved-state/training-row checks passed. Case loading emitted a pandas mixed-type warning, but value and metadata verification succeeded. The original interruption cause remains unknown and the failure is retained.')
        section = section.replace('### 13.4.', note+'\n\n### 13.4.', 1)
        documents[lang] = header+words['overview']+'\n\n'+words['lead']+'\n\n'+history+section
    visual = {'status':'actual_visual_review_complete',
        'figure_sha256':sha(ANALYSIS/'learning_curves.png'), 'all_six_panels_viewed':True,
        'capacity_selected_epoch':59, 'control_selected_epoch':60,
        'observations':[
            'Primary, legacy-log and positive validation curves decline and flatten late; the larger model remains worse.',
            'Conditional zero error is also higher at the selected checkpoint but differs from the fixed-denominator zero contribution.',
            'Late clipping fraction is higher for capacity256; gradient norms fluctuate without visible divergence.',
            'All legends, labels, fixed RF/XGB lines and selected-epoch markers are visible.',
            'The original48-epoch prefix and recovery are independently documented; visual smoothness does not prove an uninterrupted counterfactual.'],
        'scope':'Actual six-panel PNG inspection plus exact machine evidence; no global-convergence or GPU counterfactual claim.'}
    decision = {'status':'screen_complete_fixed_capacity256_recipe_rejected',
        'candidate':'tf256_mae_lr3e4_60', 'run':'output/v9_r9_methods/tf256_mae_lr3e4_60_recovery1',
        'gates':analysis['screening_gates'], 'decision':'reject_this_fixed_capacity_recipe_as_completion_upgrade',
        'retained_control':'output/v9_r9/tf192_mae_lr3e4_60', 'additional_seeds_to_launch':[],
        'analysis_sha256':sha(ANALYSIS/'summary.json'), 'fit_sha256':sha(ROOT/'reports/v9_r9_capacity256_fit_v1/summary.json'),
        'recovery_audit_sha256':sha(ROOT/'reports/v9_r9_capacity256_lr3e4_60_audit_v1/recovery_verification.json'),
        'reason':'All three same-seed MAE screen gates fail; completion and legacy regress with intervals supporting regression. Training fit also worsens. No RF superiority.',
        'next_question':'Separately register width192 MAE learning-rate3e-4_to6e-4 contrast, holding all other recipe settings fixed; do not infer a guaranteed optimization benefit.',
        'next_candidate_registered_or_launched':False,
        'scientific_confirmation':False, 'goal_achieved':False,
        'data_modified':False, 'baseline_refit':False, 'complete_test_opened':False}
    version = '# V9-R9 / '+documents['ZH'].partition('## 13. ')[2]
    version = re.sub(r'^### 13\.(\d)\.', r'## \1.', version, flags=re.M)
    def rebase(match):
        target = match[2]
        if re.match(r'^[a-z]+:|^#|^/', target):
            return match[0]
        path, separator, anchor = target.partition('#')
        relative = Path(os.path.relpath((OUT/path).resolve(), VERSION)).as_posix()
        return f'[{match[1]}]({relative}{separator}{anchor})'
    version = re.sub(r'\[([^\]\n]+)\]\(([^)\n]+)\)', rebase, version)
    version = version.replace('第6节', '综合报告第6节').replace('\n\n## 1.',
        '\n\n完整[中文报告](../../report_snapshot_r9_capacity256_v1/REPORT_ZH.md)和[英文报告](../../report_snapshot_r9_capacity256_v1/REPORT_EN.md)包含数据来源、历次迭代、冻结RF/XGBoost方法与全部结果。\n\n## 1.', 1)
    (ANALYSIS/'visual_review.json').write_text(json.dumps(visual,indent=2)+'\n', encoding='utf-8')
    (VERSION/'decision.json').write_text(json.dumps(decision,indent=2)+'\n', encoding='utf-8')
    for lang, content in documents.items():
        (OUT/f'REPORT_{lang}.md').write_text(content, encoding='utf-8')
    (VERSION/'README.md').write_text(version, encoding='utf-8')
    record = {'status':'scientific_interpretation_added_actual_bilingual_review_required',
        'original_draft_hashes':evidence['document_hashes'],
        'document_hashes':{f'REPORT_{lang}.md':sha(OUT/f'REPORT_{lang}.md') for lang in documents},
        'version_readme_sha256':sha(VERSION/'README.md'), 'script_sha256':sha(Path(__file__)),
        'manual_bilingual_review_complete':False, 'goal_achieved':False,
        'training_performed':False, 'data_modified':False, 'baseline_refit':False, 'complete_test_opened':False}
    (OUT/'interpretation_edit.json').write_text(json.dumps(record,indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status':record['status']}))


if __name__=='__main__':
    main()
