"""Record the reviewed source-weight result; bilingual proofreading remains separate."""
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/foodnutrigpt_v9_research/report_snapshot_r9_source0_v1'
VERSION = ROOT/'experiments/foodnutrigpt_v9_research/r9/source0_v1'
ANALYSIS = ROOT/'reports/v9_r9_source0_analysis_v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))


def paragraph(text, prefix, replacement):
    parts = text.split('\n\n')
    matches = [i for i, part in enumerate(parts) if part.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError('Expected one paragraph: '+prefix)
    parts[matches[0]] = replacement
    return '\n\n'.join(parts)


def english_spacing(text):
    """Improve prose spacing while preserving tables, code, hashes and acronyms."""
    result=[]
    fenced=False
    for line in text.splitlines():
        if line.startswith('```'): fenced=not fenced
        if not fenced and not line.startswith(('|','```')):
            line=re.sub(r'\b([A-Za-z]{2,})(?=\d)',
                        lambda m:m[0] if m[0].isupper() else m[0]+' ',line)
            line=re.sub(r'(?<=[a-z])(?=\[)', ' ', line)
        result.append(line)
    return '\n'.join(result)


WORDS = {
    'ZH': {
        'scope': '版本范围：R0–R9完整阶段报告，含MAE三种子复验，以及MSE、轴×数值残差、关闭来源校准的单因素筛选。新增三项配方均未通过各自接受条件，保留MAE、source weight=1作为当前控制。第2–11节保留此前已审阅的历史证据，其“下一轮”等表述属于相应历史时点；第12节记录最新完整版本。Transformer优于冻结RF的整体目标仍未达到。',
        'lead': '本轮关闭来源校准完成60轮训练，实际耗时10,516.218秒，按固定验证主指标选中第58轮。补全主误差0.187741，比同种子MAE点退步2.281%，改善区间[−5.219%,+0.430%]；旧log-MAE点退步4.314%。三个预登记筛选条件均未满足，拒绝此固定配方，不追加其23/24种子。相对冻结RF的0.683%点改善区间为[−2.544%,+3.791%]，跨零且只有一个种子，不能宣称稳定超过RF。仅名称误差也上升；完整证据和归因见第12节。',
        'title': '## 12. 关闭来源校准：筛选完成，拒绝此固定升级配方',
        'diagnosis': '共同分母分解精确重构主误差增加0.004187495：正值低估+0.002141826、正值高估+0.001668563、显式零+0.000377106。注意：条件零值宏平均从0.092039降至0.090369，但零值对总体主误差的贡献反而增加，因为两种统计的轴内分母和有效权重不同；不能将条件零值均值降低直接称为总体收益。全量1,828,536个训练目标的无来源推理主误差从0.134937增至0.136431，也没有显示基础预测拟合改善。训练条件正值/零值均值虽降低，也不能相加推出主误差改善。',
        'curves': '已实际查看六面板曲线。SOURCE0选第58轮，MAE控制选第60轮，两个模型后期曲线趋缓；这不证明训练已达全局最优。末10轮梯度裁剪比例0.347503，控制为0.350883，未支持“关闭校准导致更多后期裁剪”的解释。均值梯度范数有波动，但未见发散走势。训练目标在去掉校准项后语义不同，因此机制比较使用共同的无来源拟合指标，不直接以两条训练loss的高低证明优劣。',
        'axes': '58/142轴点改善；11个逐轴区间支持改善、26个支持退步，均未作多重比较校正，9轴验证支持不足30个食品组。7个训练支持少于100组的轴贡献+0.003714232，20个100–999组轴贡献+0.000491940，115个至少1000组轴贡献−0.000018677。膳食纤维家族贡献+0.002794349、脂肪酸+0.001034044；FooDB来源分区贡献+0.003903528。Lignin和Isomeric linolenic acids各只有17个验证食品组，其逐轴区间均跨零；不能把这些稀疏轴认定为已证实的因果瓶颈。来源、家族和支持度分区相互重叠，不能相加。',
        'cases': '从保存预测重建并核对40个RF对比极端案例。20个退步案例均来自FooDB，含19个正值、1个零值，来自16个食品候选组；其中10个Cholesterol案例的预测至少比保留标签低500倍。这既不证明单位错误，也不证明标签正确。20个改善案例含15个正值、5个零值，覆盖5个来源、15个食品组。极端尾部不是代表性或独立样本，不据此修改数据；逐条名称、浓度和预测留在本地数据目录。',
        'aux': '仅名称142轴主误差0.444413，比同种子MAE高4.222%，改善区间[−7.037%,−1.252%]支持这一条件性退步。相对冻结RF/XGB，它的仅名称表现仍较好，但明显弱于名称KNN（0.262260），不能因此宣称名称任务已解决。全可见/30%可见的检索R@10为0.009677/0.004762，八个同种子检索指标均点退步且其差值区间均跨零。45轴补全点改善而187轴补全变差，辅助指标不替代142轴筛选。所有结果使用同一个第58轮检查点。',
        'causal': '观察事实是：删除校准目标未满足当前配方的任何筛选条件，基础训练主误差也更高。对照支持保留source weight=1，拒绝这个固定的关闭方案；不证明校准总是有益，也不证明所有关闭方案必然有害。参数量和初值相同，干预改变校准梯度及后续优化轨迹；真实来源偏差校正、隐式正则和优化效应仍不能唯一分离。普通GPU存在微小非确定性，而单种子条件区间不包含种子总体、反复选型、标签真实性或外部迁移的不确定性。没有改数据、指标或门槛来追求获胜。',
        'decision': '版本决定：拒绝此SOURCE0配方作为补全升级，不追加20260923/20260924复验，保留全部产物和失败记录，继续以MAE、source weight=1为方法控制。区间跨零不证明等效；保留控制是遵循预先接受条件，不是证明其所有参数最优。本轮不完成Transformer优于冻结RF的整体目标。见[机器决定](../r9/source0_v1/decision.json)。',
        'next': '下一项优先问题是原PDF的容量组合：256维、8头、FF1024，相对于当前192维、6头、FF768，保持MAE、来源权重1及其余训练方法不变。假设更大的条件表示容量可能改善拟合，但也可能增加泛化代价；宽度、头数和前馈维度共同变化，不能称为纯宽度因果实验。该方向已列于R9总计划，具体候选尚未登记或训练。数据和RF/XGB继续冻结、测试保持关闭，后续仍先估时并定时回访。来源留出、少样本迁移及可靠foundation model能力仍未验证。',
    },
    'EN': {
        'scope': 'Version scope: completed R0–R9 stage report, including three-seed MAE confirmation and single-factor MSE, axis-by-value and source-calibration-removal screens. All three added recipes fail their acceptance conditions; MAE with source weight=1 remains the current control. Sections2–11 preserve previously reviewed historical evidence, including next-step statements from those historical stages. Section12 records the latest complete version. The overall Transformer-over-frozen-RF objective remains unachieved.',
        'lead': 'Removing source calibration completed60 epochs in10,516.218 seconds and selected epoch58 by the fixed validation primary metric. Primary completion error is0.187741, a2.281% point regression against same-seed MAE, with an improvement interval of[−5.219%,+0.430%]; legacy log-MAE regresses by4.314% at the point estimate. All three preregistered screening gates fail, so this fixed recipe is rejected without seeds23/24. Its0.683% point gain over frozen RF has an interval of[−2.544%,+3.791%], crosses zero and comes from one seed; stable superiority is not established. Name-only error also increases. Section12 provides the complete evidence and interpretation.',
        'title': '## 12. Removing source calibration: screen complete, fixed upgrade recipe rejected',
        'diagnosis': 'The common-denominator decomposition reconstructs the primary increase of0.004187495 exactly: positive underprediction contributes+0.002141826, positive overprediction+0.001668563 and explicit zeros+0.000377106. Conditional zero macro error falls from0.092039 to0.090369 while its contribution to overall primary error increases, because the statistics use different within-axis denominators and effective weights. A lower conditional zero mean must not be called an overall benefit. Source-free inference on all1,828,536 training targets increases primary error from0.134937 to0.136431, providing no evidence of improved base fitting. Lower conditional positive and zero training means likewise cannot be added to infer primary improvement.',
        'curves': 'All six curve panels were visually inspected. SOURCE0 selects epoch58 and MAE selects epoch60; both curves flatten late, without proving global optimality. Last-ten-epoch clipping is0.347503 versus0.350883 for the control, which does not support more frequent late clipping as the explanation. Mean gradient norms fluctuate without a divergent trend. Removing calibration changes the meaning of training loss; therefore mechanism comparisons use common source-free fit metrics rather than interpreting raw training-loss levels as a performance comparison.',
        'axes': '58/142 axes improve at the point estimate;11 axis intervals support improvement and26 support regression, without multiplicity correction, while9 axes have fewer than30 validation food groups. The7 axes below100 training groups contribute+0.003714232, the20 with100–999 contribute+0.000491940, and the115 with at least1000 contribute−0.000018677. Dietary fibre contributes+0.002794349 and fatty acids+0.001034044; the FooDB source partition contributes+0.003903528. Lignin and Isomeric linolenic acids each have only17 validation food groups and both axis intervals cross zero, so they are not established causal bottlenecks. Source, family and support partitions overlap and must not be added together.',
        'cases': 'Forty extreme RF-comparison cases were reconstructed and checked against saved predictions. All20 worsening cases come from FooDB:19 positive labels and1 explicit zero across16 food groups. Ten Cholesterol predictions are at least500-fold below the retained labels, proving neither unit error nor label correctness. The20 improving cases include15 positive and5 zero labels across5 sources and15 food groups. These tails are neither representative nor independent samples and do not justify data changes. Individual names, concentrations and predictions remain in the local data directory.',
        'aux': 'Name-only142-axis error is0.444413, a4.222% regression against same-seed MAE; its improvement interval of[−7.037%,−1.252%] supports this conditional regression. It remains better than frozen RF/XGB on name-only prediction but substantially worse than name kNN at0.262260, so that task is not solved. Full/30%-visible retrieval R@10 is0.009677/0.004762. All eight same-seed retrieval metrics worsen at the point estimate and all difference intervals cross zero. Completion improves on45 axes but worsens on187 axes; auxiliary metrics do not replace the142-axis screen. Every task uses the same epoch58 checkpoint.',
        'causal': 'Observed facts are that removing calibration fails every gate for this fixed recipe and also raises source-free training primary error. The control supports retaining source weight=1 and rejecting this removal recipe, not a universal benefit of calibration or harm from all removal schemes. Parameter count and initialization match, but the intervention changes calibration gradients and subsequent optimization. Real source-bias correction, implicit regularization and optimization effects remain inseparable. Ordinary GPU execution has small nondeterministic variation, and single-seed conditional intervals exclude seed-population, repeated-selection, label-validity and external-transfer uncertainty. Data, metrics and gates were not changed to obtain a win.',
        'decision': 'Version decision: reject this SOURCE0 recipe as a completion upgrade, do not add seeds20260923/20260924, retain all artifacts and failures, and continue using MAE with source weight=1 as the method control. An interval crossing zero does not establish equivalence; retaining the control follows the predeclared acceptance conditions rather than proving every parameter optimal. This version does not complete the overall Transformer-over-frozen-RF objective. See the[machine-readable decision](../r9/source0_v1/decision.json).',
        'next': 'The next priority is the original PDF capacity combination: width256,8 heads and FF1024 versus width192,6 heads and FF768, retaining MAE, source weight1 and the other training settings. Greater conditional representation capacity may improve fit but may also increase generalization costs. Width, head count and feedforward width change together, so this is not a pure width-effect experiment. The direction appears in the R9 master plan, but a concrete candidate has not yet been registered or trained. Data and RF/XGB remain frozen, test stays closed, and future runs continue to use runtime estimates and scheduled follow-ups. Source-held-out transfer, few-shot transfer and a reliable foundation-model claim remain unverified.',
    },
}


def main():
    for path in [OUT/'interpretation_edit.json',VERSION/'decision.json',ANALYSIS/'visual_review.json']:
        if path.exists(): raise FileExistsError(path)
    evidence=read(OUT/'evidence.json')
    analysis=read(ANALYSIS/'summary.json')
    if analysis['status']!='complete_registered_source0_analysis' or any(analysis['screening_gates'].values()):
        raise ValueError('This interpretation requires the observed failed source0 screen')
    for name,expected in evidence['document_hashes'].items():
        if sha(OUT/name)!=expected: raise ValueError('Draft changed: '+name)
    visual={'status':'actual_visual_review_complete','figure_sha256':sha(ANALYSIS/'learning_curves.png'),
        'all_six_panels_viewed':True,'source0_selected_epoch':58,'control_selected_epoch':60,
        'observations':['Both validation primary curves decline and flatten late; source0 remains above the control near the end.',
            'Legacy and positive conditional errors are not improved by removing calibration at the selected point.',
            'Conditional zero error is lower but is not the common-denominator primary contribution.',
            'Clipping fractions are similar late; gradient means fluctuate without a visible divergent trend.',
            'Legends, axis labels, fixed RF/XGB horizontal lines and selected-epoch stars are visible in all relevant panels.'],
        'scope':'Actual inspection of the generated six-panel PNG, supplemented by exact machine records; no claim of global convergence.'}
    (ANALYSIS/'visual_review.json').write_text(json.dumps(visual,indent=2)+'\n',encoding='utf-8')
    decision={'status':'screen_complete_fixed_source0_recipe_rejected','candidate':'tf192_mae_source0_lr3e4_60',
        'gates':analysis['screening_gates'],'decision':'reject_this_fixed_recipe_as_completion_upgrade',
        'retained_control':'output/v9_r9/tf192_mae_lr3e4_60','additional_seeds_to_launch':[],
        'analysis_sha256':sha(ANALYSIS/'summary.json'),
        'fit_sha256':sha(ROOT/'reports/v9_r9_source0_fit_v1/summary.json'),
        'reason':'All same-seed MAE screen gates fail. One-seed RF point improvement has a food-group interval crossing zero; no final RF confirmation.',
        'next_question':'Separately register and verify the PDF256/8/FF1024 capacity group while keeping MAE and source weight1. This changes a capacity combination, not width alone.',
        'next_candidate_registered_or_launched':False,'scientific_confirmation':False,'goal_achieved':False,
        'data_modified':False,'baseline_refit':False,'complete_test_opened':False}
    (VERSION/'decision.json').write_text(json.dumps(decision,indent=2)+'\n',encoding='utf-8')
    documents={}
    for lang,words in WORDS.items():
        if lang=='EN': words={key:english_spacing(value) for key,value in words.items()}
        content=(OUT/f'REPORT_{lang}.md').read_text(encoding='utf-8')
        content=paragraph(content,'草稿：' if lang=='ZH' else 'DRAFT:',words['scope'])
        content=content.replace('最新的轴×数值残差配方','此前的轴×数值残差配方',1) if lang=='ZH' else content.replace('The latest axis-by-value recipe','The preceding axis-by-value recipe',1)
        before,separator,after=content.partition('## 2. ')
        if not separator: raise ValueError('Missing historical sections')
        content=before.rstrip()+'\n\n'+words['lead']+'\n\n'+separator+after
        history,separator,section=content.partition('## 12. ')
        if not separator: raise ValueError('Missing new section')
        section=words['title']+'\n'+section.partition('\n')[2]
        section=paragraph(section,'正值低估、正值高估和零值贡献' if lang=='ZH' else 'Positive underprediction, positive overprediction and explicit-zero contributions',
            '\n\n'.join(words[key] for key in ['diagnosis','curves','axes','cases','aux'])+'\n\n[Visual review](../../../reports/v9_r9_source0_analysis_v1/visual_review.json) · [Axis contrasts](../../../reports/v9_r9_source0_axis_changes_v1/summary.json) · [Case review](../../../reports/v9_r9_source0_cases_v1/summary.json)')
        section=paragraph(section,'参数量和初值未变。' if lang=='ZH' else 'Parameter count and initialization are unchanged.',words['causal'])
        section=paragraph(section,'此处只呈现自动筛选。' if lang=='ZH' else 'This is the automatic screen only.',words['decision'])
        section=paragraph(section,'完成审阅后按登记条件决定' if lang=='ZH' else 'After review, apply the registered conditions',words['next'])
        commands='\n\n```powershell\n.venv/Scripts/python.exe -X utf8 scripts/train_foodnutrigpt_v9_r9_source0.py --candidate tf192_mae_source0_lr3e4_60 --output-dir output/v9_r9_methods/tf192_mae_source0_lr3e4_60\n.venv/Scripts/python.exe -X utf8 scripts/audit_foodnutrigpt_r9_source0.py --run output/v9_r9_methods/tf192_mae_source0_lr3e4_60 --output-dir reports/v9_r9_source0_lr3e4_60_audit_v1\n.venv/Scripts/python.exe -X utf8 scripts/analyze_foodnutrigpt_r9_source0.py --output-dir reports/v9_r9_source0_analysis_v1\n.venv/Scripts/python.exe -X utf8 scripts/diagnose_foodnutrigpt_r9_source0_fit.py --output-dir reports/v9_r9_source0_fit_v1 --local-dir data/local/research_diagnostics/v9_r9_source0_fit_v1\n```\n\n'
        note=('上述运行已完成。独立审计和11项比较均通过，正式训练无失败；案例读取的pandas混合类型提示未影响逐值和元数据核验。功能预检失败及恢复见下文。额外训练拟合推理耗时56.656秒，没有优化器更新。' if lang=='ZH' else 'These runs completed. Independent audit and all11 comparisons passed without a formal training failure. A pandas mixed-type warning during case loading did not affect value and metadata checks. The functional-preflight failure and recovery are recorded below. Additional training-fit inference took56.656 seconds without optimizer updates.')
        section=section.replace('### 12.4.',commands+note+'\n\n### 12.4.',1)
        documents[lang]=history+(english_spacing(section) if lang=='EN' else section)
    version='# V9-R9 / '+documents['ZH'].partition('## 12. ')[2]
    version=re.sub(r'^### 12\.(\d)\.',r'## \1.',version,flags=re.M)
    def rebase(match):
        target=match[2]
        if re.match(r'^[a-z]+:|^#|^/',target): return match[0]
        path,separator,anchor=target.partition('#')
        relative=Path(os.path.relpath((OUT/path).resolve(),VERSION)).as_posix()
        return f'[{match[1]}]({relative}{separator}{anchor})'
    version=re.sub(r'\[([^\]\n]+)\]\(([^)\n]+)\)',rebase,version)
    version=version.replace('第6节','综合报告第6节').replace('\n\n## 1.',
        '\n\n完整[中文报告](../../report_snapshot_r9_source0_v1/REPORT_ZH.md)和[英文报告](../../report_snapshot_r9_source0_v1/REPORT_EN.md)包含数据来源、全部历史迭代及冻结RF/XGB方法。\n\n## 1.',1)
    for lang,content in documents.items(): (OUT/f'REPORT_{lang}.md').write_text(content,encoding='utf-8')
    (VERSION/'README.md').write_text(version,encoding='utf-8')
    receipt={'status':'scientific_interpretation_added_actual_bilingual_review_required',
        'original_draft_hashes':evidence['document_hashes'],
        'document_hashes':{f'REPORT_{lang}.md':sha(OUT/f'REPORT_{lang}.md') for lang in documents},
        'version_readme_sha256':sha(VERSION/'README.md'),'script_sha256':sha(Path(__file__)),
        'manual_bilingual_review_complete':False,'goal_achieved':False,'training_performed':False,
        'data_modified':False,'baseline_refit':False,'complete_test_opened':False}
    (OUT/'interpretation_edit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status']}))


if __name__=='__main__': main()
