"""Record actual reviewed learning-rate findings; final proofreading is separately verified."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/foodnutrigpt_v9_research/report_snapshot_r9_lr6e4_v1'
VERSION=ROOT/'experiments/foodnutrigpt_v9_research/r9/lr6e4_v1'
ANALYSIS=ROOT/'reports/v9_r9_lr6e4_analysis_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
WORDS=json.loads(r'''{
  "ZH": {
    "scope": "版本范围：R0–R9完整阶段报告，包含MAE三种子复验，以及MSE、轴×数值残差、关闭来源校准、PDF容量组合和提高学习率的筛选。五个追加配方均未通过接受条件，方法控制仍为192维MAE、学习率3e-4、source weight=1。第2–13节保留此前已审阅的历史证据；第14节为最新完整学习率实验。整体Transformer优于冻结RF的目标仍未达到。",
    "overview": "## 1. 研究目标与主要结论\n\n当前目标是在冻结数据与RF/XGBoost结果的条件下，通过有验证依据的单阶段Transformer方法改善营养补全，同时跟踪仅名称预测和营养到名称检索。每次迭代保留预先假设、全部结果、失败和归因边界；不同模型的任务最佳结果不拼接成单个模型的能力。\n\n192维MAE控制的三种子补全主误差为0.189184 ± 0.006925，固定RF为0.189031。食品组改善区间跨零，旧log指标均值退步也超过2%保护条件，尚无稳定超越RF的证据。完整数据来源、历史迭代与基线方法见第2–13节。\n\n最新同种子学习率3e-4→6e-4实验完成60轮并选中第60轮，补全主误差0.233616，较控制0.183553退步27.274%，改善区间[−30.750%,−24.050%]；旧log-MAE退步33.798%。相对固定RF主误差退步23.586%，相对XGB退步33.887%。训练拟合也更差，三个筛选门槛均失败，拒绝本6e-4配方，不追加23/24种子。仅名称点改善未经区间支持，检索明显退步；详见第14节。",
    "title": "## 14. 提高学习率：筛选完成，拒绝此6e-4配方",
    "intro": "本轮完整结果已实际审阅，包括六面板学习曲线、全部142轴、全部来源/家族/支持数分区与40个RF差值极端案例。保留失败启动记录、完整60轮训练和全部负结果；本阶段完成不等于整体研究目标完成。",
    "diagnosis": "共同分母分解精确重构主误差增加0.050062398：正值低估+0.035975859、正值高估+0.007335887、显式零+0.006750652。三项均增加，主要代价来自正值低估。条件正值与零值宏平均分别从0.302773/0.092039升至0.368750/0.119552；条件统计与共同分母贡献不能相加。对全部1,828,536训练目标的基础推理主误差由0.134937升至0.194387，差+0.059450107，训练正值和零值误差也均升高。",
    "curves": "已实际查看六个面板。两个模型都选第60轮；6e-4的主误差、旧log和正值误差在前期及后期都更高，后期下降趋缓，不能据此证明全局收敛。末10轮裁剪比例反而较低：0.306816对0.350883；初10轮也较低，平均裁剪前梯度范数没有可见爆炸。因此数据不支持“更多梯度裁剪导致失败”这一简单解释。运行9025.391秒（约2.51小时）短于估算，但不同运行时的机器负载未受控，不能归因于提高学习率使计算更快。",
    "axes": "5/142轴点改善，没有未校正逐轴区间支持改善，123个支持退步；9轴验证支持不足30食品组。20个氨基酸轴全部点退步，只有20轴点优于固定RF。115个至少1000训练组的轴贡献+0.035465，20个100–999组轴贡献+0.007104，7个少于100组轴贡献+0.007494；退步并非仅由稀疏轴造成。脂肪酸与维生素家族分别贡献+0.020593/+0.010597。24个来源分区全部点退步，FooDB贡献+0.019208。最大单轴代价Lignin仅17验证组，未校正区间跨零且很宽；不能把该轴认定为唯一瓶颈。",
    "cases": "已逐条核对40个极端案例。20个退步案例均为FooDB正值，来自14食品组，含10个Biotin和10个Cholesterol；后者预测都至少比保留标签低500倍。20个改善案例含1正值、19显式零，来自4来源、15食品组。这些尾部案例不是总体的代表性独立样本。规模差异既不确认单位错误，也不证明保留标签正确；不据此更改标签，食物级原始数值只保留在本地。",
    "aux": "仅名称主误差0.415257，较控制点改善2.616%，改善区间[−0.690%,+5.627%]跨零；旧log-MAE点改善4.074%，区间同样跨零。它优于冻结RF/XGB的名称预测，但仍较名称KNN的0.262260差58.338%。全可见/30%可见检索R@10为0.002739/0.001456；相对控制，8项排名指标均点退步，除30%可见R@1外的7个区间均支持退步。45轴与187轴补全也点退步。三项任务使用同一个第60轮检查点，不能将名称的点改善视作已证实的整体收益。",
    "causal": "已观察事实是：同一初始化、数据、任务顺序和60轮曝光下，整条学习率日程加倍后，训练和验证的主误差均明显更高，退步广泛覆盖营养轴及来源。对照支持拒绝此固定6e-4配方；它不支持所有较大学习率都无效，也不能识别唯一优化机制。AdamW每步衰减也随学习率变化；后续优化路径及GPU非确定性、单种子与重复验证选择仍是限制。训练拟合变差，与“训练更好、验证更差”的单纯过拟合叙述不一致；较低的裁剪比例不能证明梯度变得更健康。没有梯度冲突诊断，不能据此宣称多任务冲突。",
    "decision": "版本决定：拒绝6e-4作为补全升级，不追加20260923/20260924，保留所有产物及失败记录。继续以192维MAE、3e-4、source weight=1作为方法控制；保留控制不等于已证明最优。整体RF目标仍未达到。[机器决定](../r9/lr6e4_v1/decision.json)记录全部筛选条件。",
    "next": "下一项候选优先在已有1e-4（主误差0.194543）与3e-4（0.183553）之间检验2e-4，其余配置完全固定。6e-4的失败说明收益并非随学习率单调增加，中间值只是待检验的参数假设，不预设获胜。该候选尚未登记或启动；需独立登记、核验、估时和定时回访。数据及RF/XGB继续冻结、测试关闭；来源留出、少样本迁移和foundation model能力未获证明。"
  },
  "EN": {
    "scope": "Version scope: completed R0–R9 stage report, including three-seed MAE confirmation and screens of MSE, an axis-by-value residual, source-calibration removal, the PDF capacity group and a higher learning rate. All five added recipes fail their acceptance conditions. The method control remains width-192 MAE, learning rate 3e-4 and source weight=1. Sections 2–13 preserve previously reviewed history; Section 14 presents the latest complete learning-rate experiment. The overall Transformer-over-frozen-RF objective remains unachieved.",
    "overview": "## 1. Research objective and principal findings\n\nThe objective is to improve nutrition completion through validated single-stage Transformer methods while keeping data and RF/XGBoost results frozen. Name-only prediction and nutrition-to-name retrieval remain tracked tasks. Every iteration retains its prior hypothesis, all results, failures and attribution limits; different models’ best task results are not combined into one model’s capabilities.\n\nThe width-192 MAE control has three-seed primary completion error of 0.189184 ± 0.006925, compared with 0.189031 for fixed RF. Its food-group improvement interval crosses zero and mean legacy-log regression exceeds the 2% guard. Stable superiority over RF remains unestablished. Sections 2–13 provide data provenance, historical iterations and baseline methods.\n\nThe latest same-seed learning-rate 3e-4 to 6e-4 experiment completed 60 epochs and selected epoch 60. Primary completion error is 0.233616 versus 0.183553 for the control: a 27.274% regression, with an improvement interval of [−30.750%,−24.050%]. Legacy log-MAE regresses by 33.798%. Primary error is 23.586% worse than fixed RF and 33.887% worse than XGBoost. Training fit also worsens and all three screening gates fail. This 6e-4 recipe is rejected without seeds 23/24. Name-only point improvement lacks interval support, while retrieval deteriorates substantially. See Section 14.",
    "title": "## 14. Higher learning rate: screen complete, 6e-4 recipe rejected",
    "intro": "The complete results have been reviewed, including all six curve panels, all 142 axes, every source/family/support partition and 40 extreme error-difference cases against RF. The failed launch, full 60-epoch run and all negative results are retained. Completion of this stage does not complete the overall research objective.",
    "diagnosis": "The common-denominator decomposition reconstructs the primary increase of 0.050062398: positive underprediction contributes +0.035975859, positive overprediction +0.007335887 and explicit zeros +0.006750652. All three increase, with positive underprediction the largest contributor. Conditional positive and zero macro errors rise from 0.302773/0.092039 to 0.368750/0.119552. Conditional averages cannot be added to common-denominator contributions. Source-free inference on all 1,828,536 training targets raises primary error from 0.134937 to 0.194387, a difference of +0.059450107; both training positive and zero errors also rise.",
    "curves": "All six panels were visually inspected. Both models select epoch 60. The 6e-4 model has higher primary, legacy-log and positive errors early and late, with flattening late improvements that do not prove global convergence. Its last-ten-epoch clipping fraction is lower, 0.306816 versus 0.350883; the first-ten fraction is also lower, and mean pre-clipping gradient norms show no visible explosion. These observations do not support a simple explanation based on more frequent clipping. Recorded duration is 9025.391 seconds, about 2.51 hours, below the estimate. Machine load across runs was uncontrolled, so this does not demonstrate that a larger learning rate accelerates computation.",
    "axes": "Only 5/142 axes improve by point estimate; no unadjusted axis interval supports improvement and 123 support regression. Nine axes have fewer than 30 validation food groups. All 20 amino-acid axes regress by point estimate, and only 20 axes are point-better than fixed RF. The 115 axes with at least 1,000 training groups contribute +0.035465, the 20 with 100–999 contribute +0.007104 and the seven below 100 contribute +0.007494. Regression is therefore not confined to sparse axes. Fatty acids and vitamins contribute +0.020593/+0.010597. All 24 source partitions regress by point estimate, with FooDB contributing +0.019208. Lignin has the largest individual axis contribution but only 17 validation groups and a wide unadjusted interval crossing zero; it cannot be identified as the sole bottleneck.",
    "cases": "All 40 extreme cases were inspected. The 20 worse cases are FooDB positive labels from 14 food groups, comprising ten Biotin and ten Cholesterol cases. All ten Cholesterol predictions are at least 500-fold below retained labels. The 20 better cases comprise one positive and 19 explicit zeros from four sources and 15 food groups. These tails are not representative independent samples. Scale discrepancies establish neither a unit error nor label correctness. They do not justify relabeling, and individual numeric profiles remain local.",
    "aux": "Name-only primary error is 0.415257, a 2.616% point improvement over the control, but its improvement interval [−0.690%,+5.627%] crosses zero. Legacy log-MAE improves by 4.074% at the point estimate, also without interval support. Name-only performance exceeds fixed RF/XGB but remains 58.338% worse than name KNN at 0.262260. Full/30%-visible retrieval Recall@10 is 0.002739/0.001456. All eight ranking metrics regress against the control by point estimate; seven intervals support regression, with 30%-visible Recall@1 the exception. Completion on the 45- and 187-axis subsets also worsens. All tasks use the same epoch-60 checkpoint, so name-only point improvement is not a confirmed overall benefit.",
    "causal": "The observed result is that doubling the full learning-rate schedule, while matching initialization, data, task orders and 60-epoch exposure, substantially worsens both training and validation primary error across many axes and sources. The controlled comparison supports rejecting this fixed 6e-4 recipe, not a universal claim against larger learning rates or a unique optimization mechanism. AdamW shrinkage per step changes with learning rate; subsequent optimization paths, GPU nondeterminism, one seed and repeated validation selection limit attribution. Worse training fit does not match a simple better-training/worse-validation overfitting account, and lower clipping does not establish healthier gradients. No gradient-conflict diagnosis was performed, so task conflict is unproven.",
    "decision": "Version decision: reject 6e-4 as a completion upgrade and do not add seeds 20260923/20260924. Preserve every artifact and failure record. Retain width-192 MAE, 3e-4 and source weight=1 as the method control; retention does not prove optimality. The overall RF objective remains unachieved. The [machine-readable decision](../r9/lr6e4_v1/decision.json) records all gates.",
    "next": "The next proposed candidate is 2e-4, between the completed 1e-4 result (primary error 0.194543) and 3e-4 control (0.183553), holding all other settings fixed. Failure at 6e-4 shows that benefit is not monotonic in learning rate; the intermediate value is a testable parameter hypothesis, not a promised win. It has not been registered or launched and requires separate registration, checks, a runtime estimate and scheduled follow-up. Data and RF/XGB remain frozen, the test stays closed, and source-held-out, few-shot and foundation-model capabilities remain unproven."
  }
}''')

def paragraph(text,prefix,replacement):
    parts=text.split('\n\n')
    matches=[i for i,p in enumerate(parts) if p.startswith(prefix)]
    if len(matches)!=1:raise ValueError('Expected one paragraph: '+prefix)
    parts[matches[0]]=replacement
    return '\n\n'.join(parts)

def main():
    for path in [OUT/'interpretation_edit.json',VERSION/'decision.json',ANALYSIS/'visual_review.json']:
        if path.exists():raise FileExistsError(path)
    evidence=read(OUT/'evidence.json')
    analysis=read(ANALYSIS/'summary.json')
    if analysis['status']!='complete_registered_lr6e4_analysis' or any(analysis['screening_gates'].values()):
        raise ValueError('This interpretation requires the observed failed screen')
    documents={}
    for lang,w in WORDS.items():
        path=OUT/f'REPORT_{lang}.md'
        if sha(path)!=evidence['document_hashes'][path.name]:raise ValueError('Draft changed')
        text=path.read_text(encoding='utf-8')
        text=paragraph(text,'草稿：' if lang=='ZH' else 'DRAFT:',w['scope'])
        start=text.index('## 1. ')
        end=text.index('## 2. ',start)
        text=text[:start]+w['overview']+'\n\n'+text[end:]
        head,_,section=text.partition('## 14. ')
        section=w['title']+'\n\n'+section.split('\n\n',1)[1]
        section=paragraph(section,'本节为' if lang=='ZH' else 'This section is',w['intro'])
        section=paragraph(section,'正值低估、高估' if lang=='ZH' else 'Positive underprediction, overprediction',
                          '\n\n'.join(w[key] for key in ['diagnosis','curves','axes','cases','aux']))
        section=paragraph(section,'本次可比较的是' if lang=='ZH' else 'The comparison identifies',w['causal'])
        section=paragraph(section,'未满足三个' if lang=='ZH' else 'Does not meet all three',w['decision'])
        section=paragraph(section,'通过筛选仅允许' if lang=='ZH' else 'Passing only permits',
            '本候选不追加复验；下一项实验必须独立登记，不能重定义既有筛选条件。' if lang=='ZH' else
            'This candidate receives no additional seeds. A subsequent experiment requires separate registration without redefining these gates.')
        section=paragraph(section,'下一项最小对照' if lang=='ZH' else 'The next minimal contrast',w['next'])
        documents[lang]=head+section
    decision={'status':'screen_complete_fixed_lr6e4_recipe_rejected','candidate':'tf192_mae_lr6e4_60',
        'run':'output/v9_r9_methods/tf192_mae_lr6e4_60','gates':analysis['screening_gates'],
        'decision':'reject_this_fixed_learning_rate_recipe_as_completion_upgrade',
        'retained_control':'output/v9_r9/tf192_mae_lr3e4_60','additional_seeds_to_launch':[],
        'analysis_sha256':sha(ANALYSIS/'summary.json'),
        'fit_sha256':sha(ROOT/'reports/v9_r9_lr6e4_fit_v1/summary.json'),
        'reason':'All three same-seed gates fail. Completion and legacy regress with supporting intervals; train fit worsens and seven retrieval intervals regress. Name-only point gain lacks interval support.',
        'next_question':'Separately register learning rate2e-4 between previously tested1e-4 and3e-4, holding other settings fixed.',
        'next_candidate_registered_or_launched':False,'scientific_confirmation':False,'goal_achieved':False,
        'data_modified':False,'baseline_refit':False,'complete_test_opened':False}
    (VERSION/'decision.json').write_text(json.dumps(decision,indent=2)+'\n',encoding='utf-8')
    visual={'status':'actual_six_panel_visual_review_complete','figure_sha256':sha(ANALYSIS/'learning_curves.png'),
        'all_six_panels_viewed':True,'labels_and_legends_readable':True,
        'observations':WORDS['EN']['curves'],'causal_limit':'No universal convergence or clipping mechanism inferred.',
        'training_performed':False,'data_modified':False,'complete_test_opened':False}
    (ANALYSIS/'visual_review.json').write_text(json.dumps(visual,indent=2)+'\n',encoding='utf-8')
    for lang,text in documents.items():(OUT/f'REPORT_{lang}.md').write_text(text,encoding='utf-8')
    receipt={'status':'actual_interpretation_added_final_proofreading_required',
        'original_draft_hashes':evidence['document_hashes'],
        'document_hashes':{f'REPORT_{lang}.md':sha(OUT/f'REPORT_{lang}.md') for lang in documents},
        'decision_sha256':sha(VERSION/'decision.json'),'visual_review_sha256':sha(ANALYSIS/'visual_review.json'),
        'script_sha256':sha(Path(__file__)),'report_complete':False,'goal_achieved':False}
    (OUT/'interpretation_edit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'decision':decision['status']}))

if __name__=='__main__':main()
