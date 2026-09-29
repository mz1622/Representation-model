"""Record actual reviewed learning-rate findings; final proofreading is separately verified."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/foodnutrigpt_v9_research/report_snapshot_r9_lr2e4_v1'
VERSION=ROOT/'experiments/foodnutrigpt_v9_research/r9/lr2e4_v1'
ANALYSIS=ROOT/'reports/v9_r9_lr2e4_analysis_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
WORDS=json.loads(r'''{
  "ZH": {
    "scope": "版本范围：R0–R9完整阶段报告，包含MAE三种子复验，以及MSE、轴×数值残差、关闭来源校准、PDF容量组合和两次学习率筛选。六个追加配方均未通过接受条件；方法控制仍为192维MAE、学习率3e-4、source weight=1。第2–14节保留已审阅历史，第15节为最新2e-4实验。整体Transformer优于冻结RF的目标仍未达到。",
    "overview": "## 1. 研究目标与主要结论\n\n目标是在冻结数据与RF/XGBoost结果的条件下，通过有验证依据的单阶段Transformer方法改善营养补全，同时跟踪仅名称预测和营养到名称检索。每次迭代保留假设、全部结果、失败与归因边界，不拼接不同模型的最好任务成绩。\n\n192维MAE控制的三种子补全主误差为0.189184 ± 0.006925，固定RF为0.189031；食品组改善区间跨零，旧log均值退步超过2%保护条件，尚无稳定超越RF的证据。数据来源、历史迭代和具体基线方法见第2–14节。\n\n最新学习率3e-4→2e-4实验完成60轮，选中第59轮。补全主误差0.185954，较控制0.183553退步1.308%，改善区间[−4.533%,+1.711%]跨零；旧log-MAE点退步3.202%。相对RF点改善1.627%，区间[−1.992%,+4.909%]仍跨零；相对XGB退步6.572%。三个筛选条件均失败，拒绝此固定配方，不追加23/24种子。\n\n本轮91/142轴点改善，但7个少于100训练组的轴贡献了+0.004935的主误差变化，抵消了115个至少1000训练组轴的−0.002620。训练整体拟合改善，稀疏轴却并非一致改善，不能仅凭总训练误差宣称普遍过拟合或训练不足。仅名称主误差退步3.320%，检索8项指标均点退步但区间跨零；详细分析见第15节。",
    "title": "## 15. 中间学习率：筛选完成，拒绝此2e-4配方",
    "intro": "本轮已实际审阅六面板曲线、142轴、全部来源/家族/支持数分区、40条RF差值极端案例，以及7个低支持轴的训练与验证拟合。保留首次功能预检失败、独立确定性诊断和完整训练记录；阶段完成不等于整体目标完成。",
    "diagnosis": "共同分母分解重构主误差增加0.002401221：正值低估+0.000168008、正值高估+0.001833093、显式零+0.000400120，主要代价来自正值高估。条件正值/零值平均反而从0.302773/0.092039小幅降为0.299753/0.091817；这些条件统计使用不同的单元权重和分母，不能相加代替主指标。全部1,828,536训练目标的基础推理主误差由0.134937降为0.131288，差−0.003649547；训练正值和零值平均也下降。",
    "curves": "六个面板均已查看。2e-4在早期有部分主误差优势，中后期与控制交错、最终略差；旧log指标后期也更差，正值和零值曲线相近。它选第59轮，控制选第60轮；后期趋缓不能证明已达到最优。末10轮裁剪比例0.407196高于控制0.350883，初10轮0.431289也高于0.401310，平均裁剪前梯度较大但没有可见发散。这是路径差异，不能据此把裁剪定为失败原因。实际训练9020.750秒（约2.51小时），机器负载未受控，不作算法速度结论。",
    "axes": "91/142轴点改善，27个未校正逐轴区间支持改善、7个支持退步，9轴验证支持不足30组；71轴点优于RF，氨基酸20轴中9个点退步。115个至少1000训练组的轴贡献−0.002620，20个100–999组轴贡献+0.000087，7个不足100组轴贡献+0.004935。13/24来源分区点改善，FooDB贡献+0.004373；这些是主指标固定分母下的分区贡献，不是来源独立平均误差。最大轴代价Lignin贡献+0.002241，其17组验证区间跨零且宽，不能认定为已确认的唯一瓶颈。",
    "support": "补充拟合检查使用已有聚合指标，未重新训练或推理。7个低支持轴中只有2个训练误差改善、2个验证误差改善，而且两组轴不同；它们的训练贡献变化为+0.000471，验证为+0.004935。Lignin训练0.046066→0.048372，验证0.358843→0.677018；Menaquinone-4训练0.042710→0.044842，验证0.153933→0.334648。Cellulose和18:2异构体训练改善但验证退步，Sorbitol和Sulfur则相反。不能将总体“训练更好、验证更差”叙述套到每个稀疏轴，也尚无证据证明增加这些轴曝光就能解决问题。[补充诊断](../../../../reports/v9_r9_lr2e4_support_fit_v1/summary.json)保存全部7轴记录与全142轴对照。",
    "cases": "已逐条阅读40条极端案例。20个退步案例都是FooDB的Cholesterol正值，来自19食品组；预测均至少比保留标签低500倍。20个改善案例含14正值、6显式零，来自4来源、14食品组，其中13条为Biotin。尾部案例不能代表总体或独立样本，也不证明单位错误或标签正确；不据此改值或排除数据，食物级数值留在本地。",
    "aux": "仅名称主误差0.440569，较控制退步3.320%，改善区间[−5.938%,−0.950%]支持退步；旧log点退步0.290%，区间跨零。仅名称仍优于固定RF/XGB，却较名称KNN的0.262260差67.990%。全可见/30%可见检索R@10为0.009326/0.004870；相对控制的8项排名指标均点退步，但8个配对区间均跨零。45轴与187轴补全主误差分别改善至0.560798/0.276157，不能因此更换预先固定的142轴主指标。所有任务来自同一第59轮检查点。",
    "causal": "已观察事实是：同初值、数据、曝光和60轮预算下，学习率全日程缩为2/3后，整体训练拟合改善、常见轴多有收益，但补全主指标没有可靠改善，旧log点退步超过保护条件，仅名称也退步。对照支持拒绝本固定2e-4配方，不支持学习率越小越差或普遍过拟合的结论。每步AdamW衰减也随学习率变化；GPU非确定性、单种子、反复验证选择以及训练/验证食品和支持不同，均限制机制归因。稀疏轴的大间隙与来源构成也可能有关，未做相应消融，不能据此指定单一原因。",
    "decision": "版本决定：拒绝2e-4作为补全升级，不追加20260923/20260924；保留所有产物和失败记录。继续以192维MAE、3e-4、source weight=1作控制，保留不代表最优。三个门槛均失败，但不能把跨零区间写成已确认的整体退步；RF点优势也不能冒充稳定获胜。[机器决定](../r9/lr2e4_v1/decision.json)记录筛选结果。",
    "next": "下一项最小方法假设优先检验更强dropout：在保留的3e-4控制上仅将.15改为.25。低支持轴原有训练误差已低、验证误差更高，因此先检验正则化，而不凭稀疏就增加采样；预期若有效，允许训练拟合略差而验证主指标改善。这仍是待检验假设，可能同时损害训练与验证。候选尚未登记或启动，需独立登记、功能核验、估时和定时回访；不叠加本次2e-4，也不修改数据、树结果、指标或测试状态。来源留出、少样本和foundation model能力仍未获证明。"
  },
  "EN": {
    "scope": "Version scope: completed R0–R9 stage report, including three-seed MAE confirmation and screens of MSE, an axis-by-value residual, source-calibration removal, the PDF capacity group and two learning-rate changes. None of the six added recipes passes acceptance. The method control remains width-192 MAE, learning rate 3e-4 and source weight=1. Sections 2–14 preserve reviewed history; Section 15 presents the latest 2e-4 experiment. The overall Transformer-over-frozen-RF objective remains unachieved.",
    "overview": "## 1. Research objective and principal findings\n\nThe objective is to improve nutrition completion through validated single-stage Transformer methods while keeping data and RF/XGBoost results frozen. Name-only prediction and nutrition-to-name retrieval remain tracked tasks. Every iteration retains hypotheses, all results, failures and attribution limits; best task scores from different models are not combined.\n\nThe width-192 MAE control has three-seed primary completion error of 0.189184 ± 0.006925 versus 0.189031 for fixed RF. Its food-group improvement interval crosses zero and mean legacy-log regression exceeds the 2% guard. Stable superiority over RF remains unestablished. Sections 2–14 cover data provenance, historical iterations and baseline methods.\n\nThe latest learning-rate 3e-4 to 2e-4 experiment completed 60 epochs and selected epoch 59. Primary error is 0.185954 versus 0.183553 for the control: a 1.308% regression, with an improvement interval of [−4.533%,+1.711%] crossing zero. Legacy log-MAE regresses by 3.202% at the point estimate. Primary error is 1.627% better than RF, but the interval [−1.992%,+4.909%] still crosses zero; it is 6.572% worse than XGBoost. All three screening gates fail, so this fixed recipe is rejected without seeds 23/24.\n\nAlthough 91/142 axes improve by point estimate, seven axes with fewer than 100 training groups contribute +0.004935 to the primary change, offsetting −0.002620 from 115 axes with at least 1,000 groups. Overall training fit improves, but sparse axes do not improve consistently. Aggregate training error alone therefore establishes neither universal overfitting nor insufficient training. Name-only primary error regresses by 3.320%; all eight retrieval metrics worsen by point estimate but their intervals cross zero. See Section 15.",
    "title": "## 15. Intermediate learning rate: screen complete, 2e-4 recipe rejected",
    "intro": "The six curve panels, all 142 axes, every source/family/support partition, 40 extreme RF error-difference cases, and training/validation fit on all seven low-support axes have been reviewed. The initial functional-check failure, isolated deterministic diagnostic and complete training record are retained. Completing this stage does not complete the overall objective.",
    "diagnosis": "The common-denominator decomposition reconstructs the primary increase of 0.002401221: positive underprediction +0.000168008, positive overprediction +0.001833093 and explicit zeros +0.000400120. Positive overprediction accounts for most of the increase. Conditional positive/zero averages instead decrease slightly from 0.302773/0.092039 to 0.299753/0.091817. These conditional statistics use different cell weights and denominators and cannot be added to replace the primary metric. Source-free inference on all 1,828,536 training targets reduces primary error from 0.134937 to 0.131288, a difference of −0.003649547; training positive and zero averages also decrease.",
    "curves": "All six panels were inspected. The 2e-4 model has some early primary-error advantages, alternates with the control later and finishes slightly worse. Late legacy-log error is also higher, while positive and zero curves are close. It selects epoch 59 versus epoch 60 for the control; late flattening does not prove optimality. Its last-ten clipping fraction is higher, 0.407196 versus 0.350883, as is its first-ten fraction, 0.431289 versus 0.401310. Mean pre-clipping gradients are larger without visible divergence. These trajectory differences do not identify clipping as the cause of failure. Training took 9020.750 seconds, about 2.51 hours; uncontrolled machine load prevents an algorithmic speed claim.",
    "axes": "Of 142 axes, 91 improve by point estimate; 27 unadjusted intervals support improvement and seven support regression. Nine axes have fewer than 30 validation groups. Seventy-one axes are point-better than RF and nine of 20 amino-acid axes are point-worse than the control. The 115 axes with at least 1,000 training groups contribute −0.002620, the 20 with 100–999 contribute +0.000087, and the seven below 100 contribute +0.004935. Thirteen of 24 source partitions improve by point estimate; FooDB contributes +0.004373. These are fixed-denominator primary contributions, not independently averaged source errors. Lignin has the largest axis contribution, +0.002241, but only 17 validation groups and a wide interval crossing zero; it is not an established sole bottleneck.",
    "support": "The supplementary fit check uses existing aggregate metrics without training or inference. Among the seven low-support axes, only two improve in training and two in validation, and they are different pairs. Their training contribution changes by +0.000471 and validation by +0.004935. Lignin training error changes from 0.046066 to 0.048372 and validation from 0.358843 to 0.677018; Menaquinone-4 changes from 0.042710 to 0.044842 in training and 0.153933 to 0.334648 in validation. Cellulose and 18:2 isomers improve in training but worsen in validation, whereas Sorbitol and Sulfur show the reverse. The aggregate better-training/worse-validation account does not apply to every sparse axis, and greater exposure is not yet established as a remedy. The [supplementary diagnosis](../../../../reports/v9_r9_lr2e4_support_fit_v1/summary.json) retains all seven records and the complete 142-axis contrast.",
    "cases": "All 40 extreme cases were read. The 20 worse cases are positive FooDB Cholesterol labels from 19 food groups; every prediction is at least 500-fold below its retained label. The 20 better cases comprise 14 positives and six explicit zeros from four sources and 14 groups, including 13 Biotin cases. These tails are neither representative nor independent samples and establish neither unit errors nor label correctness. They do not justify relabeling or exclusion; individual numeric profiles remain local.",
    "aux": "Name-only primary error is 0.440569, a 3.320% regression against the control, with an improvement interval [−5.938%,−0.950%] supporting regression. Legacy-log point regression is 0.290%, with an interval crossing zero. Name-only remains better than fixed RF/XGB but is 67.990% worse than name KNN at 0.262260. Full/30%-visible retrieval Recall@10 is 0.009326/0.004870. All eight ranking metrics regress against the control by point estimate, but all eight paired intervals cross zero. Completion primary errors on the 45- and 187-axis subsets improve to 0.560798/0.276157; this does not justify replacing the preregistered 142-axis primary metric. Every task uses the same epoch-59 checkpoint.",
    "causal": "The observed result is that scaling the full learning-rate schedule to two thirds, while matching initialization, data, exposure and the 60-epoch budget, improves aggregate training fit and many common axes but does not reliably improve primary completion, exceeds the legacy point-regression guard and worsens name-only prediction. This controlled contrast supports rejecting the fixed 2e-4 recipe, not a monotonic learning-rate claim or universal overfitting. Per-step AdamW shrinkage also changes with learning rate. GPU nondeterminism, one seed, repeated validation selection, and different training/validation foods and support limit mechanism attribution. Large sparse-axis gaps may also reflect source composition; without relevant ablations, no single cause is established.",
    "decision": "Version decision: reject 2e-4 as a completion upgrade and do not add seeds 20260923/20260924. Preserve all artifacts and failures. Retain width-192 MAE, 3e-4 and source weight=1 as the control; retention does not establish optimality. All three gates fail, but intervals crossing zero must not be presented as confirmed overall regression, nor may the RF point advantage be presented as a stable win. The [machine-readable decision](../r9/lr2e4_v1/decision.json) records the screen.",
    "next": "The next proposed minimal method test is stronger dropout: change only .15 to .25 on the retained 3e-4 control. Low-support axes already have low training errors and larger validation errors, motivating a regularization test before increasing sampling merely because support is sparse. If useful, somewhat worse training fit should accompany better validation primary error. This remains a hypothesis and may worsen both partitions. The candidate is not yet registered or launched and requires separate registration, functional checks, a runtime estimate and scheduled follow-up. It will not combine with 2e-4 or modify data, trees, metrics or test status. Source-held-out, few-shot and foundation-model capabilities remain unproven."
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
    if analysis['status']!='complete_registered_lr2e4_analysis' or any(analysis['screening_gates'].values()):
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
        head,_,section=text.partition('## 15. ')
        section=w['title']+'\n\n'+section.split('\n\n',1)[1]
        section=paragraph(section,'本节为' if lang=='ZH' else 'This section is',w['intro'])
        section=paragraph(section,'正值低估、高估' if lang=='ZH' else 'Positive underprediction, overprediction',
                          '\n\n'.join(w[key] for key in ['diagnosis','curves','axes','support','cases','aux']))
        section=paragraph(section,'本次可比较的是' if lang=='ZH' else 'The comparison identifies',w['causal'])
        section=paragraph(section,'未满足三个' if lang=='ZH' else 'Does not meet all three',w['decision'])
        section=paragraph(section,'通过筛选仅允许' if lang=='ZH' else 'Passing only permits',
            '本候选不追加复验；下一项实验必须独立登记，不能重定义既有筛选条件。' if lang=='ZH' else
            'This candidate receives no additional seeds. A subsequent experiment requires separate registration without redefining these gates.')
        section=paragraph(section,'下一项最小对照' if lang=='ZH' else 'The next minimal contrast',w['next'])
        documents[lang]=head+section
    decision={'status':'screen_complete_fixed_lr2e4_recipe_rejected','candidate':'tf192_mae_lr2e4_60',
        'run':'output/v9_r9_methods/tf192_mae_lr2e4_60','gates':analysis['screening_gates'],
        'decision':'reject_this_fixed_learning_rate_recipe_as_completion_upgrade',
        'retained_control':'output/v9_r9/tf192_mae_lr3e4_60','additional_seeds_to_launch':[],
        'analysis_sha256':sha(ANALYSIS/'summary.json'),
        'fit_sha256':sha(ROOT/'reports/v9_r9_lr2e4_fit_v1/summary.json'),
        'reason':'All three same-seed gates fail. Completion and legacy point regressions have intervals crossing zero; name-only primary regression is interval-supported. Aggregate training fit improves but low-support axes are mixed.',
        'next_question':'Separately register dropout .15 to .25 on retained learning-rate3e-4 control, holding other settings fixed; test regularization without assuming sparse-axis underfitting.',
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
