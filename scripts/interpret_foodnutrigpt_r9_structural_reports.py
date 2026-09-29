"""Render reviewed observations into bilingual drafts; final verification is separate.

Keeps previously reviewed Sections 2-15 byte-for-byte modulo relative links.
Only aggregate statistics are exported; individual numeric profiles remain local.
"""
import copy
import json
from pathlib import Path
import re
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import report_foodnutrigpt_r9_drop25 as base
from foodcomp.research_transformer_r9 import frozen_inputs
from foodcomp.research_r0 import write_json
read=base.read;sha=base.digest;table=base.table
DROP=ROOT/'experiments/foodnutrigpt_v9_research/report_snapshot_r9_drop25_v1'
OUT=ROOT/'experiments/foodnutrigpt_v9_research/report_snapshot_r9_rmsnorm_v1'
VERSION=ROOT/'experiments/foodnutrigpt_v9_research/r9'


def evidence(method):
    a=read(ROOT/f'reports/v9_r9_{method}_analysis_v1/summary.json')
    f=read(ROOT/f'reports/v9_r9_{method}_fit_v1/summary.json')
    for obj in [a,f]:base.verify_hashes(obj['input_hashes'])
    if any(a['screening_gates'].values()):
        raise ValueError('This interpretation is specific to the actually failed screens')
    assert not a['data_modified'] and not a['baseline_refit'] and not a['complete_test_opened']
    return a,f


def shared_tables(a,f,method):
    # Reuse the already tested numeric renderer with an explicit role mapping.
    mapped=copy.deepcopy(a); fitted=copy.deepcopy(f)
    if method!='drop25':
        mapped['records']['drop25']=mapped['records'].pop(method)
        mapped['fixed_denominator_drop25_minus_mae']=mapped['fixed_denominator_'+method+'_minus_mae']
        for key in ['source_free_training_fit','source_free_validation']:
            fitted[key]['drop25']=fitted[key].pop(method)
    result=base.tables(mapped,fitted)
    return {k:v.replace('DROP25',method.upper()) for k,v in result.items()}


def common_overview(lang,method):
    if lang=='ZH':
        main=('最新RMSNorm结构候选完成60轮，选中第55轮：补全0.187225，比原Transformer 0.183553退步2.000%，改善区间[−4.458%,+0.181%]跨零；旧log-MAE退步5.919%。相对RF点改善0.956%但区间跨零，相对XGB退步7.300%。Name-only点改善1.198%但区间跨零；检索全部8项点退步。拒绝替换父结构，见第17节。' if method=='rmsnorm' else
              '本快照完成dropout .15→.25版本收尾：60轮选中57轮，补全0.206847，比控制0.183553退步12.690%，改善区间[−15.874%,−9.682%]；name-only退步9.105%，检索8项点退步。拒绝此固定配方，见第16节。')
        return ('## 1. 研究目标与主要结论\n\n目标是在冻结数据和树参照下验证Transformer结构及训练方法，同时报告补全、仅名称和名称检索。用户最新要求禁止种子、dropout、学习率、MAE/MSE及维度搜索；历史章节中的“下一轮”与种子复验安排仅保留当时记录，不构成当前执行指令。\n\n'+main+
        '\n\n原Transformer三种子补全为0.189184 ± 0.006925，固定RF为0.189031、XGB为0.174487；尚无稳定超越RF的证据。历史MLP128为0.183435 ± 0.001222，名称维度不同，不能将差异只归因于架构。单次最好成绩不替代三种子统计。新结构只运行固定种子，食品组区间不涵盖种子总体。\n\n数据来源、训练数据、基线具体方法及R0–R8迭代见第2–4节；旧Transformer及方法对照完整保留在后续历史章节。原始单位证据、别名、外部来源和少样本迁移仍未闭环，不能宣称已建立nutrition foundation model。\n\n')
    main=('The latest RMSNorm structural candidate completed 60 epochs and selected epoch 55. Completion MAE is 0.187225 versus 0.183553 for the parent: a 2.000% regression, with a gain interval [−4.458%,+0.181%] crossing zero. Legacy log-MAE regresses by 5.919%. Its point gain over RF is 0.956% with an interval crossing zero, while error is 7.300% higher than XGBoost. Name-only improves by 1.198% at the point estimate with an interval crossing zero; all eight retrieval metrics worsen. Reject replacing the parent; see Section 17.' if method=='rmsnorm' else
          'This snapshot closes the dropout .15 to .25 experiment: 60 epochs, selected epoch 57, completion error 0.206847 versus 0.183553 for the control, a 12.690% regression with gain interval [−15.874%,−9.682%]. Name-only regresses by 9.105% and all eight retrieval metrics worsen by point estimate. Reject this fixed recipe; see Section 16.')
    return ('## 1. Research objective and principal findings\n\nThe objective is to validate Transformer structure and training methods on frozen data and tree references, reporting completion, name-only prediction and name retrieval. The latest user directive prohibits seed, dropout, learning-rate, MAE/MSE and dimensionality searches. Next-experiment and seed-replication statements in historical sections describe past decisions, not current execution instructions.\n\n'+main+
    '\n\nThe original Transformer has three-seed completion error 0.189184 ± 0.006925, versus 0.189031 for fixed RF and 0.174487 for fixed XGBoost; stable superiority over RF remains unestablished. Historical MLP128 yields 0.183435 ± 0.001222, with a different name dimensionality, so differences cannot be attributed solely to architecture. The best single seed does not replace three-seed statistics. New structures use one fixed seed; food-group intervals exclude seed-population uncertainty.\n\nSections 2–4 document data provenance, training data, baseline methods and R0–R8 iterations. Subsequent historical sections retain the original Transformer and all method contrasts. Unresolved units, aliases, external-source and few-shot transfer prevent a nutrition foundation-model claim.\n\n')


def update_dropout(lang,text):
    zh=lang=='ZH'
    words=lambda a,b:a if zh else b
    marker='## 16. '
    before,_,section=text.partition(marker)
    assert section
    section=marker+section
    replacements=[
      (words('本节为已完成实验的机器证据草稿；实际曲线、逐轴、案例解读和版本决定尚待审阅。不得将草稿当作完整版本或最终模型接受.',
             'This section is a machine-evidence draft for the completed experiment. Actual curve, axis and case interpretation and the version decision still require review. A draft does not constitute a complete version or final model acceptance.'),
       words('本节完成训练、独立重放、曲线及全部142轴、分区与40条本地案例审阅；结论是拒绝此固定dropout配方，不代表研究目标已经达成。',
             'Training, independent replay, six curve panels, all 142 axes, partitions and 40 local cases have been reviewed. Reject this fixed dropout recipe; completing its report does not achieve the broader research objective.')),
      (words('本节数字尚不能替代实际曲线与案例审阅。','These numbers do not substitute for actual curve and case review.'),
       words('实际曲线显示主误差、旧log与正值误差长期高于控制；后段零值误差、裁剪比例与平均裁剪前梯度也更高，没有数值发散。',
             'The actual curves show persistently higher primary, legacy-log and positive errors; late zero error, clipping fraction and mean preclip gradient norm are also higher, without numerical divergence.')),
      (words('未满足三个预注册筛选条件；仍需实际审阅形成版本决定。单种子不能确认稳定改进或超越RF。',
             'Does not meet all three registered screening conditions; the version decision still requires actual review. One seed cannot confirm stable improvement or superiority over RF.'),
       words('三个筛选条件全部失败；拒绝此固定配方，保留原MAE192控制。补全、名称和检索同时退步，不选择其中另一检查点拼接能力。',
             'All three screening gates fail. Reject this recipe and retain the original MAE192 control. Completion, name-only and retrieval all worsen; no alternate checkpoint is selected to combine task-specific best scores.')),
      (words('通过筛选仅允许另行登记20260923/24，不自动启动或改变现有门槛。最终接受需三种子与固定RF的共同协议证据。失败或无改善同样形成完整版本，不能删去。',
             'Passing only permits separately registered seeds 20260923/24; it does not automatically launch them or change the gates. Final acceptance requires three-seed evidence against fixed RF under the common protocol. Failures and null improvements must also remain complete documented versions.'),
       words('最新用户指令已停止种子及超参数搜索，本候选不追加种子。单种子局部结论不构成稳定超越RF或外部泛化证明；失败完整保留。',
             'The latest user directive stops seed and hyperparameter searches; no additional seeds are run. A fixed-seed local result does not establish stable superiority over RF or external generalization; the failed experiment is retained in full.')),
      (words('下一项最小对照或复验由完整审阅决定。数据、树参照和测试状态保持冻结；来源留出、少样本迁移及foundation model能力仍未被证明。',
             'The next minimal contrast or replication follows complete review. Data, tree references and test status stay frozen; source-held-out transfer, few-shot transfer and foundation-model capabilities remain unproven.'),
       words('研究方向已转为结构对照；后续RMSNorm单独登记并完成，其结果在下一快照报告。数据、树参照和测试状态不变，不将本轮失败推广为所有正则化无效。',
             'Research has moved to structural contrasts. RMSNorm was separately registered and completed; its outcome is reported in the next snapshot. Data, tree references and test status remain unchanged; this failure does not show that regularization is universally ineffective.'))]
    # The Chinese draft uses a Chinese final stop in this paragraph.
    replacements[0]=(replacements[0][0].replace('接受.','接受。'),replacements[0][1])
    for old,new in replacements:
        if section.count(old)!=1:raise ValueError(('Expected one draft phrase',lang,old))
        section=section.replace(old,new,1)
    diagnosis=words(
      '观察事实：补全误差增加0.023293381，其中正值低估贡献+0.012578782、正值高估+0.009289540、显式零+0.001425059。训练同口径误差从0.134937增至0.162434（+0.027497173）。仅13/142轴点改善，未校正区间支持3轴改善、107轴退步；24来源中仅1个贡献改善，全部10个家族贡献退步。常见/中等/稀疏轴贡献均退步（+0.015204、+0.002461、+0.005628）。7个稀疏轴训练全部变差，验证仅Menaquinone-4改善。因而失败并非只由少量稀疏轴抵消广泛收益。\n\n案例审阅：20个较好案例包含10个正值、10个显式零，来自4来源、12食品组；20个较差案例均为FooDB正值，16个Cholesterol、4个Biotin，来自18食品组。16个胆固醇预测至少比保留标签低500倍。它们是预先固定的极端尾部检查，不能据此认定标签错误或数据库质量，更未修改数据。\n\n对照支持的解释：提高共享dropout在此预算下同时损害训练拟合和验证表现，不支持本轮正则化假设。优化不足、特征表达受扰、来源组成和标签问题均未被单独隔离，不能只称为“过拟合”或把失败唯一归因于某一dropout位置。',
      'Observed facts: completion error increases by 0.023293381, decomposed into positive underprediction +0.012578782, positive overprediction +0.009289540 and explicit zero +0.001425059. Training error under the common scoring rule rises from 0.134937 to 0.162434 (+0.027497173). Only 13/142 axes improve by point estimate; unadjusted intervals support 3 improvements and 107 regressions. Only 1 of 24 source contributions improves, and all 10 family contributions worsen. Common, medium-support and sparse axes all worsen (+0.015204, +0.002461, +0.005628). All seven sparse axes have worse training fit; only Menaquinone-4 improves on validation. The failure is not confined to sparse axes offsetting broad gains.\n\nCase review: the 20 better cases contain 10 positive and 10 explicit-zero labels, from 4 sources and 12 food groups. All 20 worse cases are FooDB positives: 16 Cholesterol and 4 Biotin, from 18 groups. All 16 cholesterol predictions are at least 500-fold below retained labels. These fixed extreme-tail checks establish neither label errors nor database quality; no data were changed.\n\nControlled interpretation: stronger shared dropout harms both training fit and validation performance under this budget, failing to support the regularization hypothesis. Insufficient optimization, disrupted feature expression, source composition and label issues remain unisolated alternatives. Neither overfitting nor any individual dropout site is established as the unique cause.')
    heading='### 16.6.'
    section=section.replace(heading,diagnosis+'\n\n'+heading,1)
    pre=before[:before.index('## 1.')]
    scope=pre.splitlines()[4]
    pre=pre.replace(scope,words('已审阅版本：第2–15节保留先前历史证据，第16节完成dropout负结果与归因记录。历史种子安排已被最新结构研究指令覆盖。',
        'Reviewed stage report: Sections 2–15 retain prior evidence; Section 16 closes the negative dropout result and its interpretation. The latest structural directive supersedes historical seed-expansion plans.'),1)
    history=before[before.index('## 2.'):]
    return pre+common_overview(lang,'drop25')+history+section


def rms_section(lang,a,f,shared,bridge):
    zh=lang=='ZH';w=lambda x,y:x if zh else y
    r=a['records']['rmsnorm']['manifest'];p=a['records']['mae']['manifest']
    function=read(ROOT/'reports/v9_r9_rmsnorm_functional_v1/verification.json')
    axes=read(ROOT/'reports/v9_r9_rmsnorm_axis_changes_v1/summary.json')
    titles=[w('研究问题与预先假设','Research question and preregistered hypothesis'),w('父版本与结构改动','Parent and structural intervention'),w('可复现信息、失败与成本','Reproducibility, failures and cost'),w('完整结果与固定参照','Complete results and fixed references'),w('机制诊断','Mechanism diagnostics'),w('因果分析','Causal interpretation'),w('版本决定','Version decision'),w('下一轮问题与测试状态','Next question and test status')]
    h=lambda n:f'### 17.{n}. {titles[n-1]}'
    spec_table=table(['Item','Value'],[['Training commit',r['code_commit']],['Checkpoint SHA256',r['checkpoint_sha256']],
       ['Data SHA256',r['data_hash']],['Panel SHA256',r['panel_hash']],['Name cache SHA256',r['name_cache_hash']],
       ['Seed / selected epoch / epochs',f"{r['seed']} / {r['best_epoch']} / 60"],
       ['Total / trainable parameters',f"{r['parameter_count']} / {r['trainable_parameter_count']}"],
       ['Training seconds',f"{r['elapsed_seconds']:.3f}"],['Pre-run estimate seconds',f"{function['estimated_training_seconds']:.3f}"],
       ['Training-fit inference seconds',f"{f['elapsed_seconds']:.3f}"],['Wake delay minutes',205],['Preflight tests',30]])
    parts=[w('## 17. R9-A1 / 候选10：编码器RMSNorm','## 17. R9-A1 / Candidate 10: encoder RMSNorm'),h(1),w(
       '检验取消编码器通道均值中心化、使用RMS归一化能否改善正值拟合与补全。已有Pre-LN保留，仅替换归一化构件。[RMSNorm原论文](https://arxiv.org/abs/1910.07467)提供质量与效率证据，[LLaMA §2.2](https://arxiv.org/html/2302.13971v1#S2.SS2)记录其采用；这些研究不保证营养任务收益。',
       'Test whether removing encoder channel-mean centering and using RMS normalization improves positive-value fit and completion. Existing pre-normalization is retained; only its normalization component changes. The [RMSNorm paper](https://arxiv.org/abs/1910.07467) reports quality and efficiency evidence, and [LLaMA Section 2.2](https://arxiv.org/html/2302.13971v1#S2.SS2) documents its adoption. Neither guarantees benefits for nutrition prediction.'),h(2),w(
       '对照为原MAE192/lr3e-4/dropout.15，而非上一dropout候选。3层各2处加最终输出，共7处LayerNorm改为RMSNorm，eps仍为1e-5，FP32；文本投影LayerNorm不变。取消这7处偏置，参数减少1344，其余初始张量及构造RNG完全相同。固定192维、3层、6头、FF768、MAE、dropout.15、lr3e-4、AdamW/wd1e-4、batch64、clip1、seed20260922、60轮余弦、训练来源残差。双向集合注意力不变。',
       'The control is original MAE192/lr3e-4/dropout .15, not the previous dropout candidate. Replace two norms per layer across three layers plus the final encoder norm: seven LayerNorms become RMSNorm, keeping epsilon 1e-5 and FP32. Text-projection LayerNorm is unchanged. Removing seven biases reduces parameters by 1,344; all shared initial tensors and constructor RNG match. Width 192, three layers, six heads, FF768, MAE, dropout .15, lr3e-4, AdamW/weight decay1e-4, batch64, clip1, seed20260922, 60-epoch cosine schedule and training-only source residual remain fixed. Bidirectional set attention is unchanged.'),h(3),spec_table,w(
       'Windows/Python3.10.19/PyTorch2.7.1+cu128/RTX5070Ti16GB；每轮337048任务、1828536观测目标。完整重放检查两份323809行预测、49913候选向量和19089检索排名，全部60轮学习率、顺序、曝光与父相同。预计11391.554秒，实际11875.859秒，不能以非受控用时推断算法效率。9月30日04:11（北京时间）启动，07:31队列结束，07:37定时回访核验退出；没有epoch监视循环。',
       'Environment: Windows/Python3.10.19/PyTorch2.7.1+cu128/RTX5070Ti16GB. Each epoch uses 337,048 tasks and 1,828,536 observed targets. Independent replay verifies both 323,809-row prediction tables, 49,913 candidate vectors and 19,089 retrieval ranks. Learning rates, orders and exposures match the parent across all 60 epochs. Estimated duration was 11,391.554 seconds; actual duration was 11,875.859 seconds. Uncontrolled runtime is not an algorithmic efficiency comparison. The run launched at 04:11 Beijing time on September 30, the queue ended at 07:31 and the scheduled 07:37 follow-up verified exit, without epoch polling.'),w(
       '预检失败已保留：默认融合LayerNorm与显式实现最大绝对差5.56335e-5超出原容差，正式训练尚未启动。config_v2修订只在诊断中统一非融合路径，要求逐值相等，随后恢复默认后端；训练spec没有改变。小批次50步损失0.170336→0.064700，初始/已学重载、隐藏标签/来源隔离、未观测轴和name-only候选检查通过。启动凭据解析曾因Windows七位小数时间戳失败，按共同微秒精度修复，未重启训练；这与模型失败不同。',
       'A preflight failure is retained: default fused versus explicit LayerNorm differed by up to 5.56335e-5, exceeding the initial tolerance before formal training. The config_v2 revision compares identical unfused paths with zero tolerance during diagnosis and then restores backend defaults; the training specification did not change. Small-batch loss fell 0.170336 to 0.064700 over 50 steps. Initial/learned reload, hidden-label/source isolation, unobserved queries and name-only candidate checks passed. Launch-receipt parsing initially rejected Windows seven-digit fractional timestamps; comparison at shared microsecond precision fixed this without restarting training. This operational issue is distinct from model failure.'),
       '[Plan](../r9/rmsnorm_v1/PLAN.md) · [Config](../r9/rmsnorm_v1/config_v2.json) · [Launch](../r9/rmsnorm_v1/launch.json) · [Preflight revision](../r9/rmsnorm_v1/PREFLIGHT_REVISION.md) · [Audit](../../../reports/v9_r9_rmsnorm_60_audit_v1/verification.json)',
       w('训练命令（在仓库根目录执行；已完成，不应重复启动）：','Training command (from the repository root; already completed, do not relaunch):'),
       '```powershell\n.\\.venv\\Scripts\\python.exe -X utf8 scripts/train_foodnutrigpt_v9_r9_rmsnorm.py --candidate tf192_mae_rmsnorm_lr3e4_60 --output-dir output/v9_r9_methods/tf192_mae_rmsnorm_lr3e4_60\n```',
       h(4),w('以下所有任务使用第55轮同一检查点。误差越低越好；检索为0–1比例，越高越好。各模型是冻结产物，MAE是同种子父模型。','All tasks below use the same epoch-55 checkpoint. Lower error is better; retrieval metrics are 0–1 proportions with higher values better. References are frozen artifacts; MAE denotes the same-seed parent.'),
       w('**营养补全**','**Nutrition completion**'),shared['completion'],
       w('**仅名称预测**','**Name-only prediction**'),shared['name_only'],
       w('**独立轴集合**','**Separate axis subsets**'),shared['subsets'],
       w('**名称检索**','**Name retrieval**'),shared['retrieval'],shared['intervals'],shared['retrieval_intervals'],w(
       '补全主误差0.187225，较父退步2.000%（改善95%区间−4.458%～+0.181%）；旧log退步5.919%，其区间支持退步。较RF点改善0.956%（−1.578%～+3.478%），较XGB退步7.300%（改善区间−10.588%～−3.974%）。Name-only 0.421303比父低1.198%，区间−1.400%～+3.691%，证据不足；相对KNN误差仍高60.643%。全部8项检索指标点退步，7项区间上界小于0，完整R@1区间上界恰为0。完整R@10 0.007099、30% R@10 0.003368，不能称为良好反向名称预测。',
       'Completion error is 0.187225, 2.000% worse than the parent (95% gain interval −4.458% to +0.181%). Legacy log-MAE regresses 5.919%, with an interval supporting regression. The point gain over RF is 0.956% (−1.578% to +3.478%); error is 7.300% higher than XGBoost (gain interval −10.588% to −3.974%). Name-only error 0.421303 improves 1.198% over the parent, but its interval −1.400% to +3.691% is inconclusive; error remains 60.643% above KNN. All eight retrieval point estimates worsen; seven intervals have upper bounds below zero and full-profile R@1 has an upper bound exactly zero. Full-profile R@10 is 0.007099 and 30%-visible R@10 is 0.003368, far from strong reverse-name prediction.'),w(
       '食品组配对重采样1000次；7344补全组，检索完整/30%分别7090/6458组。固定49913候选名称只由名称预测营养，不使用候选真实营养；查询不含名称。答案为精确原名称，未确认别名映射。区间不涵盖种子总体、标签真实性或反复验证选点。',
       'Intervals use 1,000 paired food-group resamples: 7,344 completion groups and 7,090/6,458 full/30%-visible retrieval groups. The fixed 49,913 candidates use nutrition predicted from name text, never measured candidate profiles; nutrient queries contain no names. Correctness uses exact original names without a confirmed alias map. Intervals exclude seed-population, label-validity and repeated-validation-selection uncertainty.'),
       h(5),shared['decomposition'],shared['fit'],shared['clipping'],
       '![RMSNorm learning curves](../../../reports/v9_r9_rmsnorm_analysis_v1/learning_curves.png)',w(
       '共同分母变化：正值低估+0.005743362、高估−0.001941824、显式零−0.000130315，合计+0.003671223。条件正值误差变差、零值略好；不能把不同分母的条件均值直接相加。训练同口径误差0.134937→0.140616（+0.005679225），训练和验证均变差。曲线没有发散，后段裁剪比例0.343497低于控制0.350883，平均裁剪前梯度也较低；较低梯度幅度没有转为更好补全。',
       'On the shared primary denominator, positive underprediction contributes +0.005743362, overprediction −0.001941824 and explicit zero −0.000130315, totaling +0.003671223. Conditional positive error worsens while zero error improves slightly; conditional means with different denominators cannot be summed. Common-protocol training error rises 0.134937 to 0.140616 (+0.005679225), so both training and validation worsen. Curves do not diverge. Late clipping fraction is 0.343497 versus 0.350883 for the control, and mean preclip gradient norm is lower; smaller gradients do not translate into better completion.'),w(
       '42/142轴点改善，未校正区间支持2轴改善、22轴退步；9轴验证支持数不足30，14/20氨基酸轴点退步。24来源中8个贡献改善、16个退步；FooDB贡献+0.001503，但不据此评判数据库质量。常见/中等/稀疏轴贡献分别+0.002520、−0.000651、+0.001802，退步并非只有稀疏轴。7个稀疏轴中3个训练改善、4个验证改善，趋势混合；Cellulose训练0.074063→0.053935、验证0.482925→0.546576，不能统一解释成训练不足。',
       '42/142 axes improve by point estimate; unadjusted intervals support 2 improvements and 22 regressions. Nine axes have fewer than 30 validation groups; 14/20 amino-acid axes worsen. Eight of 24 source contributions improve and 16 worsen. FooDB contributes +0.001503, which is not a database-quality judgment. Common, medium-support and sparse-axis contributions are +0.002520, −0.000651 and +0.001802; regression is not limited to sparse axes. Among seven sparse axes, three improve in training and four on validation, a mixed pattern. Cellulose training error falls 0.074063 to 0.053935 while validation rises 0.482925 to 0.546576, inconsistent with a universal insufficient-training explanation.'),w(
       '实际审阅40条固定极端案例：20个较好案例为12正值/8零值，来自5来源、15食品组；20个较差案例来自FooDB、18组，其中17胆固醇、2生物素、1月桂酸，含19正值/1零值。17胆固醇预测均至少比保留标签低500倍；这延续标签尺度疑点，但没有原始证据不能判定数值错误。模型对部分椰子脂肪酸有收益，也存在零标签月桂酸严重高估，不能只以“预测偏低”概括全部失败。',
       'All 40 fixed extreme cases were inspected. The 20 better cases contain 12 positives and 8 zeros from 5 sources and 15 food groups. The 20 worse cases come from FooDB and 18 groups: 17 Cholesterol, 2 Biotin and 1 lauric-acid case, containing 19 positives and 1 zero. All 17 cholesterol predictions are at least 500-fold below retained labels. This preserves a label-scale concern without proving incorrect values absent upstream evidence. Some coconut fatty-acid cases improve, while a zero-label lauric-acid case is severely overpredicted; underprediction does not describe every failure.'),
       '[All 142 axis contrasts](../../../reports/v9_r9_rmsnorm_axis_changes_v1/axis_changes.csv) · [Source/family/support partitions](../../../reports/v9_r9_rmsnorm_analysis_v1/rmsnorm_minus_mae_partitions.csv) · [Training/validation by axis](../../../reports/v9_r9_rmsnorm_fit_v1/per_axis_fit.csv) · [Case audit](../../../reports/v9_r9_rmsnorm_cases_v1/summary.json)',
       table(['Evaluation-only learned LayerNorm bridge','Value'],[['Primary error change',f"{bridge['primary_bridge_minus_original']:.12f}"],['Max cell scaled prediction difference',f"{bridge['maximum_cell_scaled_prediction_absolute_difference']:.9f}"],['All learned weights unchanged',True],['New optimization or selection',False]]),w(
       '结果后补充诊断使用父模型同一已学权重，仅改为显式LayerNorm执行，重新评分全部323809任务。主指标变化仅+0.000005322，相比RMSNorm总变化+0.003671223较小；但最大单元缩放预测差为0.034032，不声称每个预测逐位相等。这支持“仅评估融合路径数值差不足以解释总体退步”，不控制训练轨迹的所有数值扰动。',
       'A post-hoc bridge uses identical learned parent weights and only explicit LayerNorm execution, rescoring all 323,809 tasks. Primary error changes by +0.000005322, small relative to the RMSNorm change +0.003671223. Maximum cell-level scaled prediction difference is 0.034032, so individual predictions are not bitwise identical. This supports that evaluation-kernel rounding alone is insufficient to explain the aggregate regression; it does not control every numerical perturbation along training trajectories.'),
       '[Numerical bridge](../../../reports/v9_r9_rmsnorm_bridge_v1/summary.json)',h(6),w(
       '事实：固定配方下RMSNorm训练拟合与补全更差，正值低估增加，name-only点改善未获区间支持，检索退步。对照支持：这一7处归一化整体替换未改善本任务，不支持预先假设。尚未排除：取消中心化、RMS缩放、偏置去除的各自作用，固定预算下的优化适配、来源/标签差异和随机轨迹；不能据此断言RMSNorm在所有营养模型中无效。也不能仅凭较小梯度说训练更稳定或更优。',
       'Facts: under the fixed recipe, RMSNorm worsens training fit and completion, increases positive underprediction, gives inconclusive name-only point improvement and worsens retrieval. Controlled interpretation: the seven-site normalization replacement fails to improve this task, not supporting the preregistered hypothesis. Alternatives remain: individual effects of removing centering, RMS scaling and removing biases, optimization compatibility within a fixed budget, source/label differences and stochastic trajectories. This does not show RMSNorm is universally ineffective for nutrition, nor do smaller gradients establish superior stability or quality.'),h(7),shared['gates'],w(
       '拒绝用RMSNorm替换父模型，保留LayerNorm-MAE192作为后续结构控制。保留全部结果和失败；没有追加种子，也没有更换学习率、dropout、损失或维度来挽救候选。报告完成不等于模型目标达成。',
       'Reject replacing the parent with RMSNorm. Retain LayerNorm-MAE192 as the next structural control, preserving all results and failures. No extra seed, learning-rate, dropout, loss or dimensionality changes are used to rescue the candidate. Completing this report does not achieve the model objective.'),h(8),w(
       '下一项最小结构问题是SwiGLU乘法门控前馈是否改善输入相关的数值变换；仍以原LayerNorm父模型为对照，不把RMSNorm合并进去。保持FF宽度768及其他配方，新增门控矩阵将增加参数，必须明确披露，不能把收益唯一归因于门控。先独立登记和功能核验再训练；本报告时尚未启动。测试、数据和RF/XGB冻结，未来仍需来源留出与少样本迁移证据。',
       'The next minimal structural question is whether a SwiGLU multiplicative feedforward gate improves input-dependent numeric transformations. Use the original LayerNorm parent, without combining RMSNorm. Keep FF width768 and the rest of the recipe; the extra gate matrix increases parameter count and must be disclosed, so any benefit cannot be uniquely attributed to gating. Register and functionally verify it before training; it has not started at this report. Test, data and RF/XGBoost remain frozen; source-held-out and few-shot transfer evidence is still needed.')]
    return '\n\n'.join(parts)+'\n'


def main():
    if OUT.exists():raise FileExistsError(OUT)
    frozen_inputs(ROOT)
    d,df=evidence('drop25');a,f=evidence('rmsnorm')
    bridge=read(ROOT/'reports/v9_r9_rmsnorm_bridge_v1/summary.json')
    base.verify_hashes(bridge['input_hashes'])
    old_evidence=read(DROP/'evidence.json')
    for name,expected in old_evidence['document_hashes'].items():assert sha(DROP/name)==expected
    (DROP/'draft_evidence.json').write_bytes((DROP/'evidence.json').read_bytes())
    shared=shared_tables(a,f,'rmsnorm')
    for method,record,fit in [('drop25',d,df),('rmsnorm',a,f)]:
        version=VERSION/(method+'_v1')
        decision={'status':'screen_complete_fixed_'+method+'_recipe_rejected',
          'screening_gates':record['screening_gates'],'retained_parent':'output/v9_r9/tf192_mae_lr3e4_60',
          'seed_expansion_authorized':False,'data_modified':False,'baseline_refit':False,
          'complete_test_opened':False,'scientific_confirmation':False,'report_complete':False,
          'analysis_sha256':sha(ROOT/f'reports/v9_r9_{method}_analysis_v1/summary.json'),
          'fit_sha256':sha(ROOT/f'reports/v9_r9_{method}_fit_v1/summary.json'),
          'next_question':'Separately registered SwiGLU FFN against original LayerNorm parent; no prohibited hyperparameter search.'}
        if (version/'decision.json').exists():raise FileExistsError(version/'decision.json')
        write_json(version/'decision.json',decision)
    OUT.mkdir(parents=True)
    for lang in ['ZH','EN']:
        drop=update_dropout(lang,(DROP/f'REPORT_{lang}.md').read_text(encoding='utf-8'))
        (DROP/f'REPORT_{lang}.md').write_text(drop,encoding='utf-8')
        old=base.rebase(drop,DROP/f'REPORT_{lang}.md',OUT)
        first=old[:old.index('## 1.')]
        first=first.replace(first.splitlines()[4],
           '已审阅结构研究快照：第2–16节保留历史证据，第17节为RMSNorm完整负结果。最新用户结构研究约束优先。' if lang=='ZH' else
           'Reviewed structural snapshot: Sections 2–16 retain history; Section 17 documents the complete negative RMSNorm result. The latest structural-research directive takes precedence.')
        text=first+common_overview(lang,'rmsnorm')+old[old.index('## 2.'):].rstrip()+'\n\n'+rms_section(lang,a,f,shared,bridge)
        (OUT/f'REPORT_{lang}.md').write_text(text,encoding='utf-8')
    for method,folder,section_num in [('drop25',DROP,16),('rmsnorm',OUT,17)]:
        text=(folder/'REPORT_ZH.md').read_text(encoding='utf-8')
        section='## '+str(section_num)+'. '+text.partition('## '+str(section_num)+'. ')[2]
        version=VERSION/(method+'_v1')
        section=base.rebase(section,folder/'REPORT_ZH.md',version)
        section=re.sub(r'### '+str(section_num)+r'\.(\d+)\.',r'## \1.',section)
        section=section.replace('## '+str(section_num)+'. ','# ',1)
        (version/'README.md').write_text(section,encoding='utf-8')
        evidence_record={'status':'interpreted_bilingual_draft_final_check_required',
           'report_complete':False,'manual_proofreading_complete':False,'goal_achieved':False,
           'data_modified':False,'baseline_refit':False,'complete_test_opened':False,
           'document_hashes':{f'REPORT_{l}.md':sha(folder/f'REPORT_{l}.md') for l in ['ZH','EN']},
           'input_hashes':{f'reports/v9_r9_{method}_{part}_v1/summary.json':sha(ROOT/f'reports/v9_r9_{method}_{part}_v1/summary.json') for part in ['analysis','fit','axis_changes','cases']},
           'script_sha256':sha(Path(__file__))}
        write_json(folder/'evidence.json',evidence_record)
    print(json.dumps({'status':'interpretations_written_final_check_required','reports':[str(DROP),str(OUT)]}))


if __name__=='__main__':main()
