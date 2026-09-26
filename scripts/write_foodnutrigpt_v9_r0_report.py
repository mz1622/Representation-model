"""Publish the local R0 research record from completed numerical artifacts."""
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import write_json,digest

def read(path):return json.loads(path.read_text(encoding="utf-8"))

def main():
    directory=ROOT/"experiments/foodnutrigpt_v9_research/r0"
    destination=directory/"README.md"
    if destination.exists():raise FileExistsError(destination)
    analysis=ROOT/"reports/v9_r0_analysis_v1"
    runs=read(analysis/"run_inventory.json")
    if any(r["status"]!="complete" for r in runs):raise ValueError("All attempted model runs must finish or have an explicit failure report before closing this exploration.")
    rows=pd.read_csv(analysis/"results.csv")
    assert "rf200_quarantined_completion" in set(rows.run)
    assert "v8_optimized20_quarantined" in set(rows.run)
    intervals=read(analysis/"paired_intervals.json")
    sensitivity=read(analysis/"sensitivity.json")
    diagnostics=read(ROOT/"reports/v9_r0_diagnostics_v1/diagnostics.json")
    retrieval={name:read(ROOT/f"output/v9_r0/{name}/metrics.json") for name in ["retrieval_name_mlp_v1","retrieval_ridge_v1"]}
    source=read(ROOT/"reports/v9_r0_sources_v1/source_probes.json")
    support=pd.read_csv(ROOT/"reports/v9_r0_diagnostics_v1/support.csv")
    data=read(ROOT/"data/processed/foodnutrigpt_v9_r0_v1/manifest.json")
    reproducibility=read(analysis/"reproducibility.json")
    summary={"version":"V9-R0","status":"exploration_complete_confirmation_incomplete","milestone_reached":False,
        "model_results":rows.to_dict("records"),"paired_intervals":intervals,"retrieval":retrieval,"source_probes":source,
        "sensitivity":sensitivity,"diagnostics":diagnostics,"complete_test_opened":False,
        "api_checks":read(ROOT/"reports/v9_r0_api_checks.json"),
        "model_artifact_hashes":{str(f.relative_to(ROOT)):digest(f) for pattern in ["output/v9_r0/*/*.pt","output/v9_r0/*/ridge.npz"] for f in ROOT.glob(pattern)}}
    write_json(directory/"results_summary.json",summary)
    table="| 配置 | 任务 | 视图 | 142轴主指标↓ | 原log-MAE↓ | 正值误差↓ | 显式零误差↓ |\n|---|---|---|---:|---:|---:|---:|\n"
    for r in rows.to_dict("records"):
        table+=f"| {r['run']} | {r['mode']} | {r['view']} | {r['scaled_log_mae']:.6f} | {r['log_mae']:.6f} | {r['positive_scaled_log_mae']:.6f} | {r['zero_scaled_log_mae']:.6f} |\n"
    ci="| 与 XGB300 比较的补全模型 | 主指标相对改善 | 食品组配对95%区间 |\n|---|---:|---:|\n"
    for name,result in intervals.items():
        x=result['scaled_log_mae'];lo,hi=x['relative_improvement_95_interval']
        ci+=f"| {name} | {100*x['relative_improvement']:.2f}% | [{100*lo:.2f}%, {100*hi:.2f}%] |\n"
    retr="| 检索基线 | 可见营养保留比例 | Recall@1 | Recall@5 | Recall@10 | MRR |\n|---|---:|---:|---:|---:|---:|\n"
    for name,result in retrieval.items():
        for m in result["metrics"]:
            retr+=f"| {name} | {m['visible_fraction']:.0%} | {m['recall_at_1']:.4%} | {m['recall_at_5']:.4%} | {m['recall_at_10']:.4%} | {m['mrr']:.6f} |\n"
    seconds=sum(r['elapsed_seconds'] for r in runs)
    costs="| 运行 | 分钟 | 验证选择epoch |\n|---|---:|---:|\n"
    for r in runs:costs+=f"| {Path(r['directory']).name} | {r['elapsed_seconds']/60:.2f} | {r.get('best_epoch','—')} |\n"
    text=f"""# V9-R0：共同协议与第一轮探索记录

**版本决定：探索完成，确认未完成；保持“研究中”。没有达到超越充分调参 RF/XGBoost 的里程碑。**

本轮完成了数据视图、共同评分、12 个模型配置的探索（含检索映射）、接口与功能验收。原始 FooDB 证据、充分调参、共同训练任务消融及三种子确认仍未完成，因此 R0 的科学冻结门槛尚未通过；R1–R6 未宣称完成。本轮没有打开历史测试集进行模型选择或评价。

## 1. 研究问题与预先假设

主要问题是：已有模型差距有多少来自数据与比较流程，剩余差距应先从哪里研究？预先假设是单位异常、重复标签聚合、文本字段和标签存在性输入可能使比较失真。若只改变评分标签即可显著改变差距，评分流程是重要解释；若共同协议下差距仍在，则应继续做训练和输出目标的受控实验。

方案、种子、预算和后续门槛见 [research_config.json](research_config.json)。主指标是在任何本轮模型分数出来前写入并哈希冻结的 142 个营养轴宏平均 `MAE(log1p(y/s_a))`；另外 45 个 metabolome 轴、187 轴历史指标和正值/零值误差单列。65 个 context-only 轴不宣称可靠预测。

## 2. 父版本、改动与控制

父提交为 `1289d38129dffb7d3490239fb516328fa5c905e3`。冻结 V8 包的 10 个文件 SHA256 已在构建时和真实推理验收时复核一致。新视图为 `foodnutrigpt_v9_r0_v1`，只含 64,700 个训练档案和 11,175 个验证档案，共 {data['source_tokens']:,} 条原始观测、{data['canonical_cells']:,} 个档案×营养单元。训练与验证的完全同名候选组隔离通过。

档案内使用原始 g/100g 的中位数；原始观测保留，跨来源标签不自动合并。训练正值典型尺度由候选组内来源等权的加权中位数拟合，验证与测试不参与。评分先在候选组×轴×来源内取档案中位数，再计算来源误差并等权平均，最后按候选组及营养轴宏平均。

隔离视图移除 {data['quarantine_cells']} 个疑似尺度冲突单元，其中训练 {data['training_quarantine_cells']} 个、验证 {data['quarantine_cells']-data['training_quarantine_cells']} 个，涉及 {data['quarantine_observations']} 条原始记录；所有标记来自 FooDB。规则只检查同档案同轴正值极值是否相差 10³/10⁶ 倍，不是完整错误检测器，也不判断哪条正确。含异常的视图另存；两视图使用相同的隔离训练尺度，避免敏感性分析同时改变标尺。

名称仅取 `original_name`，重新生成固定 MiniLM revision 的缓存；原 384 维向量保留，本轮各模型共同使用仅在训练集拟合的 32 维 PCA（解释方差约54.52%）。这是一项容量限制，不能将本轮名称结果视为编码器能力上限。文本内容、训练划分、模型修订和缓存文件均有指纹。

评价标签、家族遮蔽、查询轴、可见上下文和评分权重由统一模块拥有。Transformer 输入固定 252 轴网格，未观测与隐藏位置均采用同一遮蔽值；标签存在性不决定查询 token，避免旧可变 token 列表和256上限的影响。原训练集中有 {data['train_profiles_above_legacy_256_cap']} 个档案超过该上限。

**仍需控制的训练差异：** 树模型逐目标家族训练；神经网络随机遮蔽约30%已观测家族。两者评价完全相同，训练任务采样不完全相同。树模型按轴拟合，神经网络共享多轴目标，优化器及预算也不同。因此本轮是可比评价的起点，不是纯架构因果实验。V8 优化配置同时采用已有的20 epoch和amount=3，不能与V9的差异作单因素归因。

## 3. 可复现信息与计算成本

实际执行命令和三类接口示例见 [REPRODUCE.md](REPRODUCE.md)。所有筛选训练使用 seed `20260922`；尚未运行确认种子 `20260923、20260924`。GPU为 RTX 5070 Ti 16 GB，Python3.10.19，PyTorch2.7.1+cu128；树模型使用全部合格训练行，没有历史每轴5000行上限。

本轮实现的本地代码提交：`{reproducibility['code_commit']}`。运行期间使用的具体源码哈希另保存在各run manifest；报告在实现提交之后生成。

本轮RF预算为1个配置（200树），XGBoost补全预算为2个配置（300树深6、600树深4），另有name-only配置。**这仍不是充分调参的最终强基线。** 本轮12个模型配置达到筛选候选上限，确认流程需在后续新版本登记预算。

数据 manifest SHA256：`{digest(ROOT/'data/processed/foodnutrigpt_v9_r0_v1/manifest.json')}`。
评分协议 SHA256：`{data['protocol_sha256']}`。
验证任务清单 SHA256：`{data['artifact_hashes']['quarantined_validation_jobs.parquet']}`。

父提交、未提交改动清单、代码哈希、实际依赖、所有运行配置和模型哈希保存在本地 `reports/v9_r0_analysis_v1/reproducibility.json`、`run_inventory.json` 及本目录 `results_summary.json`。数值预测、原始值和模型文件不进入公开实验文档。各运行耗时之和约 {seconds/3600:.2f} 小时；任务有并行，此数不是总墙钟时间或GPU小时。

{costs}

## 4. 完整结果与不确定性

以下均为验证集单种子探索，越低越好；原始单位 MAE、45轴结果、187轴结果及全部逐轴指标另存机器可读文件。name-only MLP 不使用营养上下文，因此其 completion 分数与 name-only 相同；numeric MLP 的 name-only 行只是“空输入”对照，不表示它能理解名称。

{table}

配对重采样每次抽取整个食品候选组，组内全部营养轴一起保留，重新计算逐轴宏平均；1000次，seed20260922。正的相对改善表示优于XGB300。这些区间只反映给定模型的验证食品采样波动，**不包含训练种子波动、标签错误和调参选择偏差**，不能用于最终优越性声明。原log-MAE区间也已存入 `results_summary.json`。

{ci}

检索固定49,913个候选名称，包含验证未见名称的文本，但不使用候选真实营养档案。查询营养分支不接收名称。两种基线分别为“名称MLP预测营养后匹配”及“营养到名称PCA空间的Ridge映射”。仅对至少3个已观测营养轴的查询评分；完整输入10,479个档案，30%保留输入8,610个档案。因此两个保留比例的差异包含查询覆盖变化，不能直接当作缺失比例的纯因果效应。

{retr}

正确答案仅按完全相同原名匹配；尚无经核实的别名映射，语义等价名称可能被当作错误。营养本身也未必唯一识别食物名称。上述结果是严格原名检索的起点，不是开放名称生成能力的证明。

逐轴支持数和显式零/正值支持保存在 `reports/v9_r0_diagnostics_v1/support.csv`；187个监督轴中，验证候选组支持少于30的轴有 {int((support.candidate_support<30).sum())} 个，最少 {int(support.candidate_support.min())} 个。分来源结果在 `reports/v9_r0_analysis_v1/source_metrics.csv`，各来源覆盖的轴不同，不能直接作来源质量排名。

## 5. 机制诊断与失败记录

- **单位追溯未闭环。** Goose fat/Cholesterol 同档案疑似1000倍冲突确实存在。但当前 `normalize_unit` 对 `mg/100 g` 和 `mg/100g` 都给出0.001转换因子，不能把空格认定为根因。发布包没有对应原始数值字段/完整转换历史；FooDB公开下载返回403，目前缺原始 `Content.csv` 或作者 staging。没有凭常识改值。
- **重复聚合不同。** 训练营养单元中4194个、验证营养单元中700个在原始中位数与逆log中位数规则下不同。保持本轮预测不变、只换评分标签后，XGB300主指标约从0.191561变为0.191564，V9约从0.309307变为0.309304。此局部评分差异无法单独解释目前差距；训练标签改变的效应未由该诊断隔离。
- **Hurdle检查。** 在同一个V9检查点上去掉存在概率乘法，正值误差从约0.4116降为0.3926，但显式零误差从0.2178升为0.4848，总体主指标从0.3093升为0.4245。支持进一步研究校准和输出损失，不支持直接取消概率乘法。此为事后诊断，未登记为获胜配置。
- **来源信息。** 名称PCA的来源预测平衡准确率约34.92%，仅观测模式约97.17%，随机平衡基准约4.17%。来源可预测并不证明泄漏；数据库采集规则、地域和食物覆盖仍混杂。
- **近名称候选。** 找到736对跨划分高相似名称，没有自动合并。样例既包括词序调整的nori/seaweed，也包括不同水稻品种编号、3小时/6小时处理等真实差异；字符相似度不能替代身份审查。翻译别名、转载关系和完整化学家族边界尚未核实。
- **缺失和显式零。** 缺失不参加损失，显式零仍参加；修改隐藏标签不改变模型输入及预测；V9来源不进入编码器；NaN/Inf使运行失败；保存重载保持预测一致。新增21项验收测试全部通过，并通过真实检查点的任意监督轴查询、三种表征及检索接口检查。
- **原仓库测试。** 整体为128通过、1失败、10错误。未通过项来自缺失的历史分类JSON和历史轴registry夹具，未伪造数据以让测试变绿；详见本地测试日志。开发期间还修复了测试命令未设PYTHONPATH、示例营养名称未用规范名的问题。所有本轮模型训练均正常结束。

所有epoch学习曲线保存在各运行的 `history.csv`；若安装matplotlib，分析脚本同时生成验证曲线图。包含/排除视图的中位数基线敏感性结果已保存，使用相同尺度并报告共同候选单元的误差；它不是“纠正后真实标签”的因果估计。异常尾部、零比例及单位/basis逐来源审计在 `unit_basis_audit.csv` 和数据视图审计文件中；泛化和数据可信性尚不能据此全部验收。

## 6. 因果分析

**已观察事实：** 本轮共同验证协议下，V9未超过XGBoost；名称近邻优于本轮名称MLP和名称XGBoost；简单数值/融合MLP可建立有竞争力的起点；来源观测模式非常可预测；直接去掉hurdle概率乘法损害总体误差。

**受固定输入/预测对照支持的解释：** 仅替换当前验证聚合标签的影响很小；hurdle概率乘法确实改变正值与零值间的取舍；来源ID本身不影响V9编码器输出。这些结论仅覆盖相应操作，不能推出历史全部差距的原因。

**仍未排除：** 训练家族采样和可见信息量不同、PCA压缩、优化时长及学习率日程、多轴梯度竞争、稀疏监督、源表达偏移、近似名称重叠、未解释的单位/basis问题。V8优化配置与V9同时存在多项差异，不能把其结果归因于来源残差或amount权重。尚无证据支持“更大Transformer必然更好”或“简单回归头已经解决问题”。

## 7. 版本决定

接受共同数据/评价接口、名称缓存隔离、显式零语义、固定查询和本轮记录机制作为后续研究基础；保留全部候选结果和失败记录。**不接受任何模型为已确认改进，不宣称nutrition foundation model已建立。**

R0科学冻结仍需原始FooDB证据或有充分依据的研究范围限定、近名称/来源转载审查、共同训练任务对照和更充分的树模型调参。任何数据/划分/指标更改都必须新建版本并重算基线。本轮没有运行三种子确认，不能把食品组bootstrap替代种子波动。来源留出、类别留出和少样本迁移尚未执行。

## 8. 下一轮问题

1. 优先补齐FooDB `Content.csv` / staging的原始数值、单位、转换链路和basis；若能明确错误，生成独立修订视图及逐条变更清单，重新跑全部基线。若证据仍不足，保持隔离视图为探索结果。
2. 在明确可研究数据范围后登记R1：同一V9、相同数据/文本/遮蔽和固定学习率日程下比较8与20 epoch，再分别比较amount=1/2/3，最后比较source residual开/关；来源项使用 `(base+w*calibrated)/(1+w)` 控制整体损失尺度。延长训练时不能同时偷改学习率日程；本轮cosine周期随总epoch变化，需在R1显式固定。
3. 加入共同训练家族任务清单消融，以及主指标选点与历史hurdle选点对照；补足RF/XGBoost调参。收益明确后，再用3固定种子和配对食品组区间确认。
4. 若R1仍显示数值瓶颈，进入同架构直接缩放回归头的R2；不能用当前不同MLP架构之间的差异代替该消融。名称任务同时按近邻相似度分层，并核验32维压缩是否丢失关键语义。
5. R3–R6按后续证据进入：混合缺失模式/名称任务比例，表征探针与少样本迁移，名称—营养对比学习，来源/类别留出。当前历史测试集继续关闭；最终泛化需另行登记独立来源或外部数据。

机器可读汇总：[results_summary.json](results_summary.json)。冻结协议与运行命令：[REPRODUCE.md](REPRODUCE.md)。
"""
    destination.write_text(text,encoding="utf-8")
    print(destination)

if __name__=="__main__":main()
