# R0 复现与接口

在仓库根目录运行。Windows 使用 `.venv\Scripts\python.exe`；其他系统将其替换为已安装依赖的 `python`。原始冻结包必须已按项目下载清单安装。

本轮使用 Python 3.10.19、PyTorch 2.7.1+cu128、RTX 5070 Ti 16 GB。其他主要依赖固定在根目录 `requirements-research.txt`。新环境先安装对应平台的 PyTorch（本机使用 CUDA 12.8 wheel），再安装该文件；原仓库 requirements 中的旧 Hugging Face Hub 版本与当前 Transformers 不兼容，不用于本轮。完整实际环境清单见分析目录 `reproducibility.json`。

## 构建与核验

```powershell
.\.venv\Scripts\python.exe scripts/build_foodnutrigpt_v9_r0.py
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv\Scripts\python.exe -m pytest tests/test_research_r0.py tests/test_research_neural.py tests/test_research_statistics.py -q
```

构建器验证冻结包 SHA256，拒绝覆盖已存在的数据视图。发布包的原始观察值保留；疑似尺度冲突只生成排除视图，不改值。`data/processed/foodnutrigpt_v9_r0_v1/manifest.json` 和 `protocol.json` 是实际运行依据。若修订数据处理语义，必须更改版本名并重新构建和运行基线。

仅名称缓存由首次运行自动生成；MiniLM 必须已缓存固定 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。32 维 PCA 仅在训练样本拟合，原始 384 维向量也保留在缓存中。

## 初轮基线命令

以下目录已用于本轮。复现时使用新的输出目录，不覆盖原结果。所有命令默认使用隔离视图和 seed `20260922`。

```powershell
.\.venv\Scripts\python.exe scripts/run_foodnutrigpt_v9_r0_baseline.py --method median --view quarantined --output-dir output/v9_r0/median_quarantined
.\.venv\Scripts\python.exe scripts/run_foodnutrigpt_v9_r0_baseline.py --method median --view inclusive --output-dir output/v9_r0/median_inclusive
.\.venv\Scripts\python.exe scripts/run_foodnutrigpt_v9_r0_baseline.py --method name_knn --mode name_only --output-dir output/v9_r0/name_knn_quarantined
.\.venv\Scripts\python.exe scripts/run_foodnutrigpt_v9_r0_baseline.py --method rf --trees 200 --max-depth 16 --n-jobs 4 --output-dir output/v9_r0/rf200_quarantined_completion
.\.venv\Scripts\python.exe scripts/run_foodnutrigpt_v9_r0_baseline.py --method xgb --trees 300 --max-depth 6 --n-jobs 4 --output-dir output/v9_r0/xgb300_quarantined_completion
.\.venv\Scripts\python.exe scripts/run_foodnutrigpt_v9_r0_baseline.py --method xgb --trees 600 --max-depth 4 --learning-rate 0.03 --n-jobs 4 --output-dir output/v9_r0/xgb600d4_quarantined_completion
.\.venv\Scripts\python.exe scripts/run_foodnutrigpt_v9_r0_baseline.py --method xgb --mode name_only --trees 300 --max-depth 6 --n-jobs 4 --output-dir output/v9_r0/xgb300_quarantined_name_only
.\.venv\Scripts\python.exe scripts/train_foodnutrigpt_v9_r0_neural.py --kind name_mlp --epochs 8 --batch-size 256 --learning-rate 0.001 --output-dir output/v9_r0/name_mlp8_quarantined
.\.venv\Scripts\python.exe scripts/train_foodnutrigpt_v9_r0_neural.py --kind numeric_mlp --epochs 8 --batch-size 256 --learning-rate 0.001 --output-dir output/v9_r0/numeric_mlp8_quarantined
.\.venv\Scripts\python.exe scripts/train_foodnutrigpt_v9_r0_neural.py --kind mlp --epochs 8 --batch-size 256 --learning-rate 0.001 --output-dir output/v9_r0/mlp8_quarantined
.\.venv\Scripts\python.exe scripts/train_foodnutrigpt_v9_r0_neural.py --kind v9 --epochs 8 --batch-size 64 --output-dir output/v9_r0/v9_8_quarantined
.\.venv\Scripts\python.exe scripts/train_foodnutrigpt_v9_r0_neural.py --kind v8_optimized --epochs 20 --batch-size 64 --amount-weight 3 --output-dir output/v9_r0/v8_optimized20_quarantined
```

V8/V9 是原架构在新协议下的重跑，**不等价于历史分数复现**。文本、重复处理、数值变换、轴查询和选点已统一更改。V8 优化配置同时包含已有的 20 epoch 和 amount=3，因此它与 V9 的差异不能归因于某一个因素。

MLP/Transformer 训练随机隐藏约 30% 已观测家族；树模型逐轴隐藏目标家族训练。评分面板完全相同，训练任务采样不同。这一差异必须保留为解释模型差距的替代假设，后续应做共同训练任务清单消融。

## 诊断、检索与统计

```powershell
.\.venv\Scripts\python.exe scripts/audit_foodnutrigpt_v9_r0_sources.py --output-dir reports/v9_r0_sources_v1
.\.venv\Scripts\python.exe scripts/diagnose_foodnutrigpt_v9_r0.py --output-dir reports/v9_r0_diagnostics_v1
.\.venv\Scripts\python.exe scripts/evaluate_foodnutrigpt_v9_r0_retrieval.py --checkpoint output/v9_r0/name_mlp8_quarantined/best_model.pt --output-dir output/v9_r0/retrieval_name_mlp_v1
.\.venv\Scripts\python.exe scripts/evaluate_foodnutrigpt_v9_r0_retrieval.py --method ridge_to_text --output-dir output/v9_r0/retrieval_ridge_v1
.\.venv\Scripts\python.exe scripts/analyze_foodnutrigpt_v9_r0.py --output-dir reports/v9_r0_analysis_v1
```

配对区间以整个名称候选组为抽样单位，组内所有营养轴一起重采样，再重新计算逐轴宏平均。它不包含训练种子方差，也不能补救错误标签。原始预测、疑似错误记录、逐组误差和检索排名只保存在本地忽略目录中。

## 推理

```powershell
.\.venv\Scripts\python.exe scripts/predict_foodnutrigpt_v9_r0.py --checkpoint output/v9_r0/v9_8_quarantined/best_model.pt --request experiments/foodnutrigpt_v9_research/r0/example_request.json --output output/v9_r0/example_prediction.json
```

请求中的名称必须是单个原始食物名称，不能拼接来源或其他元数据。`observed_profile` 是“营养轴规范名：g/100g 数值”的字典；省略缺失项，显式零写成 `0`。`target_axes` 由调用者指定，可查询在该食品上未观测的监督轴；65 个 context-only 轴不能作为已验证预测目标。

Python 接口位于 `foodcomp.research_inference.NutritionModel`：

```python
model.predict(food_name, observed_profile, target_axes)
model.encode(food_name=food_name, observed_profile=profile, modality="fused")
model.encode(observed_profile=profile, modality="nutrition")
model.retrieve_names(observed_profile, candidate_names, top_k=10)
```

`encode` 明确区分 `name`、`nutrition`、`fused`，单模态 MLP 不提供融合表征。Transformer 的返回值是轴 token 的均值，当前仅供探针研究，尚未证明可迁移性。检索接口不接受查询名称；候选向量只由名称预测营养产生。返回分数是负变换空间 MSE，不是概率。当前多正确答案协议仅支持完全相同的原名，尚无经核实的别名映射。
