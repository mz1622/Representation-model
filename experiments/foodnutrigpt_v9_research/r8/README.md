# V9-R8：同名称输入补全实验（MLP阶段完成，树队列运行中）

[预登记](PLAN.md)提交`41340ad`，12项配置及预算见[config.json](config.json)。进入条件为R7所有任务、审计和完整报告结束，并通过已登记名称维度筛选；当前准备工作不代表正式候选已完成。

两个新任务面板已逐项核验与原R1相同的337048任务、1828536监督目标、家族、训练行、标签、来源权重及验证任务。新输入统一632维（128名称槽位＋252数值＋252可见标记），树特征与神经家族特征全量相等。仅名称矩阵有效维度不同。冻结旧面板不变。

|面板|新清单SHA256|
|---|---|
|32有效方向|`3e8714b9d0033372c1eefe96e7536b440c90d2151cb434d3219d45251025549a`|
|128有效方向|`384b66c10142aefa801a5d0fbed2772b2e128150c9bb1575e2c2fe9bc513fbf8`|

路径分别`data/processed/foodnutrigpt_v9_r8_tasks_name{32,128}_v2`；凭据`reports/v9_r8_input_panels_v2/verification.json`。`completion_view`加载时校验新缓存、数据与面板指纹，不绕过旧任务的名称约束。

真实768训练任务功能检查通过：两臂717052参数、初始化哈希同为`5b71bb0daf3f9966ed6bddbd908e4b786c7565c382728f3be1c5542b5dc9f20b`，所有非文本batch字段相等、32前缀相同、32额外方向梯度为零/128非零。隐藏标签/来源不改变预测，非目标值不进入loss，NaN/Inf失败；数值上下文实际影响输出，保存重载及调用者指定未观测轴预测通过。仅32行/50步过拟合诊断loss分别.594729→.028996、.594904→.033083，未查看验证结果或选择模型。凭据`reports/v9_r8_input_functional_v2/verification.json`，诊断检查点明确标记`not_candidate`，不计正式候选结果。

保留两个准备阶段失败：面板v1读取历史`family_names`对象数组时因默认禁用pickle失败；只对哈希已验证、本地产生的任务文件允许读取该数组后，在v2新目录通过。功能审计v1把标量`objective_multiplier`误传给`torch.equal`，按类型比较后v2通过。失败源码/日志及凭据均保留；没有覆盖失败目录，没有R8正式训练失败。

```powershell
.venv/Scripts/python.exe scripts/build_foodnutrigpt_v9_r8_tasks.py
.venv/Scripts/python.exe scripts/audit_foodnutrigpt_completion_input.py --output-dir reports/v9_r8_input_functional_v2
```

以上为实际准备命令；既有目录不得覆盖，重复审计需新目录，重新构建需新版本。完整版本结束时再补齐八节研究报告，不能把这里的预检当实验结果。

正式执行入口现已实现：`train_foodnutrigpt_v9_r8_completion.py`调用原共享训练器，仅载入新的名称缓存和已审计家族面板；`run_foodnutrigpt_v9_r8_trees.py`对同一批632维输入逐轴全数据拟合，并从同一模型产生补全、name-only和检索候选。全部源码和输入在启动前独立冻结；后续文档提交不改变模型执行版本。

43项相关测试通过（8条已有Transformer提示），另12个真实训练部分的小树诊断全部通过；凭据`reports/v9_r8_tree_functional_v1/verification.json`。RF统一使用并行拟合、串行固定树顺序预测；保存/重载在内存中逐轴核验。诊断使用较少行和树数只是功能检查，正式RF/XGB仍使用预登记的全部合格训练行及400/800棵树。执行清单已冻结，正式MLP和树队列均已启动；运行后还须完成独立全量审计与版本结果报告。旧测试始终关闭，原始单位证据仍未闭环。

正式实现提交`6ff6f36d4f32c9c2dbf12eac6d32d0f3c21d6df4`；执行契约`reports/v9_r8_execution_contract_v1/manifest.json`。MLP32→128顺序运行各60轮；另一个CPU队列顺序运行全部八项预登记树配置。每次运行记录冻结实现提交与实际启动时工作区提交，后续文档/独立审计提交不会改变拟合代码。中间分数不作模型胜负结论，八项树尚未全部完成。

已完成[MLP八节阶段报告](MLP_STAGE_RESULTS.md)：两臂60轮及三个任务、完整重放/配对/拟合诊断完成。128相对32的补全主指标改善0.931% [−1.456%,2.991%]，区间跨零；name-only改善18.909%，仍落后同输入近邻。没有接受模型，八项全数据树仍须按登记完成。审计v1路径分隔符不一致失败记录保留，仅统一键表示后v2逐文件哈希和数值重放均通过。
