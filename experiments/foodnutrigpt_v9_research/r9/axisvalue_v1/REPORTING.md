# 轴×数值残差：完成后的双语证据草稿

`scripts/report_foodnutrigpt_r9_axisvalue.py`等待原训练、独立重放、11项比较、分析和完整训练拟合全部成功，才生成新快照`report_snapshot_r9_axisvalue_v1`。现有MSE综合报告的第1–10节作为已审阅历史保留，第11节按八节研究记录加入本轮证据。

新表格覆盖三项任务、142/45/187轴、原单位/旧log/正值/显式零、两个可见比例下的检索指标、食品组配对区间、误差贡献、训练拟合及梯度裁剪。双方同种子、共同初值与新增48,384参数分别说明，不误称整模型参数或哈希相同。

格式预检已完成：10张新增共享表格逐项核对、八节结构、旧报告哈希及链接重定位通过；NaN/Inf拒绝。预检在内存中显式复用MAE控制作为格式占位，不读取当前候选结果、不执行模型、不生成结果报告，详见[预检凭据](../../../../reports/v9_r9_axisvalue_report_preflight_v1/verification.json)。

```powershell
.venv/Scripts/python.exe -X utf8 scripts/report_foodnutrigpt_r9_axisvalue.py --check-only
.venv/Scripts/python.exe -X utf8 scripts/report_foodnutrigpt_r9_axisvalue.py --output-dir experiments/foodnutrigpt_v9_research/report_snapshot_r9_axisvalue_v1
```

当前只执行了编译、格式预检及依赖检查，后者正确返回未齐备。正式生成入口核对证据哈希及共同验证分数；输出仍标记草稿。实际曲线查看、逐轴/来源和案例解释、因果边界、版本决定和中英文通读均须随后完成。不能把自动生成等同于审阅或模型接受。
