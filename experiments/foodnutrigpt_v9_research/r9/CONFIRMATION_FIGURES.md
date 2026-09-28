# 三种子学习曲线：正式统计完成后生成

`scripts/plot_foodnutrigpt_r9_confirmation.py`读取完成的三种子统计及其绑定的三个完整训练轨迹，不启动训练、评价或重新选点。所有统计输入哈希、模型配方、60轮任务顺序/曝光、余弦日程及最早严格最小选点都重新核对；三个所选主误差的平均值还须与统计汇总一致。

生成PNG和SVG六面板：187轴校准训练目标、142轴验证主误差、最后20轮主误差放大图、旧log-MAE、正值误差和显式零误差。每条线是一种固定种子，星号始终标记该种子的补全选点；不会分别为正值/零值挑选更好轮次。相关面板显示同协议固定RF/XGB参考，旧log面板同时显示RF+2%保护线。

最后20轮仅是视觉放大，不是新增选点规则；在线训练目标不能与验证主指标直接相减，正值/零值条件误差也不能直接相加。不给三条曲线绘制伪造的置信带；真实种子SD和食品组配对区间由统计入口报告。

```powershell
.venv/Scripts/python.exe scripts/plot_foodnutrigpt_r9_confirmation.py --check-only --output-dir reports/v9_r9_three_seed_curves_v1
.venv/Scripts/python.exe scripts/plot_foodnutrigpt_r9_confirmation.py --output-dir reports/v9_r9_three_seed_curves_v1
```

当前已通过编译与真实依赖预检：统计文件尚未生成，因此正确返回ready=false，没有创建图表目录。这不等于正式绘图或视觉核验通过。待当前训练和已排队统计完成后再执行；实际图片必须查看，并与三种子结果一起解释。生成凭据保留`visual_review_complete=false`，不能由程序代替研究者宣称已经查看图片。已有输出拒绝覆盖。此绘图步骤尚未排队，不要在输入未就绪时运行正式命令。
