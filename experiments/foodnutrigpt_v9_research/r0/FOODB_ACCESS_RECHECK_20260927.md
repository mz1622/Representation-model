# FooDB 原始证据访问复核：仍未取得 Content.csv

2026-09-27在R8树训练等待期间复核原始证据入口。本记录是一次外部可访问性核验，不是数据修订，也不关闭R0的标签真实性要求。

- 官方[下载页](https://foodb.ca/downloads)的搜索索引仍列出2020-04-07 CSV包；实际打开下载页及[字段说明页](https://foodb.ca/schema)均返回HTTP403。搜索索引不等同于取得原始文件。
- 2026-09-27 15:37:07 UTC，使用普通、无凭据的流式GET访问官方`foodb_2020_4_7_csv.tar.gz`，实际HTTP403，返回类型为HTML。只读取响应首32字节以确认不是gzip；没有下载或解压数据库。机器凭据在本地`data/local/research_diagnostics/foodb_access_recheck_20260927_v1/archive_request.json`。
- 按Browser技能尝试浏览器入口，工具初始化返回`failed to write kernel assets`及系统路径不存在错误，未建立浏览器连接。没有可据此声称已访问浏览器页面的证据。
- 官方[API说明](https://foodb.ca/api_doc)要求申请实验API key；文档中的示例不是本任务已获授权的个人密钥。本次没有调用认证API，也没有发送申请消息。

代码复核边界：当前`src/foodcomp/sources.py`的FooDB加载器从`orig_content`取原始值、从`orig_unit`取原始单位；旧`prepare_foodb_raw_mass_compounds_v2.py`另有读取`standard_content`并除以1000的路径。这说明仓库存在不同历史处理入口，但**不足以证明冻结V8中的冲突来自字段混用**，也不能确认两条原始记录哪条正确。仍需原始记录和实际staging/转换链路；空格不是已确认根因。

保留既有64个疑似冲突单元隔离规则及全部冻结数据、缓存、权重、划分和模型结果。下一步需要原始Content文件、作者staging或官方访问恢复后逐条匹配，不以其他网站近似食品值替代原证据，也不自动跨来源合并。已经在当前任务中询问是否另有原始文件路径；训练无需等待答复，科学结论继续明确限定于当前隔离视图。
