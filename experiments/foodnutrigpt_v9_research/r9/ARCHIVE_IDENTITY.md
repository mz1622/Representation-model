# 原始下载包的一致性核验 / Original archive identity

本次直接数据来源是用户提供的 `foodnutrigpt_v8_source_native_baseline.tar.gz`。
运行中的 R9 实验继续使用原有冻结研究视图。本核验只读取文件字节和清单，
没有解压落盘、解析食品或营养观测行、修改数据、重拟合变换、训练模型或评价测试集。

The immediate data source is the user-supplied `foodnutrigpt_v8_source_native_baseline.tar.gz`.
The running R9 experiments continue to use the existing frozen research view.
This check reads opaque file bytes and manifests. It does not extract files to disk,
parse food or nutrient rows, change data, refit transformations, train a model,
or evaluate the test set.

| 项目 / Item | 实际结果 / Observed result |
|---|---|
| Archive size | 135,711,396 bytes |
| Archive SHA-256 | `83ee2b50f04963943797aa1818a91f66d0ffaeb7a5bc3c5b2db7a11be6dff792` |
| Dataset version | `global_foodnutrigpt_v8_single_stage_v2_complete_test` |
| Payload files | 10 |
| Archive payload vs embedded release manifest | All ten matched |
| Archive payload vs local immutable V8 files | All ten matched |
| Archive payload vs R0 registered original-input fingerprints | All ten matched |

机器可读证据包含每个文件的路径、长度和 SHA-256：
[verification.json](../../../reports/v9_r9_original_archive_identity_v1/verification.json)。
程序为 [audit_foodnutrigpt_original_archive.py](../../../scripts/audit_foodnutrigpt_original_archive.py)。

The [machine-readable receipt](../../../reports/v9_r9_original_archive_identity_v1/verification.json)
records each payload path, size and SHA-256. The
[verification script](../../../scripts/audit_foodnutrigpt_original_archive.py) is reproducible from the repository root:

```powershell
.venv\Scripts\python.exe scripts/audit_foodnutrigpt_original_archive.py --archive 'C:\Users\Admin\Downloads\foodnutrigpt_v8_source_native_baseline.tar.gz' --output-dir reports/v9_r9_original_archive_identity_v2
```

示例使用新的输出目录；既有核验产物不得覆盖。代码的 `--archive` 参数可接受其他本地路径。
原始数据和数值产物仍留在本地，不进入公开报告。

The example uses a new output directory; existing evidence must not be overwritten.
The `--archive` argument also accepts another local path. Raw data and numerical
prediction artifacts remain local and are not included in public reports.

该结果证明的是收到的数据包、包内清单、本地 V8 与 R0 输入登记之间的字节一致性。
包内校验和并不是独立发布方签名。尚未据此独立确认各上游数据库的准确版本、
授权或逐记录来源；也没有解决 FooDB 单位冲突、跨库转载、别名或剩余标签正确性问题。
不能把此核验当作新的数据版本、完整数据质量通过或额外模型实验。

This establishes byte identity between the received archive, its embedded manifest,
the local V8 payload and the R0 input registry. Embedded checksums are not an
independent publisher signature. Exact upstream releases, licensing and record-level
provenance are not independently established by this check. FooDB unit conflicts,
cross-database copying, aliases and residual label validity remain unresolved.
This is neither a new dataset version, a complete data-quality clearance, nor an
additional model experiment.
