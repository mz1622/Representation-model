$ErrorActionPreference='Stop'
$taskRepo=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../outputs/Representation-model'))
Set-Location -LiteralPath $taskRepo
$taskPython=Join-Path $taskRepo '.venv/Scripts/python.exe'
$taskStatus=Join-Path $PSScriptRoot 'r9-rmsnorm-v1-status.json'
$taskPlan='experiments/foodnutrigpt_v9_research/r9/rmsnorm_v1/config_v2.json'
$taskRun='output/v9_r9_methods/tf192_mae_rmsnorm_lr3e4_60'
if (Test-Path -LiteralPath $taskStatus) { throw 'Controller already registered; do not duplicate.' }
if (Test-Path -LiteralPath $taskRun) { throw 'Output exists; inspect instead of overwriting.' }
$taskActive=Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and ($_.CommandLine -like '*train_foodnutrigpt*' -or $_.CommandLine -like '*resume_foodnutrigpt*') }
if ($taskActive) { throw 'An existing nutrition trainer is live; do not duplicate.' }
if (Test-Path Env:CUBLAS_WORKSPACE_CONFIG) { throw 'Do not change formal CUDA backend.' }
$taskFunctional=Get-Content -LiteralPath 'reports/v9_r9_rmsnorm_functional_v1/verification.json' -Raw -Encoding utf8 | ConvertFrom-Json
if ($taskFunctional.status -ne 'complete' -or -not $taskFunctional.shared_initial_tensors_exact -or -not $taskFunctional.seven_encoder_norms_replaced) { throw 'RMSNorm structural functional preflight required.' }
$taskPlanData=Get-Content -LiteralPath $taskPlan -Raw -Encoding utf8 | ConvertFrom-Json
$taskPrepared=Get-Content -LiteralPath 'reports/v9_r9_rmsnorm_preparation_v1/verification.json' -Raw -Encoding utf8 | ConvertFrom-Json
if ($taskPrepared.status -ne 'prepared_verified' -or $taskPrepared.training_performed -or $taskPrepared.tests_passed -ne 30) { throw 'Verified preparation required.' }
$taskPins=@{}
foreach ($item in $taskPlanData.input_hashes.PSObject.Properties) { $taskPins[$item.Name]=$item.Value }
foreach ($item in $taskFunctional.implementation_hashes.PSObject.Properties) { $taskPins[$item.Name]=$item.Value }
foreach ($item in $taskPrepared.prepared_code_hashes.PSObject.Properties) { $taskPins[$item.Name]=$item.Value }
foreach ($path in @($taskPlan,'reports/v9_r9_rmsnorm_functional_v1/verification.json','reports/v9_r9_rmsnorm_preparation_v1/verification.json')) {
    $taskPins[$path]=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
}
$taskLaunch=[DateTime]::UtcNow
$taskState=@{status='running';stage='registered';launch_utc=$taskLaunch.ToString('o');estimated_training_seconds=$taskFunctional.estimated_training_seconds;first_followup_due_utc=$taskLaunch.AddMinutes($taskFunctional.followup_delay_minutes).ToString('o');followup_automation_id='nutrition';controller_pid=$PID;controller_creation_utc=(Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o');controller_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash;plan=$taskPlan;current_run=$taskRun;pinned_hashes=$taskPins;completed=@();data_modified=$false;baseline_refit=$false;complete_test_opened=$false;goal_achieved=$false;report_complete=$false}
function Save-TaskState {
    $taskState.updated_utc=[DateTime]::UtcNow.ToString('o')
    $taskTemporary="$taskStatus.tmp"
    $taskState | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $taskTemporary -Encoding utf8
    Move-Item -LiteralPath $taskTemporary -Destination $taskStatus -Force
}
function Assert-TaskPins {
    foreach ($path in $taskPins.Keys) {
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $taskPins[$path]) { throw "Pinned input changed: $path" }
    }
}
function Invoke-TaskPython {
    param([string[]]$PythonArguments,[string]$LogPath='')
    Assert-TaskPins
    if ($LogPath) { & $taskPython -X utf8 @PythonArguments *> $LogPath }
    else { & $taskPython -X utf8 @PythonArguments }
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $($PythonArguments -join ' ')" }
    Assert-TaskPins
    $taskState.completed+=($PythonArguments -join ' ')
    Save-TaskState
}
try {
    Assert-TaskPins
    $taskState.stage='rmsnorm_training'
    Save-TaskState
    Invoke-TaskPython -PythonArguments @('scripts/train_foodnutrigpt_v9_r9_rmsnorm.py','--candidate','tf192_mae_rmsnorm_lr3e4_60','--output-dir',$taskRun) -LogPath (Join-Path $PSScriptRoot 'r9-rmsnorm-v1-training.log')
    $taskState.stage='independent_replay'
    Save-TaskState
    Invoke-TaskPython -PythonArguments @('scripts/audit_foodnutrigpt_r9_rmsnorm.py','--run',$taskRun,'--output-dir','reports/v9_r9_rmsnorm_60_audit_v1')
    $taskState.stage='same_seed_and_fixed_reference_comparisons'
    Save-TaskState
    foreach ($reference in @(@{tag='mae_parent';run='output/v9_r9/tf192_mae_lr3e4_60'},@{tag='rf32';run='output/v9_r8/rf400leaf1half_name32'},@{tag='xgb32';run='output/v9_r8/xgb800d10_name32'})) {
        foreach ($mode in @('completion','name_only')) {
            Invoke-TaskPython -PythonArguments @('scripts/compare_foodnutrigpt_research_predictions.py','--baseline',"$($reference.run)/$($mode)_predictions.parquet",'--candidate',"$taskRun/$($mode)_predictions.parquet",'--task',$mode,'--output-dir',"reports/v9_r9_rmsnorm_vs_$($reference.tag)_$($mode)_v1")
        }
        Invoke-TaskPython -PythonArguments @('scripts/compare_foodnutrigpt_research_retrieval.py','--baseline-dir',"$($reference.run)/retrieval",'--candidate-dir',"$taskRun/retrieval",'--output-dir',"reports/v9_r9_rmsnorm_vs_$($reference.tag)_retrieval_v1")
    }
    Invoke-TaskPython -PythonArguments @('scripts/compare_foodnutrigpt_research_predictions.py','--baseline','output/v9_r7/exact_name_knn32/nutrition_predictions.parquet','--candidate',"$taskRun/name_only_predictions.parquet",'--task','name_only','--output-dir','reports/v9_r9_rmsnorm_vs_knn32_name_only_v1')
    Invoke-TaskPython -PythonArguments @('scripts/compare_foodnutrigpt_research_retrieval.py','--baseline-dir','output/v9_r7/exact_name_knn32','--candidate-dir',"$taskRun/retrieval",'--output-dir','reports/v9_r9_rmsnorm_vs_knn32_retrieval_v1')
    $taskState.stage='gap_diagnosis'
    Save-TaskState
    Invoke-TaskPython -PythonArguments @('scripts/diagnose_foodnutrigpt_research_gap.py','--neural-dir',$taskRun,'--tree-dir','output/v9_r8/rf400leaf1half_name32','--tree-prediction-file','completion_predictions.parquet','--local-case-dir','data/local/research_diagnostics/v9_r9_rmsnorm_rf32_gap_v1','--output-dir','reports/v9_r9_rmsnorm_rf32_gap_v1')
    $taskState.stage='analysis_and_full_training_fit'
    Save-TaskState
    Invoke-TaskPython -PythonArguments @('scripts/analyze_foodnutrigpt_r9_rmsnorm.py','--output-dir','reports/v9_r9_rmsnorm_analysis_v1')
    Invoke-TaskPython -PythonArguments @('scripts/diagnose_foodnutrigpt_r9_rmsnorm_fit.py','--output-dir','reports/v9_r9_rmsnorm_fit_v1','--local-dir','data/local/research_diagnostics/v9_r9_rmsnorm_fit_v1')
    $taskState.status='complete'
    $taskState.stage='machine_evidence_complete_actual_review_and_bilingual_reports_required'
    Save-TaskState
} catch {
    $taskState.status='failed'
    $taskState.error_message=$_.Exception.Message
    Save-TaskState
    throw
}

