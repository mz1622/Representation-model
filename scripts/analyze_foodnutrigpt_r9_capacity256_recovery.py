"""Use the outcome-blind capacity analyzer with the separately audited recovery run."""
from pathlib import Path
import analyze_foodnutrigpt_r9_capacity256 as analysis

analysis.RUNS['capacity256'] = 'output/v9_r9_methods/tf256_mae_lr3e4_60_recovery1'
original_requirements = analysis.requirements


def requirements():
    expected, pending = original_requirements()
    extras = {
        analysis.ROOT/'reports/v9_r9_capacity256_lr3e4_60_audit_v1/recovery_verification.json':
            'complete_registered_capacity256_recovery_audit',
        analysis.ROOT/'experiments/foodnutrigpt_v9_research/r9/capacity256_v1/recovery1/config.json':
            'registered_operational_recovery',
        Path(__file__): None}
    for name in ['audit_foodnutrigpt_r9_capacity256_recovery.py',
                 'diagnose_foodnutrigpt_r9_capacity256_recovery_fit.py',
                 'review_foodnutrigpt_r9_capacity256_recovery_cases.py',
                 'report_foodnutrigpt_r9_capacity256_recovery.py']:
        extras[analysis.ROOT/'scripts'/name] = None
    for path, status in extras.items():
        reason = 'absent' if not path.exists() else None
        if reason is None and status and analysis.read(path).get('status') != status:
            reason = 'incomplete_recovery_evidence'
        if reason:
            pending.append({'path': path.relative_to(analysis.ROOT).as_posix(), 'reason': reason})
    expected.update(extras)
    return expected, pending


analysis.requirements = requirements
if __name__ == '__main__':
    analysis.main()
