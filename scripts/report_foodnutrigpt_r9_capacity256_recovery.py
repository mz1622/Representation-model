"""Include the operational interruption in the prepared bilingual evidence draft."""
from pathlib import Path
import report_foodnutrigpt_r9_capacity256 as report

original_supplement = report.supplement


def supplement(lang, analysis, fit, shared, output):
    content = original_supplement(lang, analysis, fit, shared, output)
    record = analysis['records']['capacity256']['manifest']
    if (record.get('resume_count') != 1 or record.get('original_completed_epochs') != 48
            or record.get('resumed_first_epoch') != 49 or record['numerical_recipe_changed_by_recovery']):
        raise ValueError('Expected unchanged candidate6 with operational recovery disclosed')
    recovery_path = report.ROOT/'experiments/foodnutrigpt_v9_research/r9/capacity256_v1/recovery1'
    note = ('运行中断记录：原进程和会话消失，最后完整保存为48轮，具体退出原因未知。原目录保持原样，在独立恢复目录从49轮继续至60轮；完整恢复模型、优化器、调度器及随机状态，未改方法。上表运行耗时仅为原已保存耗时与恢复段耗时之和，不包括未知的未保存工作或等待时间。' if lang == 'ZH' else
            'Operational interruption: the original processes and session disappeared after the last complete save at epoch 48; the exit cause is unknown. Original files are preserved, and a separate recovery directory continues epochs 49–60 with model, optimizer, scheduler and random states restored and the method unchanged. Run elapsed time above sums the original saved elapsed time and the resumed segment; it excludes unknown unsaved work and downtime.')
    evidence = report.table(['Recovery item', 'Value'], [
        ['Original run', record['original_run']], ['Recovery code commit', record['recovery_code_commit']],
        ['Original saved elapsed (s)', f"{record['original_saved_elapsed_seconds']:.3f}"],
        ['Resumed segment elapsed (s)', f"{record['recovery_segment_elapsed_seconds']:.3f}"],
        ['Resume started (UTC)', record['recovery_started_utc']],
        ['Resume finished (UTC)', record['recovery_finished_utc']]])
    note += '\n\n'+evidence+'\n\n'+f"[Recovery registration]({report.relative(recovery_path/'PLAN.md',output)}) · [Recovery configuration]({report.relative(recovery_path/'config.json',output)})"
    point = '### 13.4.'
    if content.count(point) != 1:
        raise ValueError('Missing section boundary for recovery disclosure')
    return content.replace(point, note+'\n\n'+point, 1)


report.supplement = supplement
if __name__ == '__main__':
    report.main()
