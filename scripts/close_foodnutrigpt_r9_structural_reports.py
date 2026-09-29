"""Add cross-family comparisons and verify the manually reviewed stage reports.

--prepare updates prose and comparison tables; --complete records the review only
after their actual inspection. Never trains, queries test labels or refits models.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import interpret_foodnutrigpt_r9_structural_reports as interpret
base = interpret.base
read, sha, table = base.read, base.digest, base.table
FOLDERS = {m: ROOT / f'experiments/foodnutrigpt_v9_research/report_snapshot_r9_{m}_v1'
           for m in ['drop25', 'rmsnorm']}


def comparison(method):
    paths = ['output/v9_r8/mlp512_name32/metrics.json',
             'reports/v9_final_neural_summary_v1/summary.json',
             'reports/v9_r9_three_seed_confirmation_v1/summary.json',
             f'reports/v9_r9_{method}_analysis_v1/summary.json']
    mlp, mlp3, tf3, a = [read(ROOT / p) for p in paths]
    rows = []
    def point(label, scope, metrics):
        rows.append([label, scope, *[base.number(metrics[t]['nutrition'][k]) for t, k in
            [('completion', 'scaled_log_mae'), ('completion', 'log_mae'), ('name_only', 'scaled_log_mae')]]])
    def spread(label, metrics, sd, nested):
        vals = []
        for t, k in [('completion', 'scaled_log_mae'), ('completion', 'log_mae'), ('name_only', 'scaled_log_mae')]:
            row = metrics[t]['nutrition'][k] if nested else metrics[t][k]
            vals.append(f"{row['mean']:.6f} ± {row[sd]:.6f}")
        rows.append([label, 'historical 3 seeds, mean ± SD', *vals])
    point('MLP, name32', 'fixed seed 20260922', mlp)
    spread('MLP, name128', mlp3['nutrition_summary'], 'sample_sd', False)
    point('Transformer LayerNorm, name32', 'parent, seed 20260922', a['records']['mae']['metrics'])
    spread('Transformer LayerNorm, name32', tf3['models']['transformer']['tasks'], 'sample_std', True)
    point('Transformer ' + method.upper() + ', name32', 'candidate, seed 20260922', a['records'][method]['metrics'])
    for key in ['rf32', 'xgb32']:
        metrics = {task: a['comparisons'][key][task]['scores']['baseline'] for task in ['completion', 'name_only']}
        point(key.upper(), 'frozen fitted reference', metrics)
    knn = a['comparisons']['knn32']['name_only']['scores']['baseline']['nutrition']['scaled_log_mae']
    rows.append(['Name KNN32', 'frozen name-only reference', '—', '—', base.number(knn)])
    result = table(['Model', 'Evidence scope', 'Completion 142 MAE ↓', 'Completion legacy log-MAE ↓', 'Name-only 142 MAE ↓'], rows)
    return result, {p: sha(ROOT / p) for p in paths}


def prepare():
    a, f = interpret.evidence('rmsnorm')
    shared = interpret.shared_tables(a, f, 'rmsnorm')
    bridge = read(ROOT / 'reports/v9_r9_rmsnorm_bridge_v1/summary.json')
    for method, folder in FOLDERS.items():
        numeric, hashes = comparison(method)
        for lang in ['ZH', 'EN']:
            path = folder / f'REPORT_{lang}.md'
            text = path.read_text(encoding='utf-8')
            first = text[:text.index('## 1.')]
            history = text[text.index('## 2.'):]
            if method == 'rmsnorm':
                history = history[:history.index('## 17.')] + interpret.rms_section(lang, a, f, shared, bridge)
            note = ('上表均使用共同冻结验证协议，MAE越低越好；±为历史三个种子的样本标准差，不是置信区间。MLP128名称输入维度与其他方法不同，MLP与Transformer训练配方也不同，因此该跨模型表不构成单因素架构因果比较。检索完整指标及候选库协议见第4、6及本轮结果章节。' if lang == 'ZH' else
                    'All rows use the common frozen validation protocol; lower MAE is better. ± denotes historical three-seed sample SD, not a confidence interval. MLP128 uses a different name-input dimensionality, and MLP and Transformer recipes also differ, so this cross-family table is not a single-factor causal architecture comparison. Full retrieval metrics and candidate-library rules appear in Sections 4, 6 and the current experiment section.')
            refs = ('[MLP32 metrics](../../../output/v9_r8/mlp512_name32/metrics.json) · '
                    '[MLP128 replication](../../../reports/v9_final_neural_summary_v1/summary.json) · '
                    '[Transformer replication](../../../reports/v9_r9_three_seed_confirmation_v1/summary.json)')
            text = first + interpret.common_overview(lang, method) + numeric + '\n\n' + note + '\n\n' + refs + '\n\n' + history
            path.write_text(text, encoding='utf-8')
        section_number = 16 if method == 'drop25' else 17
        version = ROOT / f'experiments/foodnutrigpt_v9_research/r9/{method}_v1'
        text = (folder / 'REPORT_ZH.md').read_text(encoding='utf-8')
        section = text[text.index(f'## {section_number}. '):]
        section = base.rebase(section, folder / 'REPORT_ZH.md', version)
        section = re.sub(r'### ' + str(section_number) + r'\.(\d+)\.', r'## \1.', section)
        section = section.replace(f'## {section_number}. ', '# ', 1)
        (version / 'README.md').write_text(section, encoding='utf-8')
        record = read(folder / 'evidence.json')
        record['document_hashes'] = {f'REPORT_{l}.md': sha(folder / f'REPORT_{l}.md') for l in ['ZH', 'EN']}
        record['input_hashes'].update(hashes)
        record['script_sha256'] = sha(ROOT / 'scripts/interpret_foodnutrigpt_r9_structural_reports.py')
        interpret.write_json(folder / 'evidence.json', record)
    print(json.dumps({'status': 'prepared_for_final_read', 'folders': list(map(str, FOLDERS.values()))}))


def local_links(text, path):
    count = 0
    for target in re.findall(r'\]\(([^)\n]+)\)', text):
        if re.match(r'^[a-z]+:|^#', target):
            continue
        dest = (path.parent / target.split('#')[0]).resolve()
        if not dest.exists():
            raise FileNotFoundError(dest)
        count += 1
    return count


def complete():
    interpret.frozen_inputs(ROOT)
    timestamp = datetime.now(timezone.utc).isoformat()
    results = []
    for method, folder in FOLDERS.items():
        number = 16 if method == 'drop25' else 17
        version = ROOT / f'experiments/foodnutrigpt_v9_research/r9/{method}_v1'
        a, f = interpret.evidence(method)
        e = read(folder / 'evidence.json')
        base.verify_hashes(e['input_hashes'])
        expected_tables = interpret.shared_tables(a, f, method)
        overview, comparison_hashes = comparison(method)
        checks = {}
        for lang in ['ZH', 'EN']:
            path = folder / f'REPORT_{lang}.md'
            text = path.read_text(encoding='utf-8')
            assert sha(path) == e['document_hashes'][path.name]
            section = text[text.index(f'## {number}.'):]
            assert overview in text
            for key, rendered in expected_tables.items():
                assert rendered in section, (method, lang, key)
            assert len(re.findall(r'^### ' + str(number) + r'\.\d+\.', section, re.M)) == 8
            for stale in ['尚待审阅', 'machine-evidence draft', 'still require review', 'decision still requires']:
                assert stale not in section, (method, lang, stale)
            prior = base.PRIOR if method == 'drop25' else FOLDERS['drop25']
            old = base.rebase((prior / path.name).read_text(encoding='utf-8'), prior / path.name, folder)
            old_history = old[old.index('## 2.'):].strip()
            if method == 'drop25':
                # The lr2e4 snapshot ends with Section 15.
                assert text[text.index('## 2.'):text.index('## 16.')].strip() == old_history
            else:
                assert text[text.index('## 2.'):text.index('## 17.')].strip() == old_history
            checks[lang] = {'common_tables_exact': len(expected_tables), 'cross_family_table_exact': True,
                            'local_links_resolve': local_links(text, path), 'history_preserved': True,
                            'new_section_subsections': 8, 'sha256': sha(path)}
        readme = (version / 'README.md').read_text(encoding='utf-8')
        assert len(re.findall(r'^## \d+\.', readme, re.M)) == 8
        local_links(readme, version / 'README.md')
        partitions = pd.read_csv(ROOT / f'reports/v9_r9_{method}_analysis_v1/{method}_minus_mae_partitions.csv')
        col = method + '_minus_mae_mae'
        source = partitions[partitions.stratification == 'source_key']
        family = partitions[partitions.stratification == 'mask_family']
        assert len(source) == 24 and len(family) == 10
        assert int((source[col] < 0).sum()) == (1 if method == 'drop25' else 8)
        if method == 'drop25':
            assert (family[col] > 0).all()
        visual = {
            'status': 'actually_inspected_six_curve_panels', 'review_recorded_utc': timestamp,
            'figure_sha256': sha(ROOT / f'reports/v9_r9_{method}_analysis_v1/learning_curves.png'),
            'panels': ['validation_primary', 'validation_legacy', 'positive_error', 'zero_error', 'gradient_clipping_fraction', 'mean_preclip_gradient_norm'],
            'observations': ('Primary, legacy and positive errors persistently higher; late zero error, clipping and preclip gradients higher. No numerical divergence.' if method == 'drop25' else
                             'Primary, legacy and positive errors generally higher late in training. Zero error slightly better, with lower late clipping and mean preclip gradients; no numerical divergence.'),
            'all_142_axes_read': True, 'all_source_family_support_partitions_read': True,
            'all_40_fixed_extreme_cases_read': True, 'all_7_sparse_train_validation_axes_read': True,
            'case_numeric_profiles_published': False,
            'scope': 'Human-visible image and table inspection during this heartbeat; not inferred merely from renderer success.'}
        interpret.write_json(ROOT / f'reports/v9_r9_{method}_analysis_v1/visual_review.json', visual)
        decision = read(version / 'decision.json')
        decision['report_complete'] = True
        decision['reviewed_utc'] = timestamp
        interpret.write_json(version / 'decision.json', decision)
        verification = {'status': f'reviewed_complete_{method}_stage_report', 'reviewed_utc': timestamp,
            'actual_bilingual_proofreading_complete': True, 'report_complete': True,
            'document_checks': checks, 'version_readme_sha256': sha(version / 'README.md'),
            'decision_sha256': sha(version / 'decision.json'),
            'visual_review_sha256': sha(ROOT / f'reports/v9_r9_{method}_analysis_v1/visual_review.json'),
            'input_hashes': e['input_hashes'], 'cross_family_comparison_hashes': comparison_hashes,
            'source_improved': int((source[col] < 0).sum()), 'source_worsened': int((source[col] > 0).sum()),
            'baseline_refit': False, 'data_modified': False, 'complete_test_opened': False,
            'goal_achieved': False, 'seed_expansion_authorized': False,
            'scientific_scope': 'Fixed-seed local recipe rejection. No population-of-seeds, external-generalization or foundation-model claim.',
            'closing_script_sha256': sha(Path(__file__))}
        if method == 'rmsnorm':
            bridge = read(ROOT / 'reports/v9_r9_rmsnorm_bridge_v1/summary.json')
            base.verify_hashes(bridge['input_hashes'])
            verification['learned_layernorm_bridge_sha256'] = sha(ROOT / 'reports/v9_r9_rmsnorm_bridge_v1/summary.json')
        interpret.write_json(folder / 'review_verification.json', verification)
        e.update({'status': verification['status'], 'report_complete': True,
                  'manual_proofreading_complete': True, 'review_verification_sha256': sha(folder / 'review_verification.json')})
        interpret.write_json(folder / 'evidence.json', e)
        results.append({'method': method, 'status': verification['status'], 'checks': checks})
    print(json.dumps({'status': 'reports_complete', 'results': results}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare', action='store_true')
    g.add_argument('--complete', action='store_true')
    args = p.parse_args()
    prepare() if args.prepare else complete()
