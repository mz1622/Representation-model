"""Add train-only distribution tables and ECDF plots to the composition report.

Colab: pip install -r requirements-colab.txt
       python scripts/report_composition_distributions.py
       python scripts/finalize_scientific_food_composition_v6_1.py --report-only
This command never changes the dataset, normalizer, targets, splits or masks.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.benchmark import load_release, fit_train_normalizer
from foodcomp.distributions import summarize_compositions
from foodcomp.util import write_csv, write_json, sha256_file
from report_v4_provenance import tex, table

DEFAULT_VERSION = 'scientific_food_composition_v6_1'
REFERENCES = {
    'location': 'https://www.itl.nist.gov/div898/handbook/eda/section3/eda351.htm',
}
CENTER_RTOL = 1e-6
CENTER_ATOL = 1e-12


def number(value):
    if pd.isna(value):
        return '--'
    return f'{float(value):.6g}' if value else '0'


def ecdf_figure(raw, destination):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 1.65))
    fig.subplots_adjust(left=.08, right=.98, bottom=.28, top=.89, wspace=.29)
    for ax, values, label in zip(axes, [raw, np.log1p(raw)], ['g/100 g', 'log1p(g/100 g)']):
        if len(values):
            x, counts = np.unique(values, return_counts=True)
            span = x[-1] - x[0]
            pad = .02 * span if span else max(.02 * abs(x[0]), 1e-10)
            lo, hi = max(0, x[0] - pad), x[-1] + pad
            ax.step(np.r_[lo, x, hi], np.r_[0, counts.cumsum() / len(values) * 100, 100],
                    where='post', color='#46515a', linewidth=1)
            ax.axvline(values.mean(), color='#ab3f59', linewidth=1.1, label='Mean', zorder=4)
            ax.axvline(np.median(values), color='#137f7c', linestyle='--', linewidth=1.1, label='Median', zorder=4)
            ax.set_xlim(lo, hi)
            ax.legend(loc='lower right', fontsize=6.5, frameon=False)
        else:
            ax.text(.5, .5, 'No observed training values', ha='center', va='center', transform=ax.transAxes, fontsize=7)
        ax.set_ylim(-2, 103)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel('ECDF (%)', fontsize=7)
        ax.set_xlabel(label, fontsize=7)
        ax.tick_params(labelsize=6.5, length=2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(6)
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(destination, dpi=190)
    plt.close(fig)


def overview_figure(stats, destination):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), layout='constrained')
    for ax, prefix, label in zip(axes, ['raw', 'log'], ['g/100 g', 'log1p(g/100 g)']):
        finite = np.isfinite(stats[f'{prefix}_mean']) & np.isfinite(stats[f'{prefix}_median'])
        upper = max(stats.loc[finite, [f'{prefix}_mean', f'{prefix}_median']].max().max(), 1e-10)
        ax.plot([0, upper], [0, upper], color='#777777', linestyle=':', linewidth=.8, label='Mean = median')
        for role, color, title in [('maskable_target', '#137f7c', 'Target'), ('context_only', '#ab3f59', 'Context only')]:
            keep = finite & stats.training_role.eq(role)
            ax.scatter(stats.loc[keep, f'{prefix}_median'], stats.loc[keep, f'{prefix}_mean'],
                       s=12, color=color, alpha=.7, label=title)
        ax.set_xlim(-.025 * upper, 1.05 * upper)
        ax.set_ylim(-.025 * upper, 1.05 * upper)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'Median ({label})', fontsize=8)
        ax.set_ylabel(f'Mean ({label})', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
        ax.legend(fontsize=6, frameon=False)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def write_sections(stats, summary, output):
    paths = output.relative_to(ROOT).as_posix()
    counts = summary['axis_counts']
    labels = {
        'raw_mean_above_median': 'Raw mean above median',
        'raw_mean_below_median': 'Raw mean below median',
        'raw_centers_close': 'Raw mean approximately equals median',
        'log_mean_above_median': 'Log mean above median',
        'log_mean_below_median': 'Log mean below median',
        'log_centers_close': 'Log mean approximately equals median',
        'zero_median': 'Observed median equals zero',
    }
    overview = [
        r'\section{Composition Mean and Median}',
        f"This module covers all {len(stats)} retained compositions using only the {summary['training_foods']:,} training food concepts. Each accepted food--composition cell contributes once. Values, partitions, training eligibility and normalization parameters are unchanged. The comparison describes the curated food database, not population dietary exposure.",
        r'\subsection{Definitions and Missing Values}',
        r"For each composition, the arithmetic mean is $\bar{x}=n^{-1}\sum_i x_i$; the median is the middle ordered value, or the average of the two middle values when $n$ is even. The reported difference is mean minus median, not an error or a performance score.",
        r"Raw means and medians use g/100 g. Log means and medians are computed independently after transforming each observation to $y=\log(1+x)$. In particular, $\mathrm{mean}(\log(1+x))$ is not $\log(1+\mathrm{mean}(x))$. Log values here have not been centered or rescaled.",
        r"Missing observations are excluded, never replaced by zero. The main summaries include explicitly recorded zeros. Positive-only means and medians use $x>0$ and are shown separately; they do not replace the main summaries or change the current normalizer. The missing percentage uses all training foods; the zero percentage uses observed values only. An empty subset is shown as --.",
        r'\subsection{Current Mean--Median Comparison}',
        table(['Comparison', 'All axes', 'Targets', 'Context'], [
            [tex(labels[key]), str(value['all']), str(value['target']), str(value['context'])]
            for key, value in counts.items()], [.60, .09, .09, .10]),
        r"For the aggregate counts only, approximate equality means $|\bar{x}-m|\le10^{-12}+10^{-6}|m|$, with the same rule applied to log values. This avoids counting floating-point rounding differences as a directional difference; it is not a statistical significance threshold. Unrounded values and signed differences remain in the CSV tables.",
        r'\begin{center}\includegraphics[width=\textwidth]{' + paths + r'/overview.png}\end{center}',
        r"Each point represents one composition. Points above the diagonal have mean greater than median, and points below it have mean smaller than median. Very small concentrations overlap near the origin in this cross-axis view; the per-composition atlas and numeric tables show both centers individually.",
        r'\subsection{Reading the Per-Composition Plots}',
        r"Each atlas entry compares all observed values with the positive-only subset and contains a raw/log empirical cumulative distribution function (ECDF) pair. The vertical axis is the percentage of observed values less than or equal to the horizontal-axis value. The full observed range is plotted without clipping, subsampling or imputation. Rose solid lines mark means and blue-green dashed lines mark medians; the plots include explicit zeros.",
        r"A mean above the median indicates that higher values pull the average upward relative to the middle ordered observation. A zero median can be a valid consequence of recorded zeros. Neither observation alone establishes that one center is better for model training. Mean and median describe different notions of location; see NIST: \url{" + REFERENCES['location'] + '}.',
        r"This module reports the current centers only and makes no automatic choice between them. Predictive benefit would require a controlled comparison using training data for model selection, with all other settings held fixed. No validation values or model outcomes were used in this descriptive comparison.",
    ]
    (output / 'overview.tex').write_text('\n\n'.join(overview) + '\n', encoding='utf-8')
    atlas = [r'\clearpage\section{Per-Composition Mean and Median Atlas}',
             r'\label{sec:distribution-atlas}',
             r"Training observations only. Missing values are omitted; explicit zeros are included in the main comparison. Raw values are g/100 g; log values are $\log(1+x)$ before centering or rescaling. Difference means mean minus median. Curves show all observed values; positive-only centers are tabulated separately."]
    for i, row in enumerate(stats.to_dict('records')):
        if i > 0 and i % 2 == 0:
            atlas.append(r'\clearpage')
        atlas.extend([
            r'\subsection*{Distribution ' + tex(row['display_id'] + ': ' + row['canonical_name']) + '}',
            r'{\small ' + tex(f"{'Prediction target' if row['training_role']=='maskable_target' else 'Context only'}; observed n={row['n_observed']:,}; missing={100-row['observed_pct']:.1f}% of training foods; zeros={row['zero_pct_observed']:.1f}% of observed; positive-only n={row['n_positive']:,}.") + '}',
            table(['Observations', 'Scale', 'Mean', 'Median', 'Difference'], [
                [label, scale] + [number(row[f'{prefix}_{k}']) for k in ['mean', 'median', 'mean_minus_median']]
                for label, scale, prefix in [('All observed', 'Raw', 'raw'), ('All observed', 'Log1p', 'log'),
                                             ('Positive only', 'Raw', 'positive_raw'), ('Positive only', 'Log1p', 'positive_log')]],
                  [.19, .10, .20, .20, .21], compact=True),
            r'\begin{center}\includegraphics[width=0.96\textwidth]{' + paths + '/figures/' + row['display_id'] + r'.png}\end{center}',
        ])
    (output / 'atlas.tex').write_text('\n\n'.join(atlas) + '\n', encoding='utf-8')
    zh_labels = {
        'raw_mean_above_median': '原始 mean > median',
        'raw_mean_below_median': '原始 mean < median',
        'raw_centers_close': '原始 mean 与 median 近似相等',
        'log_mean_above_median': 'log1p mean > median',
        'log_mean_below_median': 'log1p mean < median',
        'log_centers_close': 'log1p mean 与 median 近似相等',
        'zero_median': '观测值 median 为0',
    }
    md = [f"# Composition mean 与 median：{summary['dataset_version']}", '',
          f"全部 {len(stats)} 个保留轴，仅使用 {summary['training_foods']:,} 个 train food concept。数据、划分及归一化参数未改动。", '',
          '## 当前对比', '', '| 指标 | 全部轴 | target | context |', '|---|---:|---:|---:|']
    md.extend(f"| {zh_labels[k]} | {v['all']} | {v['target']} | {v['context']} |" for k, v in counts.items())
    md.extend(['', '## 统计定义', '',
        '- Mean：已观测值的算术平均数。Median：排序后的中间值；偶数样本取中间两值的平均数。',
        '- Difference：mean 减 median，保留正负号；不是模型预测误差。',
        '- Raw：g/100g。Log：逐个观测先计算 ln(1+x)，再分别计算 mean、median；尚未中心化或缩放。',
        '- Missing 完全忽略；明确记录的0进入主统计。正值子集只包含 x>0，单独展示，不替换主统计。',
        '- 缺失比例的分母是全部训练食品；零值比例的分母是已观测食品。',
        '- 仅汇总计数使用近似相等判据：|mean-median| <= 1e-12 + 1e-6*|median|；它是数值容差，不是显著性检验。CSV保留完整数值。',
        '- ECDF表示不超过横轴数值的观测比例。每个成分展示原始/log两张图，玫红实线为mean，青绿色虚线为median；曲线包含零值，未裁剪尾部。',
        '- 这些结果仅描述中心位置差异，不证明哪种中心会带来更好的预测。没有自动选择中心或重新训练。', '',
        '## 文件', '',
        'composition_distribution.csv：逐轴名称、分类、样本数、缺失/零值比例，以及原始/log、全部观测/仅正值的mean、median与差值。',
        'normalization_review.csv：同一对比的精简表，保留稳定ID和每种口径的中心与差值。',
        'atlas.tex / PDF：全部轴的逐项对比与分布图。没有有效观测的统计量留空或显示 --。', '',
        '## 定义来源', '', f"- NIST Measures of Location: {REFERENCES['location']}"])
    (output / 'README_zh.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def main(version=DEFAULT_VERSION):
    release = ROOT / 'data/processed' / version / 'release'
    report = ROOT / 'reports' / version
    output = report / 'distributions'
    checks = json.loads((report / 'audit_checks.json').read_text())
    if not checks['checks_passed']:
        raise ValueError('Run the dataset audit before distribution reporting')
    manifest = json.loads((release.parent / 'finalized_manifest.json').read_text())
    protected = {str((release / p).relative_to(ROOT)): sha256_file(release / p) for p in manifest['release_hashes']}
    for p, digest in manifest['release_hashes'].items():
        if protected[str((release / p).relative_to(ROOT))] != digest:
            raise ValueError(f'Release file changed after audit: {p}')
    values, foods, components, _, _ = load_release(release)
    stats = summarize_compositions(values, foods, components).sort_values('display_id').reset_index(drop=True)
    by_id = stats.set_index('component_concept_id').loc[components.component_concept_id]
    if not np.array_equal(by_id.n_observed, components.train_count):
        raise AssertionError('Distribution counts differ from registered training support')
    center, scale, normalizer = fit_train_normalizer(values, np.flatnonzero(foods.partition.eq('train')))
    np.testing.assert_allclose(by_id.log_median, center, rtol=0, atol=0)
    np.testing.assert_allclose(by_id.current_effective_scale, scale, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(by_id.current_scale_fallback, normalizer.degenerate_scale)
    (output / 'figures').mkdir(parents=True, exist_ok=True)
    # Export only center comparisons; retain internal consistency checks above.
    metadata = ['display_id', 'component_concept_id', 'canonical_name', 'classification_category',
                'classification_category_en', 'training_role', 'prediction_stage', 'partition',
                'n_foods', 'n_observed', 'n_missing', 'observed_pct', 'n_zero', 'n_positive', 'zero_pct_observed']
    centers = []
    for prefix in ['raw', 'log', 'positive_raw', 'positive_log']:
        stats[f'{prefix}_mean_minus_median'] = stats[f'{prefix}_mean'] - stats[f'{prefix}_median']
        centers.extend(f'{prefix}_{key}' for key in ['mean', 'median', 'mean_minus_median'])
    write_csv(stats[metadata + centers], output / 'composition_distribution.csv')
    compact = ['display_id', 'component_concept_id', 'canonical_name', 'training_role',
               'n_observed', 'n_positive', 'zero_pct_observed'] + centers
    write_csv(stats[compact], output / 'normalization_review.csv')
    target = stats.training_role.eq('maskable_target')
    conditions = {}
    for prefix in ['raw', 'log']:
        mean, median = stats[f'{prefix}_mean'], stats[f'{prefix}_median']
        valid = np.isfinite(mean) & np.isfinite(median)
        close = valid & np.isclose(mean, median, rtol=CENTER_RTOL, atol=CENTER_ATOL)
        conditions[f'{prefix}_mean_above_median'] = valid & ~close & mean.gt(median)
        conditions[f'{prefix}_mean_below_median'] = valid & ~close & mean.lt(median)
        conditions[f'{prefix}_centers_close'] = close
    conditions['zero_median'] = stats.raw_median.eq(0)
    summary = {'dataset_version': version, 'training_foods': int(foods.partition.eq('train').sum()),
               'axes': len(stats), 'observed_training_cells': int(stats.n_observed.sum()),
               'validation_values_used': False, 'normalizer_changed': False, 'report_scope': 'mean_median_only',
               'center_comparison_tolerance': {'rtol': CENTER_RTOL, 'atol': CENTER_ATOL},
               'axis_counts': {k: {'all': int(v.sum()), 'target': int((v & target).sum()), 'context': int((v & ~target).sum())}
                              for k, v in conditions.items()}, 'method_references': REFERENCES}
    write_json(summary, output / 'summary.json')
    train = values[foods.partition.eq('train').to_numpy()]
    column_lookup = {c: i for i, c in enumerate(components.component_concept_id)}
    for i, row in enumerate(stats.itertuples(index=False)):
        x = train[:, column_lookup[row.component_concept_id]]
        ecdf_figure(x[np.isfinite(x)], output / 'figures' / f'{row.display_id}.png')
        if (i + 1) % 50 == 0:
            print(f'Distribution figures: {i + 1}/{len(stats)}', flush=True)
    overview_figure(stats, output / 'overview.png')
    write_sections(stats, summary, output)
    for path, digest in protected.items():
        if sha256_file(ROOT / path) != digest:
            raise AssertionError(f'Distribution reporting modified the dataset: {path}')
    write_json({'dataset_version': version, 'input_hashes': protected, 'complete_axes': len(stats),
                'figure_count': len(stats), 'report_scope': 'mean_median_only', 'validation_values_used': False, 'release_unchanged': True,
                'implementation_hashes': {str(p.relative_to(ROOT)): sha256_file(p) for p in [Path(__file__), ROOT / 'src/foodcomp/distributions.py']}},
               output / 'manifest.json')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default=DEFAULT_VERSION)
    main(parser.parse_args().version)
