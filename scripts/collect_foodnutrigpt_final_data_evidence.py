"""Collect aggregate data/protocol evidence for bilingual reporting; never load test labels."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import ResearchData, VERSION, FROZEN, digest, write_json
from foodcomp.research_completion_input import execution_contract

NAMES = {
    'afcd': 'Australian Food Composition Database',
    'anfood_2_0': 'FAO/INFOODS AnFooD 2.0',
    'bangladesh_fct_2013': 'Bangladesh Food Composition Table 2013',
    'biofoodcomp_4_0': 'FAO/INFOODS BioFoodComp 4.0',
    'bls_4_0': 'German Nutrient Database BLS 4.0',
    'ciqual': 'ANSES-Ciqual',
    'cnf': 'Canadian Nutrient File',
    'cofid': 'UK Composition of Foods Integrated Dataset',
    'efsa_eu_fcdb_2013': 'EFSA EU Food Composition Database (package label: 2013)',
    'fndds': 'USDA Food and Nutrient Database for Dietary Studies',
    'foodb': 'FooDB',
    'frida': 'Frida FoodData, Denmark',
    'lesotho_fct_2006': 'Lesotho Food Composition Table 2006',
    'mext_japan_2023': 'Japan MEXT Food Composition Tables (package label: 2023)',
    'norway': 'Norwegian Food Composition Database',
    'phyfoodcomp_1_0': 'FAO/INFOODS/IZiNCG PhyFoodComp 1.0',
    'swiss_fcdb_7_1': 'Swiss Food Composition Database 7.1',
    'usda_sr_legacy': 'USDA Standard Reference Legacy',
    'wafct_2019': 'FAO/INFOODS Western Africa Food Composition Table 2019',
}
for country in ['cambodia', 'indonesia', 'laos', 'thailand', 'vietnam']:
    NAMES[f'smiling_{country}_2013'] = f'SMILING {country.title()} Food Composition Table 2013'
ALIASES = {'afcd': 'afcd_release_3', 'ciqual': 'ciqual_2025', 'cofid': 'cofid_2021',
    'efsa_eu_fcdb_2013': 'efsa_eu_fcdb_2026', 'fndds': 'usda_fndds_2021_2023',
    'foodb': 'foodb_2020', 'frida': 'frida_6_1', 'norway': 'norwegian_fcdb'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frozen, contract_path = execution_contract(ROOT)
    data = ResearchData(ROOT / 'data/processed' / VERSION)
    registry_path = ROOT / 'data/raw/global_fcdb_inventory_2026_09_10/source_registry.json'
    registry = {item['source_key']: item for item in read(registry_path)['sources']}
    partition_path = ROOT / 'data/processed' / FROZEN / 'partition_summary.csv'
    build_path = partition_path.parent / 'build_manifest.json'
    source_vocabulary = partition_path.parent / 'source_vocabulary.csv'
    assert set(data.profiles.partition) == {'train', 'validation'}
    assert len(data.train) == 64700 and len(data.validation) == 11175
    assert data.profiles.groupby('exact_name_group_id').partition.nunique().max() == 1
    assert set(data.profiles.source_key) == set(NAMES)
    nutrition = data.axes.loss_eligible & data.axes.loss_group.eq('nutrition')
    metabolome = data.axes.loss_eligible & data.axes.loss_group.eq('food_metabolome')
    context = ~data.axes.loss_eligible
    assert (int(nutrition.sum()), int(metabolome.sum()), int(context.sum())) == (142, 45, 65)
    panels = {}
    for dim in [32, 128]:
        path = ROOT / f'data/processed/foodnutrigpt_v9_r8_tasks_name{dim}_v2/manifest.json'
        manifest = read(path)
        panels[str(dim)] = {'manifest_sha256': digest(path),
            'tasks': manifest['tasks'], 'target_cells': manifest['observed_target_cells'],
            'families': manifest['families'], 'name_cache_sha256': manifest['name_cache_hash']}
    rows = []
    for key in sorted(NAMES):
        record = {'source_key': key, 'name': NAMES[key], 'scope': 'included_in_training_and_validation',
            'reference_url': registry[ALIASES.get(key, key)]['official_url'],
            'reference_registry_key': ALIASES.get(key, key),
            'reference_is_not_exact_release_verification': True}
        for split, indices in [('train', data.train), ('validation', data.validation)]:
            indices = indices[data.profiles.iloc[indices].source_key.to_numpy() == key]
            record[split + '_profiles'] = len(indices)
            record[split + '_food_groups'] = int(data.profiles.iloc[indices].exact_name_group_id.nunique())
            for label, mask in [('nutrition142', nutrition), ('metabolome45', metabolome), ('context65', context)]:
                observed = data.observed[indices][:, mask]
                values = data.raw[indices][:, mask]
                record[f'{split}_{label}_observed_cells'] = int(observed.sum())
                record[f'{split}_{label}_explicit_zero_cells'] = int((observed & (values == 0)).sum())
        rows.append(record)
    records = pd.DataFrame(rows)
    assert int(records.train_profiles.sum()) == len(data.train)
    assert int(records.validation_profiles.sum()) == len(data.validation)
    counts = {}
    for split, indices in [('train', data.train), ('validation', data.validation)]:
        counts[split] = {'profiles': len(indices),
            'food_groups': int(data.profiles.iloc[indices].exact_name_group_id.nunique()),
            'unique_original_names': int(data.profiles.iloc[indices].original_name.nunique()),
            'sources': int(data.profiles.iloc[indices].source_key.nunique()),
            'all252_observed_cells': int(data.observed[indices].sum()),
            'supervised187_observed_cells': int(data.observed[indices][:, data.targets].sum()),
            'nutrition142_observed_cells': int(data.observed[indices][:, nutrition].sum()),
            'nutrition142_explicit_zero_cells': int((data.observed[indices][:, nutrition] & (data.raw[indices][:, nutrition] == 0)).sum())}
    assert counts['train']['supervised187_observed_cells'] == 1828536
    assert counts['validation']['supervised187_observed_cells'] == len(data.jobs) == 323809
    evidence = [data.root / 'manifest.json', data.root / 'profiles.csv.gz', data.root / 'axes.csv',
        data.root / 'quarantined.npz', registry_path, partition_path, build_path, source_vocabulary, contract_path]
    summary = {'status': 'complete_aggregate_data_evidence', 'data_version': VERSION, 'counts': counts,
        'source_rows': rows, 'native_package_manifest': read(build_path),
        'native_package_partition_counts': pd.read_csv(partition_path).to_dict('records'),
        'foundation': {'source_key': 'usda_foundation', 'name': 'USDA Foundation Foods',
            'train_profiles': 0, 'validation_profiles': 0, 'held_out_profiles_from_package_manifest': 395,
            'reference_url': registry['usda_foundation']['official_url'],
            'test_labels_loaded_or_evaluated': False},
        'research_view_manifest': data.manifest, 'training_panels': panels,
        'frozen_training_code': frozen['code_hashes'],
        'artifact_hashes': {str(path.relative_to(ROOT)): digest(path) for path in evidence},
        'script_sha256': digest(Path(__file__)), 'complete_test_opened': False,
        'scope': 'Counts recomputed from the quarantined train/validation view. Native holdout counts use pre-existing metadata only. Reference URLs identify source families; the broader acquisition inventory may have newer releases and is not proof of the exact upstream V8 release or record-level validity. No nutrient values or names exported.'}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'summary.json', summary)
    records.to_csv(args.output_dir / 'source_counts.csv', index=False)
    print(json.dumps({'status': summary['status'], 'counts': counts, 'training_panels': panels}))


if __name__ == '__main__':
    main()

