"""Verify the supplied V8 archive against its manifest and frozen local bytes.

This hashes opaque payload bytes. It does not parse food/nutrient rows, extract
files, modify a dataset, fit a model, or evaluate any test labels.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = 'global_foodnutrigpt_v8_single_stage_v2_complete_test'
PACKAGE = 'foodnutrigpt_v8_source_native_baseline'
FILES = {
    'data/axis_registry.csv', 'data/build_manifest.json',
    'data/complete_test_axis_coverage.csv', 'data/food_profiles.csv.gz',
    'data/partition_summary.csv', 'data/source_native_axis_tokens.csv.gz',
    'data/source_vocabulary.csv', 'data/supplemental_test_exact_name_groups.csv',
    'data/train_only_axis_normalization.csv', 'splits/splits.json',
}


def hash_stream(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def digest(path):
    with path.open('rb') as stream:
        return hash_stream(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    before = args.archive.stat()
    archive_sha256 = digest(args.archive)
    members, release, release_sha256 = {}, None, None
    # Streaming members avoids extraction and treats nested .csv.gz as bytes.
    with tarfile.open(args.archive, mode='r|gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if '..' in path.parts or path.is_absolute():
                raise ValueError('Unexpected archive member path.')
            if not member.isfile() or path.name.startswith('._'):
                continue
            if path.parts[0] != PACKAGE:
                raise ValueError('Unexpected package root.')
            relative = path.relative_to(PACKAGE).as_posix()
            if relative in members:
                raise ValueError('Duplicate archive member: ' + relative)
            if relative not in FILES | {'README.md', 'release_manifest.json'}:
                raise ValueError('Unexpected package member: ' + relative)
            with archive.extractfile(member) as stream:
                if relative == 'release_manifest.json':
                    raw = stream.read()
                    release = json.loads(raw.decode('utf-8'))
                    member_hash = release_sha256 = hashlib.sha256(raw).hexdigest()
                else:
                    member_hash = hash_stream(stream)
            members[relative] = {'bytes': member.size, 'sha256': member_hash}
    if release is None or release['dataset_version'] != VERSION or release['release_name'] != PACKAGE:
        raise ValueError('Unexpected release manifest identity.')
    declared = {item['path']: item for item in release['files']}
    if len(declared) != len(release['files']) or set(declared) != FILES:
        raise ValueError('Unexpected declared payload set.')
    if set(members) != FILES | {'README.md', 'release_manifest.json'}:
        raise ValueError('Incomplete archive payload.')

    view_path = ROOT / 'data/processed/foodnutrigpt_v9_r0_v1/manifest.json'
    view_hash = digest(view_path)
    view = json.loads(view_path.read_text(encoding='utf-8'))
    frozen = view['frozen_input_hashes']
    if set(frozen) != FILES:
        raise ValueError('Unexpected R0 frozen-input set.')
    records = []
    for relative in sorted(FILES):
        category, name = relative.split('/', 1)
        local = ROOT / 'data' / ('processed' if category == 'data' else 'splits') / VERSION / name
        actual = members[relative]
        if actual != {key: declared[relative][key] for key in ['bytes', 'sha256']}:
            raise ValueError('Archive differs from release manifest: ' + relative)
        local_sha256 = digest(local)
        if local.stat().st_size != actual['bytes'] or local_sha256 != actual['sha256']:
            raise ValueError('Local native payload differs from archive: ' + relative)
        if frozen[relative] != actual['sha256']:
            raise ValueError('R0 input fingerprint differs from archive: ' + relative)
        records.append({'archive_member': relative, 'local_path': local.relative_to(ROOT).as_posix(),
                        **actual, 'release_manifest_match': True, 'r0_input_match': True})
    after = args.archive.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Archive changed during verification.')
    if digest(view_path) != view_hash:
        raise ValueError('R0 manifest changed during verification.')
    result = {
        'status': 'complete_original_archive_identity_verification',
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'archive_filename': args.archive.name,
        'archive_bytes': before.st_size, 'archive_sha256': archive_sha256,
        'dataset_version': VERSION, 'release_manifest_sha256': release_sha256,
        'r0_manifest_sha256': view_hash, 'payload_files': records,
        'all_ten_files_match_archive_release_and_r0_fingerprints': True,
        'script_sha256': digest(Path(__file__)),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'data_modified': False, 'archive_extracted': False, 'nutrient_rows_parsed': False,
        'complete_test_opened_for_model_evaluation': False, 'model_training_performed': False,
        'upstream_authenticity_independently_verified': False,
        'record_level_unit_and_label_validity_proven': False,
        'scope': 'Byte identity of the user-supplied archive, its embedded manifest, the frozen local V8 payload, and R0 input fingerprints. The embedded checksums are not an independent publisher signature. This does not establish exact upstream database releases, licensing, provenance of individual values, or label correctness. Payload tables remain opaque bytes; no food names or nutrient values are exported.',
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['status', 'archive_bytes', 'archive_sha256',
        'all_ten_files_match_archive_release_and_r0_fingerprints']}))


if __name__ == '__main__':
    main()
