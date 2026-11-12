#!/usr/bin/env python3
"""Build lazy, incremental full-test-set data for transcript browsing."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import CommitOperationDelete, HfApi, hf_hub_download

from build_cache import output_fingerprints, source_signature
from build_compare_data import DATASETS, NORMALISE, clip_id, output_slugs
from danish_asr_leaderboard.metrics import compute_cer, compute_wer
from danish_asr_leaderboard.normalizer import normalise

REPO = 'RyeAI/danish-asr-leaderboard'
REFERENCE = 'outputs/danish-foundation-models__edda-v0__2.parquet'
EXPECTED = {'coral_conversation': 8438, 'coral_read_aloud': 9122,
            'cv17_da': 2756, 'fleurs_da': 930, 'ftspeech': 5534}


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()[:20]


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(compact(value), encoding='utf-8')


def reference_data(frame):
    result = {}
    for dataset in DATASETS:
        entries = []
        seen = set()
        for row in frame[frame.dataset == dataset].itertuples(index=False):
            key = clip_id(row.id)
            if not key or key in seen or not isinstance(row.reference, str):
                raise ValueError(f'Invalid or duplicate reference ID: {dataset}/{key}')
            seen.add(key)
            entries.append([key, row.reference, normalise(row.reference, **NORMALISE)])
        if len(entries) != EXPECTED[dataset]:
            raise ValueError(f'{dataset}: expected {EXPECTED[dataset]} references, got {len(entries)}')
        result[dataset] = entries
    return result


def model_data(frame, references):
    result = {}
    for dataset in DATASETS:
        ref_by_id = {key: (ref, normalized) for key, ref, normalized in references[dataset]}
        rows = {}
        for row in frame[frame.dataset == dataset].itertuples(index=False):
            key = clip_id(row.id)
            if key in rows:
                raise ValueError(f'Duplicate model output ID: {dataset}/{key}')
            known = ref_by_id.get(key)
            if not known or row.reference != known[0]:
                continue
            hypothesis = row.hypothesis if isinstance(row.hypothesis, str) else ''
            normalized = normalise(hypothesis, **NORMALISE)
            rows[key] = [hypothesis, normalized,
                         round(compute_wer([known[1]], [normalized]), 1),
                         round(compute_cer([known[1]], [normalized]), 1)]
        if rows:
            result[dataset] = rows
    return result


def prune_old_files(api, files, manifest, parent_commit):
    """Remove obsolete generated files from the current revision, not Git history."""
    keep = set(manifest['references'].values())
    for model in manifest['models'].values():
        keep.update(model['paths'].values())
    datasets = '|'.join(map(re.escape, DATASETS))
    pattern = re.compile(
        rf'compare-all/(?:references/[0-9a-f]{{20}}|'
        rf'models/[a-zA-Z0-9_.-]+/[0-9a-f]{{20}})/({datasets})\.json'
    )
    stale = sorted(path for path in files if pattern.fullmatch(path) and path not in keep)
    if stale:
        api.create_commit(repo_id=REPO, repo_type='space', parent_commit=parent_commit,
                          operations=[CommitOperationDelete(path, is_folder=False) for path in stale],
                          commit_message='Remove outdated transcript browser data')
        print(f'Removed {len(stale)} outdated transcript files', flush=True)


def build(root, cache_dir, outputs_dir, publish=False):
    api = HfApi()
    fingerprints = output_fingerprints(api.repo_info(REPO, repo_type='dataset', files_metadata=True).siblings)
    if REFERENCE not in fingerprints or not fingerprints[REFERENCE]:
        raise ValueError('Reference model outputs are missing a content fingerprint')
    if any(not fingerprint for fingerprint in fingerprints.values()):
        raise ValueError('Some model outputs are missing content fingerprints')
    aliases = output_slugs(outputs_dir)
    signature = digest([source_signature('scripts/build_compare_data.py'), Path(__file__).read_text()])
    space_revision = api.repo_info(REPO, repo_type='space').sha
    existing_files = set(api.list_repo_files(REPO, repo_type='space', revision=space_revision))
    old = {}
    if 'compare-all/manifest.json' in existing_files:
        path = hf_hub_download(REPO, 'compare-all/manifest.json', repo_type='space',
                               revision=space_revision, local_dir=cache_dir)
        old = json.loads(Path(path).read_text(encoding='utf-8'))
    paths = {aliases.get(Path(name).stem, Path(name).stem): (name, sha)
             for name, sha in fingerprints.items()}
    if len(paths) != len(fingerprints):
        raise ValueError('Multiple output files resolve to the same model')
    valid_old = old.get('signature') == signature
    reusable_refs = (valid_old and old.get('reference_source') == fingerprints[REFERENCE]
                     and all(path in existing_files for path in old.get('references', {}).values())
                     and set(old.get('references', {})) == set(DATASETS))
    same_models = valid_old and reusable_refs and set(paths) == set(old.get('models', {})) and all(
        old['models'][slug]['fingerprint'] == sha
        and all(path in existing_files for path in old['models'][slug]['paths'].values())
        for slug, (_, sha) in paths.items())
    if same_models:
        print('All clip data is unchanged', flush=True)
        if not publish:
            raise ValueError('Published data exists; use the deployed Space for a preview')
        prune_old_files(api, existing_files, old, space_revision)
        return
    if reusable_refs:
        references = {}
        for dataset, path in old['references'].items():
            file = hf_hub_download(REPO, path, repo_type='space', revision=space_revision, local_dir=cache_dir)
            references[dataset] = json.loads(Path(file).read_text(encoding='utf-8'))
    else:
        path = hf_hub_download(REPO, REFERENCE, repo_type='dataset', local_dir=cache_dir)
        references = reference_data(pd.read_parquet(path))
    ref_digest = digest(references)
    ref_paths = {dataset: f'compare-all/references/{ref_digest}/{dataset}.json' for dataset in DATASETS}
    for dataset, path in ref_paths.items():
        if path not in existing_files:
            save(root, path, references[dataset])
    manifest = {'signature': signature, 'reference_source': fingerprints[REFERENCE],
                'reference_digest': ref_digest, 'references': ref_paths, 'models': {}}
    for number, (slug, (file, sha)) in enumerate(sorted(paths.items()), 1):
        previous = old.get('models', {}).get(slug, {})
        if (valid_old and old.get('reference_digest') == ref_digest
                and previous.get('fingerprint') == sha
                and all(path in existing_files for path in previous.get('paths', {}).values())):
            manifest['models'][slug] = previous
            continue
        path = hf_hub_download(REPO, file, repo_type='dataset', local_dir=cache_dir)
        groups = model_data(pd.read_parquet(path), references)
        version = digest([sha, signature, ref_digest])
        model_paths = {dataset: f'compare-all/models/{slug}/{version}/{dataset}.json' for dataset in groups}
        for dataset, target in model_paths.items():
            save(root, target, groups[dataset])
        manifest['models'][slug] = {'fingerprint': sha, 'paths': model_paths}
        print(f'[{number}/{len(paths)}] {slug}: {sum(map(len, groups.values()))} matching clips', flush=True)
    save(root, 'compare-all/manifest.json', manifest)
    if publish:
        if any(path.name != 'manifest.json' for path in (root / 'compare-all').rglob('*.json')):
            space_revision = api.upload_folder(
                folder_path=str(root / 'compare-all'), path_in_repo='compare-all',
                repo_id=REPO, repo_type='space', ignore_patterns=['manifest.json'],
                parent_commit=space_revision, commit_message='Update full transcript browser data').oid
        commit = api.upload_file(path_or_fileobj=str(root / 'compare-all/manifest.json'),
                                 path_in_repo='compare-all/manifest.json', repo_id=REPO, repo_type='space',
                                 parent_commit=space_revision, commit_message='Update transcript browser manifest')
        prune_old_files(api, existing_files, manifest, commit.oid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--outputs-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    build(args.out, args.cache_dir, args.outputs_dir, args.publish)


if __name__ == '__main__':
    main()
