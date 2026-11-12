"""Full-clip data must use the benchmark's scoring and incremental cache."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_browse_data as browse
from build_compare_data import score_example


def test_browse_scores_match_benchmark_and_ignore_mismatched_references():
    reference = {'coral_conversation': [['a.wav', 'Jeg har to katte.',
                                            browse.normalise('Jeg har to katte.', **browse.NORMALISE)]]}
    reference.update({dataset: [] for dataset in browse.DATASETS if dataset != 'coral_conversation'})
    frame = pd.DataFrame([
        {'dataset': 'coral_conversation', 'id': '/tmp/a.wav',
         'reference': 'Jeg har to katte.', 'hypothesis': 'Jeg har tre katte.'},
        {'dataset': 'coral_conversation', 'id': '/tmp/b.wav',
         'reference': 'wrong reference', 'hypothesis': 'ignored'},
    ])
    row = browse.model_data(frame, reference)['coral_conversation']['a.wav']
    expected = score_example('Jeg har to katte.', 'Jeg har tre katte.')
    assert row[2:] == [expected['wer'], expected['cer']]
    assert row[1] == browse.normalise(row[0], **browse.NORMALISE)
    assert len(browse.model_data(frame, reference)['coral_conversation']) == 1


def test_browse_cache_and_retire_only_obsolete_files(tmp_path, monkeypatch):
    monkeypatch.setattr(browse, 'EXPECTED', {dataset: 1 for dataset in browse.DATASETS})
    monkeypatch.setattr(browse, 'source_signature', lambda *_: 'test-signature')
    published = {}
    revisions = {browse.REFERENCE: 'ref1', 'outputs/another.parquet': 'other1'}
    calls = []
    state = {'head': 0, 'fail_manifest': False, 'race_cleanup': False}
    frame = pd.DataFrame([{'dataset': dataset, 'id': f'{dataset}.wav',
                           'reference': 'Jeg har to katte.', 'hypothesis': 'Jeg har tre katte.'}
                          for dataset in browse.DATASETS])

    class Api:
        def repo_info(self, *_args, **kwargs):
            if kwargs['repo_type'] == 'space':
                return SimpleNamespace(sha=str(state['head']))
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename=name, lfs={'sha256': sha})
                                             for name, sha in revisions.items()])

        def list_repo_files(self, *_args, **kwargs):
            assert kwargs['revision'] == str(state['head'])
            return list(published)

        def advance(self, parent):
            assert parent == str(state['head'])
            state['head'] += 1
            return SimpleNamespace(oid=str(state['head']))

        def upload_folder(self, folder_path, **kwargs):
            result = self.advance(kwargs['parent_commit'])
            base = Path(folder_path)
            for path in base.rglob('*.json'):
                if path.name != 'manifest.json':
                    published['compare-all/' + str(path.relative_to(base))] = path.read_text()
            return result

        def upload_file(self, path_or_fileobj, path_in_repo, **kwargs):
            if state['fail_manifest']:
                raise RuntimeError('manifest upload failed')
            result = self.advance(kwargs['parent_commit'])
            published[path_in_repo] = Path(path_or_fileobj).read_text()
            return result

        def create_commit(self, operations, **kwargs):
            if state['race_cleanup']:
                state['head'] += 1  # Another publisher advanced the repo.
            result = self.advance(kwargs['parent_commit'])
            for operation in operations:
                assert not operation.is_folder
                del published[operation.path_in_repo]
            return result

    def download(_repo, filename, repo_type, local_dir, revision=None):
        calls.append((repo_type, filename))
        if repo_type == 'space':
            assert revision == str(state['head'])
        dest = Path(local_dir) / repo_type / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('data' if repo_type == 'dataset' else published[filename])
        return str(dest)

    monkeypatch.setattr(browse, 'HfApi', Api)
    monkeypatch.setattr(browse, 'hf_hub_download', download)
    monkeypatch.setattr(browse.pd, 'read_parquet', lambda _: frame)
    outputs = tmp_path / 'outputs'
    outputs.mkdir()

    def build(name):
        browse.build(tmp_path / name, tmp_path / f'cache-{name}', outputs, publish=True)

    def manifest():
        return json.loads(published['compare-all/manifest.json'])

    build('first')
    first = manifest()
    assert len(first['models']) == 2
    assert len(first['references']) == 5
    old_model_paths = set(first['models']['another']['paths'].values())
    reference_paths = set(first['references'].values())
    reference_model_paths = set(first['models']['danish-foundation-models__edda-v0__2']['paths'].values())
    orphan = f"compare-all/models/orphan/{'a' * 20}/ftspeech.json"
    unrelated = f"compare-all/models/orphan/{'a' * 20}/notes.json"
    published[orphan] = '{}'
    published[unrelated] = 'keep'
    published['compare-all/README.md'] = 'keep'

    calls.clear()
    build('same')
    assert not [call for call in calls if call[0] == 'dataset']
    assert not (tmp_path / 'same').exists()
    assert orphan not in published
    assert published[unrelated] == 'keep'
    assert published['compare-all/README.md'] == 'keep'

    revisions['outputs/another.parquet'] = 'other2'
    calls.clear()
    build('changed')
    assert [name for kind, name in calls if kind == 'dataset'] == ['outputs/another.parquet']
    updated = manifest()
    assert updated['models']['another']['fingerprint'] == 'other2'
    assert updated['models']['danish-foundation-models__edda-v0__2'] == first['models']['danish-foundation-models__edda-v0__2']
    assert not old_model_paths & published.keys()
    assert reference_paths | reference_model_paths <= published.keys()
    assert set(updated['models']['another']['paths'].values()) <= published.keys()

    revisions['outputs/new.parquet'] = 'new1'
    calls.clear()
    build('added')
    assert [name for kind, name in calls if kind == 'dataset'] == ['outputs/new.parquet']
    assert set(updated['models']['another']['paths'].values()) <= published.keys()
    assert reference_paths | reference_model_paths <= published.keys()

    added_paths = set(manifest()['models']['new']['paths'].values())
    del revisions['outputs/new.parquet']
    build('removed')
    assert not added_paths & published.keys()
    assert unrelated in published

    missing_path = manifest()['models']['another']['paths']['ftspeech']
    del published[missing_path]
    calls.clear()
    build('missing')
    assert [name for kind, name in calls if kind == 'dataset'] == ['outputs/another.parquet']
    assert missing_path in published

    # Failed publication keeps the old manifest and every file it references.
    before = manifest()
    before_files = set(before['references'].values()) | {
        path for model in before['models'].values() for path in model['paths'].values()}
    revisions['outputs/another.parquet'] = 'other3'
    state['fail_manifest'] = True
    with pytest.raises(RuntimeError, match='manifest upload failed'):
        build('failure')
    assert manifest() == before
    assert before_files <= published.keys()
    state['fail_manifest'] = False

    # If another publisher wins the race, cleanup refuses to delete any files.
    state['race_cleanup'] = True
    with pytest.raises(AssertionError):
        build('concurrent')
    assert before_files <= published.keys()
    state['race_cleanup'] = False
    build('after-race')
    assert not set(before['models']['another']['paths'].values()) & published.keys()

    # A reference change retires both superseded references and model outputs.
    previous = manifest()
    previous_refs = set(previous['references'].values())
    previous_models = {path for model in previous['models'].values() for path in model['paths'].values()}
    revisions[browse.REFERENCE] = 'ref2'
    frame['reference'] = 'Jeg har fire katte.'
    build('reference-change')
    assert not previous_refs & published.keys()
    assert not previous_models & published.keys()
    assert set(manifest()['references'].values()) <= published.keys()
    assert unrelated in published
