from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from grace_h1 import inference
from grace_h1.io import write_json, write_jsonl, read_json, file_hash, sha256, row_identity
from tools import rerun_devign_test as rerun

check_runtime = rerun.check_runtime


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path
    rows = [{'record_id': str(i), 'code_sha256': sha256(str(i))} for i in range(4)]
    prompts = [{**r, 'prompt': 'prompt-' + str(min(i, 2)),
                'prompt_sha256': sha256('prompt-' + str(min(i, 2))),
                'prompt_truncation': {'context_limit': 8192}}
               for i, r in enumerate(rows)]
    parent = root / 'runs/devign/test/baseline_whitening/old'
    for name in ('retrieval.jsonl', 'graph_feature_index.jsonl'):
        write_jsonl(parent / name, [])
    write_jsonl(parent / 'prompts.jsonl', prompts)
    write_json(parent / 'run_manifest.json', {
        'config': {'dataset': 'devign', 'split': 'test'},
        'target_identity': row_identity(rows), 'target_count': len(rows),
        'prepare_fingerprint': 'old', 'prepare_seconds': 1,
    })
    write_json(parent / 'freeze.json', {'files': {
        p.relative_to(root).as_posix(): file_hash(p)
        for p in parent.iterdir() if p.name != 'freeze.json'}})
    stage = {'experiment': {'frozen': True}, 'jobs': [
        {'dataset': dataset, 'seed': 42, 'test_run': parent.relative_to(root).as_posix(),
         'base_folder': 'base', 'meta_folder': 'meta', 'test_scores': 'scores'}
        for dataset in ('devign', 'reveal')]}
    monkeypatch.setattr(rerun.workflow, 'load_stage', lambda *args: deepcopy(stage))
    monkeypatch.setattr(rerun, 'load_rows', lambda *args: rows)
    monkeypatch.setattr(rerun, 'verify', lambda *args: None)
    monkeypatch.setattr(rerun, 'check_runtime', lambda *args: {})
    (root / 'configs').mkdir()
    (root / 'configs/reference_contract.yaml').write_text(
        'inference_config:\n  temperature: 0\n  seed: 42\n  max_tokens: 256\n'
        '  retry_policy: "max_attempts=1; exponential_backoff=1s..1s"\n', encoding='utf-8')
    settings = {'model': rerun.MODEL, 'api_key': 'test-only', 'base_url': 'https://token-api.fpt.ai/v1'}
    return root, {}, settings, parent, stage


def test_fresh_cache_and_interrupted_resume(prepared, monkeypatch):
    root, cfg, settings, parent, _ = prepared
    original = {p.name: p.read_bytes() for p in parent.iterdir()}
    with rerun._provider(settings):
        spec = inference.request_spec(root)
    # Seed global cache with valid-but-bad zeros. They must never be reused.
    for p in rerun.iter_jsonl(parent / 'prompts.jsonl'):
        key = inference.request_key(p['prompt_sha256'], spec)
        write_json(inference.response_path(root, key), {
            'request_key': key, 'prompt_sha256': p['prompt_sha256'], 'request_spec': spec,
            'request_status': 'ok', 'raw_response': '0', 'raw_response_sha256': sha256('0')})
    calls = []

    class Failure(RuntimeError):
        code = 400

    class Client:
        def __init__(self, **kwargs):
            self.models = self

        def generate_content(self, **kwargs):
            text = kwargs['contents']
            calls.append(text)
            if text == 'prompt-1' and calls.count(text) == 1:
                raise Failure('transient test failure')
            return SimpleNamespace(text='1', usage_metadata=None)

        def close(self):
            pass

    monkeypatch.setattr(inference, 'fpt_sdk', lambda s: (
        SimpleNamespace(Client=Client), SimpleNamespace(
            HttpOptions=SimpleNamespace, GenerateContentConfig=SimpleNamespace)))
    old_path, old_import = inference.response_path, inference.import_prior_responses
    plan = rerun.preflight(root, cfg, settings, session='test')
    assert (plan['queries'], plan['new_unique_requests'], plan['cached_queries']) == (4, 3, 0)
    first = rerun.infer(root, cfg, settings, 'test', workers=1, allow_api=True)
    assert first['status'] == 'partial_request_errors'
    assert first['new_unique_requests'] == 3
    second = rerun.infer(root, cfg, settings, 'test', workers=1, allow_api=True)
    assert second['status'] == 'complete'
    assert second['new_unique_requests'] == 1
    assert second['prediction_counts'] == {'1': 4}
    assert second['near_constant_output']
    assert len(calls) == 4
    assert rerun.preflight(root, cfg, settings, 'test')['new_unique_requests'] == 0
    assert inference.response_path is old_path and inference.import_prior_responses is old_import
    assert original == {p.name: p.read_bytes() for p in parent.iterdir()}
    session = read_json(rerun._folder(root, 'test') / 'meta_selection.json')
    assert [j['dataset'] for j in session['jobs']] == ['devign']
    assert 'test-only' not in (rerun._folder(root, 'test') / 'request_spec.json').read_text()


def test_resume_rejects_prompt_tampering_and_endpoint_switch(prepared):
    root, cfg, settings, _, _ = prepared
    rerun.preflight(root, cfg, settings, 'test')
    with pytest.raises(ValueError, match='endpoint'):
        rerun.preflight(root, cfg, {**settings, 'base_url': 'https://other.fpt.ai/v1'}, 'test')
    p = rerun._folder(root, 'test') / 'expert/prompts.jsonl'
    p.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='Frozen input'):
        rerun.prepare(root, cfg, 'test')


def test_context_limit_and_cohort_are_checked(prepared):
    root, cfg, _, parent, _ = prepared
    prompts = list(rerun.iter_jsonl(parent / 'prompts.jsonl'))
    prompts[0]['prompt'] = 'á' * 4097
    prompts[0]['prompt_sha256'] = sha256(prompts[0]['prompt'])
    write_jsonl(parent / 'prompts.jsonl', prompts)
    freeze = read_json(parent / 'freeze.json')
    freeze['files'][(parent / 'prompts.jsonl').relative_to(root).as_posix()] = file_hash(parent / 'prompts.jsonl')
    write_json(parent / 'freeze.json', freeze)
    with pytest.raises(ValueError, match='8192-byte'):
        rerun.prepare(root, cfg, 'test')
    prompts.pop()
    write_jsonl(parent / 'prompts.jsonl', prompts)
    freeze['files'][(parent / 'prompts.jsonl').relative_to(root).as_posix()] = file_hash(parent / 'prompts.jsonl')
    write_json(parent / 'freeze.json', freeze)
    with pytest.raises(ValueError, match='complete, ordered'):
        rerun.prepare(root, cfg, 'test')


def test_evaluation_uses_frozen_devign_only(prepared, monkeypatch):
    root, cfg, settings, _, _ = prepared
    rerun.preflight(root, cfg, settings, 'test')
    folder = rerun._folder(root, 'test')
    write_json(folder / 'expert/inference_summary.json', {'status': 'complete'})
    called = []

    def predict(root, config):
        called.append('predict')
        assert config['datasets'] == ['devign']
        stage = rerun.workflow.load_stage(root, config, 'meta_selection')
        assert len(stage['jobs']) == 1
        assert rerun.workflow.stage_file(root, config, 'predictions_selection').parent == folder

    def evaluate(root, config):
        called.append('evaluate')
        return {'datasets': [{'dataset': 'devign', 'seed': 42}]}

    monkeypatch.setattr(rerun.workflow, 'predict', predict)
    monkeypatch.setattr(rerun.workflow, 'evaluate', evaluate)
    result = rerun.evaluate(root, cfg, settings, 'test')
    assert called == ['predict', 'evaluate']
    assert result['models_refitted'] is False
    assert result['benchmark_comparable'] is False
    assert (folder / 'predictions/devign/42/evaluation.json').exists()


def test_runtime_mismatch_fails_before_api(prepared, monkeypatch):
    root, cfg, _, _, _ = prepared
    for name in ('base', 'meta'):
        write_json(root / name / 'manifest.json', {'models': {
            'model': {'path': 'frozen.joblib', 'versions': {'numpy': 'old'}}}})
    monkeypatch.setattr(rerun, 'versions', lambda: {'numpy': 'new'})
    with pytest.raises(RuntimeError, match='original training kernel before inference'):
        check_runtime(root, cfg)
