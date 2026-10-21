"""Fresh, resumable Devign test inference using frozen H3-B models/prompts.

Kept outside the source-hashed learning packages. All writes are session-local;
the active benchmark and its ReVeal results are never replaced here.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import re
import shutil
import sys
import yaml
from unittest.mock import patch

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from grace_h1 import inference
from grace_h1.data import load_rows
from grace_h1.graphs import GraphStore
from grace_h1.io import (read_json, write_json, iter_jsonl, file_hash,
                         fingerprint, row_identity, sha256, write_jsonl)
from grace_h1.parser import parse_prediction
from grace_h1.pipeline import verify_frozen, INSTRUCTION
from grace_h1.reference import raw_graph_context, fit_prompt, fit_contracted_prompt
from grace_h3v2 import workflow
from grace_h3v2.common import verify, versions

MODEL = 'DeepSeek-V4-Flash'
LIMIT = 8192
DEFAULT_SESSION = 'devign_fpt_original_over8192_20261021'


def _original_prompt(template, target, graph, demos, policy):
    # Retain component caps (target/demo/graph); remove only the total-context cap.
    cap = len(template) + len(INSTRUCTION) + len(target) + len(graph)
    cap += sum(len(d['code']) + 100 for d in demos) + 1024
    return fit_prompt(template, target, graph, demos, INSTRUCTION,
                      dict(policy, context_limit=cap))[0]


def _selection(root, prompts):
    queries = {r['record_id']: r for r in load_rows(root, 'devign', 'test')}
    train = {r['record_id']: r for r in load_rows(root, 'devign', 'train', labels=True)}
    policy = yaml.safe_load((root / 'configs/reference_contract.yaml').read_text(encoding='utf-8'))['prompt_context_policy']['truncation_policy']
    template = (root / 'configs/prompt_template.txt').read_text(encoding='utf-8')
    graphs = GraphStore(root)
    result = []
    for i, p in enumerate(prompts, 1):
        query = queries[p['record_id']]
        graph = graphs.get(query['code_sha256'])  # cache only; never build graphs
        if graph['feature_sha256'] != p['graph_feature_sha256']:
            raise ValueError('Original prompt graph identity differs')
        graph_text, _ = raw_graph_context(graph, policy)
        demos = [train[x] for x in p['selected_demo_ids']]
        fitted, _ = fit_contracted_prompt(template, query['code'], graph_text, demos,
                                         INSTRUCTION, dict(policy, context_limit=LIMIT))
        if fitted != p['prompt']:
            raise ValueError('Reconstructed 8192-byte prompt differs from the saved prompt')
        original = _original_prompt(template, query['code'], graph_text, demos, policy)
        size = len(original.encode('utf-8'))
        result.append({'record_id': p['record_id'], 'original_prompt_utf8_bytes': size,
                       'original_prompt_sha256': sha256(original), 'rerun': size > LIMIT})
        if i % 500 == 0 or i == len(prompts):
            print(f'Original prompt length audit {i}/{len(prompts)}', flush=True)
    return result


def _folder(root, session):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', session):
        raise ValueError('Session must be a simple name, without directory separators')
    return Path(root).resolve() / 'runs/h3v2/inference_reruns' / session


def _check_prompts(root, run):
    manifest = verify_frozen(root, run)
    if manifest['config']['dataset'] != 'devign' or manifest['config']['split'] != 'test':
        raise ValueError('Only Devign test is allowed')
    rows = load_rows(root, 'devign', 'test')  # labels intentionally unopened
    prompts = list(iter_jsonl(run / 'prompts.jsonl'))
    if (len(prompts) != len(rows) or len({p['record_id'] for p in prompts}) != len(rows)
            or row_identity(prompts) != row_identity(rows)
            or manifest['target_identity'] != row_identity(rows)):
        raise ValueError('Expected the complete, ordered Devign test cohort')
    for p in prompts:
        if (p['prompt_sha256'] != sha256(p['prompt'])
                or p['prompt_truncation']['context_limit'] != LIMIT
                or len(p['prompt'].encode('utf-8')) > LIMIT):
            raise ValueError('Prompt hash or 8192-byte context limit differs')
    return manifest, prompts


def check_runtime(root, cfg):
    """Fail before paid inference if frozen models cannot run in this kernel."""
    current = versions()
    stage = workflow.load_stage(Path(root), cfg, 'meta_selection')

    def entries(value):
        if isinstance(value, dict):
            if 'path' in value and 'versions' in value:
                yield value
            else:
                for child in value.values():
                    yield from entries(child)

    for job in stage['jobs']:
        if job['dataset'] != 'devign':
            continue
        for key in ('base_folder', 'meta_folder'):
            manifest = read_json(Path(root) / job[key] / 'manifest.json')
            for entry in entries(manifest['models']):
                if entry['versions'] != current:
                    raise RuntimeError(
                        'Choose the original training kernel before inference. '
                        f'Required: {entry["versions"]}; current: {current}. '
                        'Existing matching WSL environment: '
                        '/mnt/d/JUNIOR/LabThayTho/GRACE-H1-SecurityProjection/.venv/bin/python. '
                        'Do not install requirements-v2.txt blindly: its pins differ from these artifacts.')
    return current


def prepare(root, cfg, session=DEFAULT_SESSION):
    """No API, fitting, graph construction, or changes to active stage files."""
    root = Path(root).resolve()
    stage = workflow.load_stage(root, cfg, 'meta_selection')
    jobs = [deepcopy(j) for j in stage['jobs'] if j['dataset'] == 'devign']
    if not jobs:
        raise ValueError('No frozen Devign job; complete meta preparation first')
    parents = {j['test_run'] for j in jobs}
    if len(parents) != 1:
        raise ValueError('Expected one shared Devign expert run')
    parent_rel = next(iter(parents))
    parent = root / parent_rel
    manifest, prompts = _check_prompts(root, parent)
    for j in jobs:
        for key in ('base_folder', 'meta_folder', 'test_scores'):
            verify(root, root / j[key])
    folder = _folder(root, session)
    identity = {
        'schema': 'h3_devign_inference_rerun_v1', 'session': session,
        'parent_test_run': parent_rel,
        'parent_prepare_fingerprint': manifest['prepare_fingerprint'],
        'parent_prompts_sha256': file_hash(parent / 'prompts.jsonl'),
        'parent_predictions_sha256': file_hash(parent / 'predictions.jsonl'),
        'parent_raw_responses_sha256': file_hash(parent / 'raw_responses.jsonl'),
        'selection_rule': 'original_prompt_utf8_bytes > 8192; component caps unchanged',
        'experiment': stage['experiment'], 'jobs': deepcopy(jobs),
        'context_limit_utf8_bytes': LIMIT, 'model': MODEL,
        'provider': 'FPT AI Factory', 'tool_sha256': file_hash(Path(__file__)),
    }
    selection = folder / 'session.json'
    if selection.exists():
        if read_json(selection)['identity'] != identity:
            raise ValueError('Session inputs changed; preserve it and choose a new session name')
        _check_prompts(root, folder / 'expert')
        audit = list(iter_jsonl(folder / 'selection.jsonl'))
    else:
        audit = _selection(root, prompts)
        expert = folder / 'expert'
        expert.mkdir(parents=True, exist_ok=True)
        locked = {}
        write_jsonl(folder / 'selection.jsonl', audit)
        locked[(folder / 'selection.jsonl').relative_to(root).as_posix()] = file_hash(folder / 'selection.jsonl')
        for name in ('prompts.jsonl', 'retrieval.jsonl', 'graph_feature_index.jsonl'):
            source, dest = parent / name, expert / name
            shutil.copy2(source, dest)
            if file_hash(source) != file_hash(dest):
                raise ValueError('Copied preparation differs')
            locked[dest.relative_to(root).as_posix()] = file_hash(dest)
        # Keep all original frozen dependencies, in addition to session copies.
        locked.update(read_json(parent / 'freeze.json')['files'])
        new_manifest = deepcopy(manifest)
        new_manifest.update(parent_run=parent_rel, inference_rerun=identity,
                            prepare_fingerprint=fingerprint(identity))
        write_json(expert / 'run_manifest.json', new_manifest)
        locked[(expert / 'run_manifest.json').relative_to(root).as_posix()] = file_hash(expert / 'run_manifest.json')
        write_json(expert / 'freeze.json', {'files': locked})
        for j in jobs:
            j['parent_test_run'] = parent_rel
            j['test_run'] = expert.relative_to(root).as_posix()
        write_json(folder / 'meta_selection.json', {
            **stage, 'jobs': jobs,
            'test_context_profile': {'limit_utf8_bytes': LIMIT},
        })
        write_json(selection, {'identity': identity, 'expert_run': expert.relative_to(root).as_posix()})
        _check_prompts(root, expert)
    return {'session': session, 'folder': str(folder),
            'expert_run': str(folder / 'expert'), 'queries': len(prompts),
            'unique_prompts': len({p['prompt_sha256'] for p in prompts}),
            'selected_queries': sum(p['rerun'] for p in audit),
            'preserved_parent_queries': sum(not p['rerun'] for p in audit),
            'max_prompt_bytes': max(len(p['prompt'].encode('utf-8')) for p in prompts),
            'context_limit_utf8_bytes': LIMIT, 'model': MODEL,
            'provider': 'FPT AI Factory', 'api_calls': 0, 'models_refitted': False}


@contextmanager
def _provider(settings):
    settings = dict(settings)
    if settings.get('model') != MODEL or not settings.get('base_url', '').startswith('https://'):
        raise ValueError('Require DeepSeek-V4-Flash with an HTTPS FPT endpoint')
    with patch.object(inference, 'load_fpt_settings', lambda: dict(settings)):
        yield


@contextmanager
def _private_cache(expert):
    # The existing executor writes each completed request here immediately.
    # A restart resumes only this session; never import global/prior responses.
    original_read = inference.read_cached

    def valid_cached(root, key, prompt_hash, spec):
        row = original_read(root, key, prompt_hash, spec)
        return row if row and parse_prediction(row['raw_response']).prediction is not None else None

    with patch.object(inference, 'response_path', lambda root, key: expert / 'requests' / (key + '.json')), \
            patch.object(inference, 'import_prior_responses', lambda root, spec: 0), \
            patch.object(inference, 'read_cached', valid_cached):
        yield


def _bound_session(root, cfg, session, settings):
    check_runtime(root, cfg)
    prepare(root, cfg, session)
    folder = _folder(root, session)
    with _provider(settings):
        spec = inference.request_spec(root)
    identity = read_json(folder / 'session.json')['identity']
    parent = root / identity['parent_test_run']
    old_summary = read_json(parent / 'inference_summary.json')
    if old_summary['status'] != 'complete' or old_summary['request_spec'] != spec:
        raise ValueError('Retained responses must be complete and use exactly the selected model/provider/spec')
    binding = folder / 'request_spec.json'
    if binding.exists() and read_json(binding) != spec:
        raise ValueError('Model, endpoint or inference parameters changed inside this session')
    if not binding.exists():
        write_json(binding, spec)  # public request identity only, no credentials
    return folder


@contextmanager
def _selected_prompts(folder):
    selected = {r['record_id'] for r in iter_jsonl(folder / 'selection.jsonl') if r['rerun']}
    original_iter = inference.iter_jsonl
    prompt_path = (folder / 'expert/prompts.jsonl').resolve()

    def iterator(path):
        rows = original_iter(path)
        if Path(path).resolve() == prompt_path:
            return (r for r in rows if r['record_id'] in selected)
        return rows

    with patch.object(inference, 'iter_jsonl', iterator):
        yield selected


def _merge_parent(root, folder, selected):
    parent = root / read_json(folder / 'session.json')['identity']['parent_test_run']
    expert = folder / 'expert'
    prompts = list(iter_jsonl(expert / 'prompts.jsonl'))
    for name in ('predictions.jsonl', 'raw_responses.jsonl'):
        old = {r['record_id']: r for r in iter_jsonl(parent / name)}
        new = {r['record_id']: r for r in iter_jsonl(expert / name)}
        if set(new) != selected or set(old) != {p['record_id'] for p in prompts}:
            raise ValueError('Prediction/raw cohort incomplete during merge')
        merged = []
        for p in prompts:
            row = (new if p['record_id'] in selected else old)[p['record_id']]
            if row['prompt_sha256'] != p['prompt_sha256'] or row['code_sha256'] != p['code_sha256']:
                raise ValueError('Retained response prompt/code differs')
            merged.append(row)  # preserve the original prediction/raw record verbatim
        write_jsonl(expert / name, merged)


def preflight(root, cfg, settings, session=DEFAULT_SESSION):
    root = Path(root).resolve()
    folder = _bound_session(root, cfg, session, settings)
    expert = folder / 'expert'
    with _provider(settings), _private_cache(expert), _selected_prompts(folder) as selected:
        plan = inference.plan_inference(root, expert)
    plan['total_queries'] = plan['queries']
    plan['queries'] = len(selected)
    plan['preserved_parent_queries'] = plan['total_queries'] - len(selected)
    write_json(expert / 'inference_plan.json', plan)
    return plan


def infer(root, cfg, settings, session=DEFAULT_SESSION, *, workers=4, allow_api=False):
    root = Path(root).resolve()
    folder = _bound_session(root, cfg, session, settings)
    expert = folder / 'expert'
    with _provider(settings), _private_cache(expert), _selected_prompts(folder) as selected:
        report = inference.run_inference(root, expert, workers=workers, allow_api=allow_api)
    if report.get('status') in {'complete', 'partial_request_errors'}:
        _merge_parent(root, folder, selected)
        report['selected_queries'] = len(selected)
        report['preserved_parent_queries'] = report['queries'] - len(selected)
        report['predictions_sha256'] = file_hash(expert / 'predictions.jsonl')
        counts = Counter(str(p['prediction']) for p in iter_jsonl(expert / 'predictions.jsonl'))
        report['prediction_counts'] = dict(counts)
        report['context_limit_utf8_bytes'] = LIMIT
        # Diagnostics only: never select/retry a run based on test accuracy/F1.
        valid = counts['0'] + counts['1']
        report['near_constant_output'] = bool(valid and max(counts['0'], counts['1']) / valid >= .99)
        report['note'] = ('Inspect raw responses if output is near constant; a fresh call does not guarantee better metrics. '
                          'Do not repeat/select attempts to maximize test scores.')
        write_json(expert / 'inference_summary.json', report)
    return report


@contextmanager
def _session_stages(root, cfg, folder):
    """Reuse the real predictor/evaluator, with Devign-only session stage files."""
    expected = read_json(folder / 'session.json')['identity']['experiment']

    def load_stage(_root, _cfg, name):
        result = read_json(folder / (name + '.json'))
        if result['experiment'] != expected:
            raise ValueError('Session experiment changed')
        return result

    with patch.object(workflow, 'stage_file', lambda _root, _cfg, name: folder / (name + '.json')), \
            patch.object(workflow, 'load_stage', load_stage):
        yield


def evaluate(root, cfg, settings, session=DEFAULT_SESSION):
    root = Path(root).resolve()
    folder = _bound_session(root, cfg, session, settings)
    report = read_json(folder / 'expert/inference_summary.json')
    if report['status'] != 'complete':
        raise ValueError('Resume failed requests before evaluation')
    local_cfg = deepcopy(cfg)
    local_cfg['datasets'] = ['devign']
    with _provider(settings), _session_stages(root, cfg, folder):
        workflow.predict(root, local_cfg)
        summary = workflow.evaluate(root, local_cfg)
    summary['inference_rerun'] = read_json(folder / 'session.json')['identity']
    summary['effective_context_limits'] = {'test_utf8_bytes': LIMIT, 'validation_utf8_bytes': LIMIT}
    summary['models_refitted'] = False
    # Existing validation is mixed Qwen/Vertex + DeepSeek/FPT and stays frozen.
    summary['benchmark_comparable'] = False
    summary['limitations'] = ('Devign test-only rerun; frozen validation includes Qwen/Vertex and DeepSeek/FPT. '
                              'This rerun aligns test context/model, not the validation backbone. '
                              'ReVeal and original benchmark reports are preserved.')
    for item in summary['datasets']:
        item['effective_context_limits'] = summary['effective_context_limits']
        item['limitations'] = summary['limitations']
        write_json(folder / 'predictions' / item['dataset'] / str(item['seed']) / 'evaluation.json', item)
    write_json(folder / 'benchmark_summary.json', summary)
    return summary
