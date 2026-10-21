"""Derive an 8192-byte test prompt profile without refitting H3 models.

This tool lives outside the hashed learning packages. It preserves their source
identity and records its own checksum in each derived run's prepare manifest.
No API requests, model fitting, Joern parsing, or test-label reads occur here.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time
import yaml

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from grace_h1.data import load_rows
from grace_h1.graphs import GraphStore
from grace_h1.io import read_json, write_json, write_jsonl, iter_jsonl, file_hash, fingerprint, sha256, row_identity
from grace_h1.pipeline import INSTRUCTION, verify_frozen
from grace_h1.reference import raw_graph_context, fit_contracted_prompt
from grace_h3v2 import workflow as workflow


def _derive_run(root, parent_rel, limit):
    parent = root / parent_rel
    old_manifest = verify_frozen(root, parent)
    if old_manifest.get('effective_context_limit_utf8_bytes') == limit:
        prompts = list(iter_jsonl(parent / 'prompts.jsonl'))
        if any(p['prompt_truncation']['context_limit'] != limit or
               len(p['prompt'].encode('utf-8')) > limit for p in prompts):
            raise ValueError('Frozen test prompt does not match its declared context profile')
        return parent_rel, {'reused_preparation': True, 'queries': len(prompts),
                            'context_limit_utf8_bytes': limit}
    if old_manifest['config']['split'] != 'test':
        raise ValueError('Context conversion accepts test runs only')
    contract = yaml.safe_load((root / 'configs/reference_contract.yaml').read_text(encoding='utf-8'))
    if limit > int(contract['prompt_context_policy']['context_limit']):
        raise ValueError('Experiment limit cannot exceed the shared contract upper bound')
    profile = {'schema': 'h3_test_context_profile_v1', 'limit_utf8_bytes': limit,
               'parent_run': parent_rel, 'parent_prepare_fingerprint': old_manifest['prepare_fingerprint'],
               'tool_sha256': file_hash(Path(__file__)),
               'fitter_sha256': file_hash(root / 'grace_h1/reference.py')}
    inputs = deepcopy(old_manifest['inputs'])
    inputs['effective_context_profile'] = profile
    signature = fingerprint(inputs)
    output = parent.parent / signature[:16]
    if (output / 'freeze.json').is_file():
        existing = verify_frozen(root, output)
        if existing['prepare_fingerprint'] != signature:
            raise ValueError('Derived test run identity mismatch')
        return output.relative_to(root).as_posix(), existing['context_profile_audit']
    started = time.perf_counter()
    original = list(iter_jsonl(parent / 'prompts.jsonl'))
    dataset = old_manifest['config']['dataset']
    queries = {row['record_id']: row for row in load_rows(root, dataset, 'test')}
    train = {row['record_id']: row for row in load_rows(root, dataset, 'train', labels=True)}
    graphs = GraphStore(root)
    template = (root / 'configs/prompt_template.txt').read_text(encoding='utf-8')
    truncation = contract['prompt_context_policy']['truncation_policy']
    new_policy = dict(truncation, context_limit=limit)
    prompts = []
    changed = 0
    before_max = 0
    for i, previous in enumerate(original, 1):
        query = queries[previous['record_id']]
        if query['code_sha256'] != previous['code_sha256']:
            raise ValueError('Query code identity changed')
        demos = [train[record_id] for record_id in previous['selected_demo_ids']]
        graph = graphs.get(query['code_sha256'])
        if graph['feature_sha256'] != previous['graph_feature_sha256']:
            raise ValueError('Cached query graph differs from the parent prompt')
        graph_text, graph_counts = raw_graph_context(graph, truncation)
        old_limit = int(previous['prompt_truncation']['context_limit'])
        old_prompt, _ = fit_contracted_prompt(template, query['code'], graph_text, demos,
                                             INSTRUCTION, dict(truncation, context_limit=old_limit))
        if old_prompt != previous['prompt'] or sha256(old_prompt) != previous['prompt_sha256']:
            raise ValueError(f"Cannot reproduce parent prompt exactly: {previous['record_id']}")
        text, metadata = fit_contracted_prompt(template, query['code'], graph_text, demos,
                                               INSTRUCTION, new_policy)
        if len(text.encode('utf-8')) > limit:
            raise ValueError('Derived prompt exceeds its byte limit')
        changed += text != previous['prompt']
        before_max = max(before_max, len(old_prompt.encode('utf-8')))
        prompts.append({**previous, 'prompt': text, 'prompt_sha256': sha256(text),
                        'prompt_truncation': metadata, 'graph_counts': graph_counts,
                        'parent_prompt_sha256': previous['prompt_sha256'],
                        'effective_context_limit_utf8_bytes': limit})
        if i % 500 == 0 or i == len(original):
            print(f'{dataset}: rerendered {i}/{len(original)} cached prompts; changed={changed}', flush=True)
    if row_identity(prompts) != old_manifest['target_identity']:
        raise ValueError('Derived run cohort/order changed')
    output.mkdir(parents=True, exist_ok=True)
    for filename in ('retrieval.jsonl', 'graph_feature_index.jsonl'):
        shutil.copy2(parent / filename, output / filename)
        if file_hash(parent / filename) != file_hash(output / filename):
            raise ValueError('Copied preparation artifact differs')
    write_jsonl(output / 'prompts.jsonl', prompts)
    audit = {'queries': len(prompts), 'changed_prompts': changed,
             'unchanged_prompts': len(prompts) - changed, 'context_limit_utf8_bytes': limit,
             'parent_max_prompt_bytes': before_max,
             'max_prompt_bytes': max(len(p['prompt'].encode('utf-8')) for p in prompts),
             'query_labels_read': False, 'graph_builds': 0, 'llm_calls': 0,
             'parent_prompts_reproduced_exactly': True}
    manifest = {**old_manifest, 'inputs': inputs, 'prepare_fingerprint': signature,
                'parent_run': parent_rel, 'effective_context_limit_utf8_bytes': limit,
                'context_profile': profile, 'context_profile_audit': audit,
                'parent_prepare_seconds': old_manifest['prepare_seconds'],
                'prepare_seconds': time.perf_counter() - started,
                'graph_built': 0, 'graph_reused': len(prompts)}
    write_json(output / 'run_manifest.json', manifest)
    locked = [output / name for name in ('prompts.jsonl', 'retrieval.jsonl',
                                        'graph_feature_index.jsonl', 'run_manifest.json')]
    locked += [root / name for name in ('configs/reference_contract.yaml',
                                       'configs/prompt_template.txt', 'data/manifest.json')]
    write_json(output / 'freeze.json', {'files': {p.relative_to(root).as_posix(): file_hash(p)
                                                 for p in locked}})
    verify_frozen(root, output)
    return output.relative_to(root).as_posix(), audit


def prepare(root, cfg, limit=8192):
    root = Path(root).resolve()
    if int(limit) != 8192:
        raise ValueError('This registered profile must match validation: exactly 8192 UTF-8 bytes')
    stage_path = workflow.stage_file(root, cfg, 'meta_selection')
    stage = workflow.load_stage(root, cfg, 'meta_selection')
    new_stage = deepcopy(stage)
    jobs = []
    prepared = {}
    changed = False
    for job in stage['jobs']:
        parent = job['test_run']
        if parent not in prepared:
            prepared[parent] = _derive_run(root, parent, limit)
        run, audit = prepared[parent]
        changed |= run != parent
        jobs.append({**job, 'test_run': run,
                     'parent_test_run': job.get('parent_test_run', parent)})
    if changed:
        # Archive test24k evaluations before the active stage points to test8192.
        tag = fingerprint(stage)[:16]
        archive = stage_path.parent / 'archive' / ('before_test_context8192_' + tag)
        archive.mkdir(parents=True, exist_ok=True)
        for name in ('meta_selection.json', 'predictions_selection.json', 'benchmark_summary.json'):
            source = stage_path.parent / name
            if source.is_file():
                destination = archive / name
                if destination.exists() and file_hash(destination) != file_hash(source):
                    raise ValueError('Archive file already exists with different bytes')
                if not destination.exists():
                    shutil.copy2(source, destination)
        source_predictions = stage_path.parent / 'predictions'
        if source_predictions.is_dir():
            for source in source_predictions.rglob('*'):
                if source.is_file():
                    destination = archive / source.relative_to(stage_path.parent)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists() and file_hash(destination) != file_hash(source):
                        raise ValueError('Prediction archive differs')
                    if not destination.exists():
                        shutil.copy2(source, destination)
        new_stage['jobs'] = jobs
        new_stage['test_context_profile'] = {'limit_utf8_bytes': limit,
                                            'previous_stage_archive': archive.relative_to(root).as_posix()}
        write_json(stage_path, new_stage)
        # Reports from the old test profile must not appear as current results.
        for name in ('benchmark_summary.json', 'predictions_selection.json'):
            source = stage_path.parent / name
            if source.is_file():
                if file_hash(source) != file_hash(archive / name):
                    raise ValueError('Report changed during preparation')
                source.unlink()
    result = {'schema': 'h3_test_context_profile_selection_v1', 'context_limit_utf8_bytes': limit,
              'stage': stage_path.relative_to(root).as_posix(),
              'runs': [{'parent': parent, 'test_run': run, **audit}
                       for parent, (run, audit) in prepared.items()],
              'api_calls': 0, 'models_refitted': False}
    write_json(stage_path.parent / 'test_context_profile.json', result)
    return result


def annotate_summary(root, cfg, summary):
    root = Path(root).resolve()
    stage = workflow.load_stage(root, cfg, 'meta_selection')
    profile = stage.get('test_context_profile')
    if not profile or profile['limit_utf8_bytes'] != 8192:
        raise ValueError('Test stage has not switched to the 8192-byte profile')
    limits = {'validation_utf8_bytes': 8192, 'test_utf8_bytes': 8192,
              'shared_contract_upper_bound_utf8_bytes': 24000}
    summary['effective_context_limits'] = limits
    summary['limitations'] = ('Exploratory mixed-model recovery: validation combines Qwen/Vertex '
                              'and DeepSeek/FPT. Both validation and test prompts use 8192 UTF-8 bytes. '
                              'Do not compare as a single-backbone benchmark.')
    summary['benchmark_comparable'] = False
    for report in summary['datasets']:
        report['effective_context_limits'] = limits
    predictions_stage = workflow.load_stage(root, cfg, 'predictions_selection')
    by_job = {(item['dataset'], item['seed']): item for item in predictions_stage['jobs']}
    for report in summary['datasets']:
        job = by_job[report['dataset'], report['seed']]
        write_json(root / job['prediction_folder'] / 'evaluation.json', report)
    write_json(workflow.stage_file(root, cfg, 'benchmark_summary'), summary)
    return summary


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    print(json.dumps(prepare(root, workflow.config(root)), indent=2))
