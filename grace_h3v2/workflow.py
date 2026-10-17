"""Explicit CPU/local handoffs; no implicit LLM call or test-driven selection."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from grace_h1.io import resolve_root, read_json, write_json, write_jsonl, iter_jsonl, file_hash, row_identity
from grace_h1.data import load_rows
from grace_h1.pipeline import prepare_run, PipelineConfig, verify_frozen
from grace_h1.inference import plan_inference, run_inference, request_spec
from grace_h1.metrics import paired_comparison
from .common import experiment_dir, sources, metrics, versions, freeze, verify
from .base import train_base, score_queries
from .meta import fit_meta, predict_all


def config(root):
    return read_json(root / 'configs/benchmark_v2.json')


def stage_file(root, cfg, name):
    folder, _ = experiment_dir(root, cfg)
    return folder / (name + '.json')


def load_stage(root, cfg, name):
    path = stage_file(root, cfg, name)
    if not path.exists():
        raise FileNotFoundError(f'Missing {name}; complete its notebook handoff first')
    result = read_json(path)
    _, spec = experiment_dir(root, cfg)
    if result['experiment'] != spec:
        raise ValueError('Stage config/source/data/contract differs')
    return result


def expert_config(dataset, split, cfg):
    return PipelineConfig(dataset=dataset, split=split, arm=cfg['expert']['arm'],
                          graph_workers=cfg['expert']['graph_workers'],
                          semantic_top_k=5, lexical_weight=.7, whitening_dimension=256)


def checked_expert(root, run, expected_rows):
    folder = root / run
    manifest = verify_frozen(root, folder)
    if manifest['target_identity'] != row_identity(expected_rows):
        raise ValueError('Expert run cohort differs')
    summary_path = folder / 'inference_summary.json'
    predictions = folder / 'predictions.jsonl'
    if not summary_path.exists() or not predictions.exists():
        raise FileNotFoundError('Run expert inference locally before this CPU stage')
    summary = read_json(summary_path)
    if summary['status'] != 'complete' or summary['request_spec'] != request_spec(root):
        raise ValueError('Expert requests incomplete or inference spec changed; resume locally')
    if file_hash(predictions) != summary['predictions_sha256']:
        raise ValueError('Expert prediction file changed')
    raw = list(iter_jsonl(predictions))
    lookup = {r['record_id']: r for r in raw}
    if len(lookup) != len(raw) or set(lookup) != {r['record_id'] for r in expected_rows}:
        raise ValueError('Expert IDs incomplete')
    for row in expected_rows:
        p = lookup[row['record_id']]
        if p['code_sha256'] != row['code_sha256']:
            raise ValueError('Expert query code differs')
    return [lookup[r['record_id']] for r in expected_rows], {
        'run': run, 'prepare_fingerprint': manifest['prepare_fingerprint'],
        'predictions_sha256': file_hash(predictions),
        'request_spec': summary['request_spec']}


def base_prepare(root, cfg, callback=None):
    path = stage_file(root, cfg, 'base_selection')
    if path.exists():
        result = load_stage(root, cfg, 'base_selection')
        from .common import verify
        for job in result['jobs']:
            verify(root, root / job['base_folder'])
            verify_frozen(root, root / job['validation_run'])
        return result
    jobs = []
    for dataset in cfg['datasets']:
        validation_run = None
        for seed in cfg['seeds']:
            model = train_base(root, dataset, seed, cfg, callback)
            if validation_run is None:
                full = load_rows(root, dataset, 'validation')
                ids = [full[i]['record_id'] for i in model['spec']['validation_indices']]
                run = prepare_run(root, expert_config(dataset, 'validation', cfg),
                                  target_ids=ids, checkpoint_callback=callback)
                validation_run = run.relative_to(root).as_posix()
            jobs.append({'dataset': dataset, 'seed': seed, 'base_folder': model['folder'],
                         'validation_run': validation_run})
    _, spec = experiment_dir(root, cfg)
    result = {'experiment': spec, 'jobs': jobs, 'stage': 'base_prepared',
              'next': 'validation-preflight then validation-infer on local LLM environment'}
    write_json(path, result)
    return result


def preflight(root, cfg, phase):
    stage = load_stage(root, cfg, 'base_selection' if phase == 'validation' else 'meta_selection')
    paths = sorted({j[phase + '_run'] for j in stage['jobs']})
    return [{'run': rel, **plan_inference(root, root / rel)} for rel in paths]


def infer(root, cfg, phase, allow_api=False):
    stage = load_stage(root, cfg, 'base_selection' if phase == 'validation' else 'meta_selection')
    paths = sorted({j[phase + '_run'] for j in stage['jobs']})
    reports = []
    for rel in paths:
        previous = root / rel / 'inference_summary.json'
        predictions = root / rel / 'predictions.jsonl'
        if previous.exists() and predictions.exists():
            old = read_json(previous)
            verify_frozen(root, root / rel)
            if (old['status'] == 'complete' and old['request_spec'] == request_spec(root)
                    and old['predictions_sha256'] == file_hash(predictions)):
                reports.append({'run': rel, 'status': 'complete', 'reused_completed_run': True,
                                'api_attempts_this_run': 0, 'new_unique_requests': 0,
                                'stored_inference_summary': previous.relative_to(root).as_posix()})
                continue
        report = run_inference(root, root / rel, allow_api=allow_api,
                               workers=cfg['inference_workers'])
        reports.append({'run': rel, **report})
    return reports


def meta_prepare(root, cfg, callback=None):
    initial = load_stage(root, cfg, 'base_selection')
    path = stage_file(root, cfg, 'meta_selection')
    if path.exists():
        result = load_stage(root, cfg, 'meta_selection')
        from .common import verify
        for job in result['jobs']:
            verify(root, root / job['meta_folder'])
            verify_frozen(root, root / job['test_run'])
        return result
    jobs = []
    test_runs = {}
    for job in initial['jobs']:
        base = read_json(root / job['base_folder'] / 'manifest.json')
        full = load_rows(root, job['dataset'], 'validation')
        rows = [full[i] for i in base['spec']['validation_indices']]
        expert, identity = checked_expert(root, job['validation_run'], rows)
        meta = fit_meta(root, base, expert, identity, cfg, callback)
        dataset = job['dataset']
        score_folder = root / 'artifacts/h3v2/test_scores' / dataset / str(job['seed']) / meta['fingerprint'][:20]
        if (score_folder / 'freeze.json').exists():
            verify(root, score_folder)
        else:
            score_rows, scores = score_queries(root, base, dataset, 'test')
            score_folder.mkdir(parents=True, exist_ok=True)
            np.save(score_folder / 'views.npy', scores, allow_pickle=False)
            write_json(score_folder / 'manifest.json',
                       {'target_identity': row_identity(score_rows), 'base': base['fingerprint'],
                        'shape': list(scores.shape), 'labels_used': False})
            freeze(root, score_folder, [score_folder / 'views.npy', score_folder / 'manifest.json'],
                   {'source': sources(root), 'base': base['fingerprint']})
            if callback:
                callback()
        if dataset not in test_runs:
            # Label-blind preparation. Test labels open only in evaluate().
            run = prepare_run(root, expert_config(dataset, 'test', cfg),
                              checkpoint_callback=callback)
            test_runs[dataset] = run.relative_to(root).as_posix()
        jobs.append(job | {'meta_folder': meta['folder'], 'test_run': test_runs[dataset],
                           'test_scores': score_folder.relative_to(root).as_posix(),
                           'development_winner': meta['winner']['name']})
    _, spec = experiment_dir(root, cfg)
    result = {'experiment': spec, 'jobs': jobs, 'stage': 'models_locked_test_prepared',
              'next': 'test-preflight then test-infer local; evaluate afterward'}
    write_json(path, result)
    return result


def predict(root, cfg):
    stage = load_stage(root, cfg, 'meta_selection')
    reports = []
    for job in stage['jobs']:
        base = read_json(root / job['base_folder'] / 'manifest.json')
        meta = read_json(root / job['meta_folder'] / 'manifest.json')
        verify(root, root / base['folder'])
        score_folder = root / job['test_scores']
        verify(root, score_folder)
        rows = load_rows(root, job['dataset'], 'test')
        score_manifest = read_json(score_folder / 'manifest.json')
        if score_manifest['target_identity'] != row_identity(rows) or score_manifest['base'] != base['fingerprint']:
            raise ValueError('CPU-prepared test features differ')
        scores = np.load(score_folder / 'views.npy', allow_pickle=False)
        if list(scores.shape) != score_manifest['shape'] or not np.isfinite(scores).all():
            raise ValueError('Invalid prepared test features')
        records, identity = checked_expert(root, job['test_run'], rows)
        expert = [-1 if p['prediction'] is None else p['prediction'] for p in records]
        arms, probability = predict_all(root, base, meta, rows, scores, expert)
        folder = stage_file(root, cfg, 'meta_selection').parent / 'predictions' / job['dataset'] / str(job['seed'])
        folder.mkdir(parents=True, exist_ok=True)
        for arm, result in arms.items():
            output = [{'record_id': r['record_id'], 'code_sha256': r['code_sha256'],
                       'prediction': result['prediction'][i], 'gate_score': float(probability[i]),
                       'expert_required': result['expert_required'][i],
                       'expert_parse_invalid': expert[i] < 0, 'gate_fallback': result['fallback'][i]}
                      for i, r in enumerate(rows)]
            write_jsonl(folder / (arm + '.jsonl'), output)
        report = job | {'prediction_folder': folder.relative_to(root).as_posix(),
                        'target_identity': row_identity(rows), 'expert_identity': identity,
                        'locked_winner': meta['winner']['name'],
                        'files': {arm: file_hash(folder / (arm + '.jsonl')) for arm in arms}}
        write_json(folder / 'manifest.json', report)
        reports.append(report)
    write_json(stage_file(root, cfg, 'predictions_selection'),
               {'experiment': stage['experiment'], 'jobs': reports})
    return reports


def evaluate(root, cfg):
    stage = load_stage(root, cfg, 'predictions_selection')
    reports = []
    for job in stage['jobs']:
        rows = load_rows(root, job['dataset'], 'test', labels=True)
        if row_identity(rows) != job['target_identity']:
            raise ValueError('Evaluation cohort changed')
        y = [r['label'] for r in rows]
        folder = root / job['prediction_folder']
        arms, outputs = {}, {}
        for arm, checksum in job['files'].items():
            path = folder / (arm + '.jsonl')
            if file_hash(path) != checksum:
                raise ValueError('Predictions changed before evaluation')
            p = list(iter_jsonl(path))
            if row_identity(p) != row_identity(rows):
                raise ValueError('Prediction ordering/code identity differs')
            values = [r['prediction'] for r in p]
            outputs[arm] = values
            arms[arm] = {'metrics': metrics(y, values),
                         'algorithmic_expert_query_ratio': sum(r['expert_required'] for r in p)/len(p),
                         'fallback_n': sum(r['gate_fallback'] for r in p)}
        winner = job['locked_winner']  # Never choose the best test result.
        report = {'dataset': job['dataset'], 'seed': job['seed'], 'n': len(y),
                  'locked_development_winner': winner, 'arms': arms,
                  'trivial': {'always_safe': metrics(y, [0]*len(y)),
                              'always_vulnerable': metrics(y, [1]*len(y))},
                  'paired_full_grace_to_winner': paired_comparison(
                      y, outputs['full_grace'], outputs[winner], bootstrap=cfg['bootstrap'], seed=job['seed']),
                  'paired_gate_to_winner': paired_comparison(
                      y, outputs['gate_tuned'], outputs[winner], bootstrap=cfg['bootstrap'], seed=job['seed']),
                  'inference_artifact_creation': read_json(root / job['test_run'] / 'inference_summary.json'),
                  'cost_note': 'All-query reference/stacking benchmark calls are shared; routing ratios are hypothetical standalone-arm requirements, not actual savings of this combined run.',
                  'status': 'complete'}
        base = read_json(root / job['base_folder'] / 'manifest.json')
        meta = read_json(root / job['meta_folder'] / 'manifest.json')
        report['cpu_artifact_creation'] = {
            'base_fit_seconds': base['fit_seconds'], 'base_resources': base['resources'],
            'meta_fit_seconds': meta['fit_seconds'], 'meta_resources': meta['resources'],
            'scope': 'Stored creation timings, not fresh latency after reuse'}
        write_json(folder / 'evaluation.json', report)
        reports.append(report)
    aggregate = {}
    for dataset in cfg['datasets']:
        subset = [r for r in reports if r['dataset'] == dataset]
        aggregate[dataset] = {name: {'mean': float(np.mean([
            r['arms'][r['locked_development_winner']]['metrics'][name] for r in subset])),
            'sd': float(np.std([r['arms'][r['locked_development_winner']]['metrics'][name]
                                for r in subset], ddof=1)) if len(subset)>1 else None}
            for name in ('mcc', 'f1', 'precision', 'recall', 'specificity')}
    summary = {'schema': 'h3v2_benchmark_summary_v1', 'config': cfg, 'status': 'complete',
               'datasets': reports, 'per_dataset_selected_mean_sd': aggregate,
               'selection': 'MCC/F1 on development, winner frozen before test',
               'limitations': 'Iterative benchmark, one seed by default, query bootstrap not project-adjusted.',
               'created_unix': time.time()}
    write_json(stage_file(root, cfg, 'benchmark_summary'), summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('base-prepare', 'validation-preflight', 'validation-infer',
                                         'meta-prepare', 'test-preflight', 'test-infer', 'predict', 'evaluate'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    root = resolve_root(args.root)
    cfg = config(root)
    if args.stage == 'base-prepare':
        result = base_prepare(root, cfg)
    elif args.stage == 'meta-prepare':
        result = meta_prepare(root, cfg)
    elif args.stage.endswith('preflight'):
        result = preflight(root, cfg, args.stage.split('-')[0])
    elif args.stage.endswith('infer'):
        result = infer(root, cfg, args.stage.split('-')[0], args.allow_api)
    elif args.stage == 'predict':
        result = predict(root, cfg)
    else:
        result = evaluate(root, cfg)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
