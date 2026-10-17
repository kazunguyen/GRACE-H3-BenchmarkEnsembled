from pathlib import Path
import importlib.metadata
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from grace_h1.io import read_json, write_json, file_hash, fingerprint, row_identity
from grace_h1.data import load_rows, load_embeddings, embedding_recipe
from grace_h1.metrics import binary_metrics


def sources(root):
    return {p.relative_to(root).as_posix(): file_hash(p)
            for package in ('grace_h1', 'grace_h3v2')
            for p in sorted((root / package).glob('*.py'))}


def validate_config(cfg):
    if not cfg['datasets'] or not set(cfg['datasets']) <= {'devign', 'reveal'}:
        raise ValueError('Choose Devign and/or ReVeal')
    if len(set(cfg['datasets'])) != len(cfg['datasets']) or not cfg['seeds']:
        raise ValueError('Duplicate dataset or empty seeds')
    if len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Duplicate seeds')
    if cfg['selection']['primary'] != 'mcc' or cfg['selection']['secondary'] != 'f1':
        raise ValueError('This experiment selects MCC, then F1')
    if cfg['selection']['cost_penalty'] != 0:
        raise ValueError('Benchmark mode requires cost_penalty=0')
    if min(cfg['folds'], cfg['inner_folds'], cfg['meta_folds']) < 2:
        raise ValueError('At least two folds required')
    if cfg['group_key'] not in ('code_sha256', 'project', 'commit'):
        raise ValueError('Unsupported grouping key')
    if cfg['expert']['arm'] != 'baseline_whitening':
        raise ValueError('H1 expert is a subsequent registered experiment, not this default')
    if cfg['inference_workers'] < 1 or cfg['expert']['graph_workers'] < 1:
        raise ValueError('Worker counts must be positive')


def experiment_dir(root, cfg):
    validate_config(cfg)
    spec = {'config': cfg, 'source': sources(root),
            'data_manifest': file_hash(root / 'data/manifest.json'),
            'contract': file_hash(root / 'configs/reference_contract.yaml'),
            'template': file_hash(root / 'configs/prompt_template.txt')}
    return root / 'runs/h3v2' / fingerprint(spec)[:20], spec


def groups(rows, key):
    if key != 'code_sha256' and any(not r.get(key) for r in rows):
        raise ValueError(f'No complete {key} metadata; choose code_sha256 explicitly')
    return np.asarray([str(r[key]) for r in rows])


def folds(rows, y, count, seed, key):
    g = groups(rows, key)
    if len(set(g)) < count or min(np.bincount(y, minlength=2)) < count:
        raise ValueError('Not enough groups/positive samples for these folds')
    result = list(StratifiedGroupKFold(count, shuffle=True, random_state=seed)
                  .split(np.zeros(len(y)), y, g))
    for a, b in result:
        if set(g[a]) & set(g[b]) or len(np.unique(y[a])) != 2 or len(np.unique(y[b])) != 2:
            raise ValueError('Fold lacks both classes or crosses groups; change registered design')
    return result


def eligible(rows, excluded=()):
    blocked = set(excluded)
    by_hash = {}
    for i, row in enumerate(rows):
        by_hash.setdefault(row['code_sha256'], []).append(i)
    return np.asarray([ids[0] for h, ids in by_hash.items()
                       if h not in blocked and len({rows[i]['label'] for i in ids}) == 1],
                      dtype=np.int64)


def training_data(root, dataset):
    train = load_rows(root, dataset, 'train', labels=True)
    valid = load_rows(root, dataset, 'validation', labels=True)
    x, tm = load_embeddings(root, dataset, 'train', train)
    v, vm = load_embeddings(root, dataset, 'validation', valid)
    if embedding_recipe(tm) != embedding_recipe(vm):
        raise ValueError('Encoder recipes differ')
    ti = eligible(train)
    vi = eligible(valid, (r['code_sha256'] for r in train))
    if not len(ti) or not len(vi):
        raise ValueError('No eligible learning data')
    return train, valid, x, v, ti, vi, tm, vm


def metrics(y, p):
    m = binary_metrics(y, p)
    cm = m['confusion_matrix_valid_only']
    m['specificity'] = cm['tn'] / max(1, cm['tn'] + cm['fp'])
    return m


def choose(records, cfg):
    """Global tolerance bands avoid order-dependent pairwise tie comparisons."""
    tol = cfg['selection']['tie_tolerance']
    if not records or tol < 0:
        raise ValueError('No candidates or invalid tolerance')
    top = max(r['metrics']['mcc'] for r in records)
    band = [r for r in records if r['metrics']['mcc'] >= top - tol]
    top_f1 = max(r['metrics']['f1'] for r in band)
    band = [r for r in band if r['metrics']['f1'] >= top_f1 - tol]
    return min(band, key=lambda r: (r.get('cost_rank', 0), r['name']))


def best_threshold(y, score, cfg, name='threshold'):
    rows = [{'name': f'{name}:{i:03}', 'threshold': t,
             'metrics': metrics(y, (score >= t).astype(int))}
            for i, t in enumerate(cfg['threshold_grid'])]
    return choose(rows, cfg)


def versions():
    return {n: importlib.metadata.version(n)
            for n in ('numpy', 'scipy', 'scikit-learn', 'joblib')}


def freeze(root, folder, files, spec):
    paths = sorted(set(Path(p) for p in files))
    write_json(folder / 'freeze.json',
               {'spec': spec, 'files': {p.relative_to(root).as_posix(): file_hash(p) for p in paths}})


def verify(root, folder):
    frozen = read_json(folder / 'freeze.json')
    if frozen['spec']['source'] != sources(root):
        raise ValueError('H3-v2 source changed; create a new experiment')
    for rel, expected in frozen['files'].items():
        p = (root / rel).resolve()
        p.relative_to(root)
        if file_hash(p) != expected:
            raise ValueError(f'Frozen artifact changed: {rel}')
    return frozen
