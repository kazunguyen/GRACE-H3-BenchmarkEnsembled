import warnings
import time
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from grace_h1.io import read_json, write_json, write_jsonl, file_hash, fingerprint, row_identity
from grace_h1.data import load_rows
from .common import sources, folds, metrics, choose, best_threshold, freeze, verify, versions, resource_usage
from .features import VIEWS, numeric
from .base import arm_scores, save_model, load_model, model_entry

META_ARMS = ('benefit_router', 'post_selector', 'stack_lr', 'stack_mlp')


def pre_features(rows, scores, probability):
    logits = logit(np.clip(scores, 1e-6, 1-1e-6))
    return np.column_stack([logits, probability, np.abs(probability-.5),
                            np.abs(scores[:, VIEWS.index('semantic_mlp')]
                                   - scores[:, VIEWS.index('lexical')]), numeric(rows)])


def post_features(pre, expert, probability):
    valid = expert >= 0
    value = np.where(valid, expert, .5)
    return np.column_stack([pre, value, valid.astype(float), value*probability,
                            value-probability])


class MetaModel:
    def __init__(self, name, seed, cfg):
        self.name, self.seed, self.cfg = name, seed, cfg

    def fit(self, features, y, h, expert):
        correct_gate = h == y
        correct_expert = expert == y
        if self.name == 'benefit_router':
            target = (~correct_gate).astype(float) - (~correct_expert).astype(float)
            self.model = make_pipeline(StandardScaler(), Ridge(alpha=10.))
            self.model.fit(features, target)
        elif self.name == 'post_selector':
            active = correct_gate != correct_expert
            target = correct_expert[active].astype(int)
            if not active.any() or len(np.unique(target)) < 2:
                self.constant = float(target.mean()) if len(target) else 0.
            else:
                self.model = make_pipeline(StandardScaler(),
                                           LogisticRegression(C=1., max_iter=2000))
                self.model.fit(features[active], target)
        else:
            if self.name == 'stack_lr':
                model = LogisticRegression(C=1., max_iter=2000, random_state=self.seed)
            else:
                model = MLPClassifier(hidden_layer_sizes=(32,), alpha=.01,
                                      batch_size=128, max_iter=self.cfg['meta_epochs'],
                                      early_stopping=False, random_state=self.seed)
            self.model = make_pipeline(StandardScaler(), model)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                self.model.fit(features, y)
                self.convergence_warnings = [str(w.message) for w in caught]
        return self

    def score(self, features):
        if hasattr(self, 'constant'):
            return np.full(len(features), self.constant)
        if self.name == 'benefit_router':
            return self.model.predict(features)
        return self.model.predict_proba(features)[:, 1]


def output_for(name, score, threshold, h, expert):
    if name.startswith('stack_'):
        output = (score >= threshold).astype(int)
        required = np.ones(len(h), dtype=bool)
    else:
        use_expert = score >= threshold
        output = np.where(use_expert, expert, h)
        required = np.ones(len(h), dtype=bool) if name == 'post_selector' else use_expert
    fallback = ((score >= threshold) if name == 'post_selector' else required) & (expert < 0)
    output[fallback] = h[fallback]
    return output.astype(int), required, fallback


def tune_policy(name, y, score, h, expert, cfg):
    grid = cfg['benefit_grid'] if name == 'benefit_router' else cfg['threshold_grid']
    records = []
    for i, t in enumerate(grid):
        p, required, _ = output_for(name, score, t, h, expert)
        records.append({'name': f'{name}:{i:03}', 'threshold': t, 'metrics': metrics(y, p),
                        'cost_rank': float(required.mean())})
    return choose(records, cfg)


def fit_meta(root, base, expert_records, expert_identity, cfg, callback=None):
    verify(root, root / base['folder'])
    dataset, seed = base['spec']['dataset'], base['spec']['seed']
    all_rows = load_rows(root, dataset, 'validation', labels=True)
    indices = base['spec']['validation_indices']
    rows = [all_rows[i] for i in indices]
    if len(expert_records) != len(rows):
        raise ValueError('Expert validation count mismatch')
    by_id = {r['record_id']: r for r in expert_records}
    if len(by_id) != len(rows) or set(by_id) != {r['record_id'] for r in rows}:
        raise ValueError('Expert validation IDs differ')
    for row in rows:
        r = by_id[row['record_id']]
        if r['code_sha256'] != row['code_sha256'] or r['request_status'] != 'ok':
            raise ValueError('Expert code mismatch or request unfinished; resume validation inference')
        if r['prediction'] not in (None, 0, 1):
            raise ValueError('Expert prediction must be 0, 1 or invalid')
    expert_records = [by_id[r['record_id']] for r in rows]
    expert = np.asarray([-1 if r['prediction'] is None else r['prediction'] for r in expert_records])
    y = np.asarray([r['label'] for r in rows], dtype=int)
    scores = np.load(root / base['folder'] / 'validation_views.npy', allow_pickle=False)[indices]
    probability = arm_scores(root, base, scores)[base['winner']['name']]
    # This cutoff was selected solely inside official TRAIN.
    h = (probability >= base['winner']['threshold']).astype(int)
    pre = pre_features(rows, scores, probability)
    post = post_features(pre, expert, probability)
    spec = {'source': sources(root), 'config': cfg, 'versions': versions(),
            'base': base['fingerprint'], 'dataset': dataset, 'seed': seed,
            'expert': expert_identity, 'validation_identity': row_identity(rows)}
    key = fingerprint(spec)
    folder = root / 'artifacts/h3v2/meta' / dataset / str(seed) / key[:20]
    if (folder / 'freeze.json').exists():
        verify(root, folder)
        return read_json(folder / 'manifest.json')
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fold_list = folds(rows, y, cfg['meta_folds'], seed + 909, cfg['group_key'])
    oof = {name: np.empty(len(rows)) for name in META_ARMS}
    fold_entries = []
    for f, (a, b) in enumerate(fold_list):
        for name in META_ARMS:
            features = pre if name == 'benefit_router' else post
            model = MetaModel(name, seed + f, cfg)
            fold_key = fingerprint({'spec': spec, 'fold': f, 'arm': name,
                                    'fit': a.tolist(), 'held': b.tolist()})
            manifest_path = root / 'cache/h3v2/meta_fits' / fold_key[:24] / 'manifest.json'
            if manifest_path.exists():
                entry = read_json(manifest_path)
                model = load_model(root, entry)
            else:
                model.fit(features[a], y[a], h[a], expert[a])
                path = manifest_path.with_name('model.joblib')
                save_model(path, model)
                write_json(manifest_path, model_entry(root, path) | {'fingerprint': fold_key})
                if callback:
                    callback()
            oof[name][b] = model.score(features[b])
        fold_entries.append({'fold': f, 'fit_indices': a.tolist(), 'held_indices': b.tolist()})
    candidates = []
    for i, name in enumerate(META_ARMS):
        policy = tune_policy(name, y, oof[name], h, expert, cfg)
        candidates.append({'name': name, 'threshold': policy['threshold'],
                           'metrics': policy['metrics'], 'cost_rank': policy['cost_rank'],
                           'selection_scope': 'validation meta OOF; tuning is development, not confirmation'})
    gate_tuned = best_threshold(y, probability, cfg, 'gate_tuned')
    candidates += [
        {'name': 'gate_tuned', 'threshold': gate_tuned['threshold'],
         'metrics': gate_tuned['metrics'], 'cost_rank': 0.},
        {'name': 'gate_0_5', 'threshold': .5, 'metrics': metrics(y, (probability >= .5).astype(int)),
         'cost_rank': 0.},
        {'name': 'gate_train_threshold', 'threshold': base['winner']['threshold'],
         'metrics': metrics(y, h), 'cost_rank': 0.}]
    # Confidence router without expert output in its input.
    uncertainty = 1. - 2.*np.abs(probability-.5)
    confidence = tune_policy('confidence_router', y, uncertainty, h, expert, cfg)
    candidates.append({'name': 'confidence_router', 'threshold': confidence['threshold'],
                       'metrics': confidence['metrics'], 'cost_rank': confidence['cost_rank']})
    # Reference can win development; never force an ensemble to be the headline.
    candidates.append({'name': 'full_grace', 'threshold': None,
                       'metrics': metrics(y, [None if v < 0 else int(v) for v in expert]),
                       'cost_rank': 1.})
    # Avoid winning solely by dropping invalid expert predictions from metric denominators.
    selectable = [c for c in candidates if c['metrics']['invalid_n'] == 0]
    winner = choose(selectable, cfg)
    entries = {}
    for name in META_ARMS:
        features = pre if name == 'benefit_router' else post
        model = MetaModel(name, seed, cfg).fit(features, y, h, expert)
        path = folder / (name + '.joblib')
        save_model(path, model)
        entries[name] = model_entry(root, path)
    np.savez_compressed(folder / 'development_oof.npz', **oof, y=y, expert=expert, h=h)
    write_json(folder / 'folds.json', fold_entries)
    write_jsonl(folder / 'expert_validation.jsonl', expert_records)
    manifest = {'schema': 'h3v2_meta_v1', 'fingerprint': key, 'spec': spec,
                'folder': folder.relative_to(root).as_posix(), 'models': entries,
                'candidates': candidates, 'winner': winner,
                'fit_seconds': time.perf_counter()-started,
                'resources': resource_usage(),
                'expert_invalid_n': int((expert < 0).sum()),
                'fallback': 'classifier train-threshold for required expert parse-invalid; API errors must resume',
                'selection': 'MCC then F1 then cost on development only; refit on all eligible validation'}
    write_json(folder / 'manifest.json', manifest)
    files = [folder / n for n in ('manifest.json', 'development_oof.npz', 'folds.json',
                                  'expert_validation.jsonl')]
    files += [root / e['path'] for e in entries.values()]
    freeze(root, folder, files + [root / base['folder'] / 'freeze.json'], spec)
    if callback:
        callback()
    return manifest


def predict_all(root, base, meta, rows, scores, expert):
    verify(root, root / meta['folder'])
    probabilities = arm_scores(root, base, scores)
    probability = probabilities[base['winner']['name']]
    h = (probability >= base['winner']['threshold']).astype(int)
    expert = np.asarray(expert, dtype=int)
    pre = pre_features(rows, scores, probability)
    post = post_features(pre, expert, probability)
    selected = {r['name']: r for r in meta['candidates']}
    result = {}
    for name in selected:
        threshold = selected[name]['threshold']
        if name.startswith('gate_'):
            pred = (probability >= threshold).astype(int)
            required = np.zeros(len(rows), dtype=bool)
            fallback = required.copy()
        elif name == 'full_grace':
            pred, required = expert, np.ones(len(rows), dtype=bool)
            fallback = np.zeros(len(rows), dtype=bool)
        else:
            if name == 'confidence_router':
                score = 1. - 2.*np.abs(probability-.5)
            else:
                model = load_model(root, meta['models'][name])
                score = model.score(pre if name == 'benefit_router' else post)
            pred, required, fallback = output_for(name, score, threshold, h, expert)
        result[name] = {'prediction': [None if p < 0 else int(p) for p in pred],
                        'expert_required': required.tolist(), 'fallback': fallback.tolist()}
    for c in base['candidates']:
        name = 'base_' + c['name']
        p = (probabilities[c['name']] >= c['threshold']).astype(int)
        result[name] = {'prediction': p.tolist(), 'expert_required': [False]*len(rows),
                        'fallback': [False]*len(rows)}
    result['base_semantic_lr_0_5'] = {
        'prediction': (probabilities['semantic_lr'] >= .5).astype(int).tolist(),
        'expert_required': [False]*len(rows), 'fallback': [False]*len(rows)}
    return result, probability
