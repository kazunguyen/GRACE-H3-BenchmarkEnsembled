from pathlib import Path
import os
import time
import warnings
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from grace_h1.io import read_json, write_json, file_hash, fingerprint, row_identity
from grace_h1.data import load_rows, load_embeddings, embedding_recipe
from .common import (training_data, folds, sources, metrics, best_threshold,
                     choose, freeze, verify, versions, inside, resource_usage)
from .features import View, VIEWS, ARMS, fusion_input, fusion_model


def save_model(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    joblib.dump(obj, temp, compress=3)
    os.replace(temp, path)


def load_model(root, entry):
    path = (root / entry['path']).resolve()
    inside(root, path)
    if file_hash(path) != entry['sha256']:
        raise ValueError('Model checksum mismatch')
    if entry['versions'] != versions():
        raise ValueError('Model runtime versions differ; install requirements-v2.txt versions')
    # Only load self-created, checksummed experiment artifacts; joblib is executable pickle.
    return joblib.load(path)


def model_entry(root, path):
    return {'path': path.relative_to(root).as_posix(), 'sha256': file_hash(path),
            'versions': versions()}


def cached_view(root, dataset, rows, x, y, name, cfg, seed, callback):
    key = fingerprint({'source': sources(root), 'dataset': dataset,
                       'embedding_manifest': file_hash(root / 'cache/embeddings' / dataset / 'train/manifest.json'),
                       'data_manifest': file_hash(root / 'data/manifest.json'),
                       'rows': row_identity(rows), 'labels': y.tolist(),
                       'view': name, 'config': cfg, 'seed': seed, 'versions': versions()})
    folder = root / 'cache/h3v2/fits' / key[:24]
    manifest = folder / 'manifest.json'
    if manifest.exists():
        entry = read_json(manifest)
        if entry['fingerprint'] != key:
            raise ValueError('Fit cache fingerprint mismatch')
        return load_model(root, entry)
    started = time.perf_counter()
    print(f'Fit {dataset}/{name}: {len(rows)} train rows', flush=True)
    obj = View(name, cfg, seed).fit(rows, x, y)
    path = folder / 'model.joblib'
    save_model(path, obj)
    write_json(manifest, model_entry(root, path) |
               {'fingerprint': key, 'fit_seconds': time.perf_counter()-started,
                'model_type': name, 'rows': row_identity(rows)})
    if callback:
        callback()
    return obj


def view_scores(root, dataset, rows, x, y, targets, q, cfg, seed, callback):
    output = np.empty((len(targets), len(VIEWS)), dtype=np.float64)
    entries = {}
    for j, name in enumerate(VIEWS):
        model = cached_view(root, dataset, rows, x, y, name, cfg, seed, callback)
        output[:, j] = model.score(targets, q)
        # Cached view key includes rows/labels, source, versions and feature config.
        key = fingerprint({'source': sources(root), 'dataset': dataset,
                           'embedding_manifest': file_hash(root / 'cache/embeddings' / dataset / 'train/manifest.json'),
                           'data_manifest': file_hash(root / 'data/manifest.json'),
                           'rows': row_identity(rows), 'labels': y.tolist(),
                           'view': name, 'config': cfg, 'seed': seed, 'versions': versions()})
        entries[name] = read_json(root / 'cache/h3v2/fits' / key[:24] / 'manifest.json')
        del model
    return output, entries


def oof_views(root, dataset, rows, x, y, cfg, seed, count, callback):
    scores = np.empty((len(rows), len(VIEWS)), dtype=np.float64)
    lineage = []
    for f, (a, b) in enumerate(folds(rows, y, count, seed, cfg['group_key'])):
        train, query = [rows[i] for i in a], [rows[i] for i in b]
        p, _ = view_scores(root, dataset, train, x[a], y[a], query, x[b],
                           cfg, seed + f * 1009, callback)
        scores[b] = p
        lineage.append({'fold': f, 'fit_indices': a.tolist(), 'held_indices': b.tolist()})
    return scores, lineage


def train_base(root, dataset, seed, cfg, callback=None):
    train, valid, x, v, ti, vi, tm, vm = training_data(root, dataset)
    rows = [train[i] for i in ti]
    y = np.asarray([r['label'] for r in rows], dtype=int)
    a = np.asarray(x[ti], dtype=np.float32)
    spec = {'source': sources(root), 'dataset': dataset, 'seed': seed, 'config': cfg,
            'versions': versions(), 'data_manifest': file_hash(root / 'data/manifest.json'),
            'train_identity': row_identity(train), 'validation_identity': row_identity(valid),
            'embedding_files': [tm['files'], vm['files']], 'recipe': embedding_recipe(tm),
            'fit_indices': ti.tolist(), 'validation_indices': vi.tolist()}
    key = fingerprint(spec)
    folder = root / 'artifacts/h3v2/base' / dataset / str(seed) / key[:20]
    if (folder / 'freeze.json').exists():
        verify(root, folder)
        return read_json(folder / 'manifest.json')
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    outer = np.empty((len(rows), len(ARMS)), dtype=np.float64)
    base_oof = np.empty((len(rows), len(VIEWS)), dtype=np.float64)
    lineage = []
    for f, (fit, held) in enumerate(folds(rows, y, cfg['folds'], seed, cfg['group_key'])):
        tr, qr = [rows[i] for i in fit], [rows[i] for i in held]
        prediction, _ = view_scores(root, dataset, tr, a[fit], y[fit], qr, a[held],
                                     cfg, seed + f * 1009, callback)
        base_oof[held] = prediction
        inner, inner_folds = oof_views(root, dataset, tr, a[fit], y[fit], cfg,
                                       seed + 30001 + f * 1009, cfg['inner_folds'], callback)
        for j, arm in enumerate(ARMS):
            if arm in VIEWS:
                outer[held, j] = prediction[:, VIEWS.index(arm)]
            else:
                model = fusion_model()
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    model.fit(fusion_input(inner, arm), y[fit])
                outer[held, j] = model.predict_proba(fusion_input(prediction, arm))[:, 1]
        lineage.append({'fold': f, 'fit_indices': fit.tolist(),
                        'held_indices': held.tolist(), 'inner_folds_local_indices': inner_folds})
    candidates = []
    for j, arm in enumerate(ARMS):
        threshold = best_threshold(y, outer[:, j], cfg, arm)
        candidates.append({'name': arm, 'threshold': threshold['threshold'],
                           'metrics': threshold['metrics'], 'cost_rank': j,
                           'metrics_at_0_5': metrics(y, (outer[:, j] >= .5).astype(int))})
    winner = choose(candidates, cfg)
    # Refit on official-train eligible rows; no validation labels select the classifier.
    val_score, entries = view_scores(root, dataset, rows, a, y, valid, np.asarray(v),
                                     cfg, seed, callback)
    models = {'views': {}}
    files = []
    for name, entry in entries.items():
        obj = load_model(root, entry)
        path = folder / (name + '.joblib')
        save_model(path, obj)
        models['views'][name] = model_entry(root, path)
        files.append(path)
    models['fusion'] = {}
    for arm in ARMS:
        if arm not in VIEWS:
            model = fusion_model().fit(fusion_input(base_oof, arm), y)
            path = folder / (arm + '.joblib')
            save_model(path, model)
            models['fusion'][arm] = model_entry(root, path)
            files.append(path)
    np.save(folder / 'validation_views.npy', val_score, allow_pickle=False)
    np.save(folder / 'train_nested_oof.npy', outer, allow_pickle=False)
    write_json(folder / 'folds.json', lineage)
    manifest = {'schema': 'h3v2_base_v1', 'fingerprint': key, 'spec': spec,
                'folder': folder.relative_to(root).as_posix(), 'models': models,
                'winner': winner, 'candidates': candidates, 'view_columns': list(VIEWS),
                'arm_columns': list(ARMS), 'fit_seconds': time.perf_counter()-started,
                'resources': resource_usage(),
                'selection_scope': 'train nested OOF; thresholds tuned on aggregated OOF for development only'}
    write_json(folder / 'manifest.json', manifest)
    files += [folder / n for n in ('manifest.json', 'validation_views.npy',
                                    'train_nested_oof.npy', 'folds.json')]
    freeze(root, folder, files + [root / 'data/manifest.json'], spec)
    if callback:
        callback()
    return manifest


def arm_scores(root, manifest, view_scores):
    output = {}
    for arm in ARMS:
        if arm in VIEWS:
            output[arm] = view_scores[:, VIEWS.index(arm)]
        else:
            model = load_model(root, manifest['models']['fusion'][arm])
            output[arm] = model.predict_proba(fusion_input(view_scores, arm))[:, 1]
    return output


def score_queries(root, manifest, dataset, split):
    verify(root, root / manifest['folder'])
    rows = load_rows(root, dataset, split)  # Test labels never enter feature generation.
    x, meta = load_embeddings(root, dataset, split, rows)
    if embedding_recipe(meta) != manifest['spec']['recipe']:
        raise ValueError('Query encoder recipe differs')
    scores = np.column_stack([load_model(root, manifest['models']['views'][name])
                              .score(rows, x) for name in VIEWS])
    return rows, scores
