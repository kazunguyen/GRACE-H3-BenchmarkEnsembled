from __future__ import annotations
from pathlib import Path
import numpy as np
from .io import read_json, iter_jsonl, file_hash, sha256, row_identity, resolve_root

SPLITS=('train','validation','test')

def load_rows(root, dataset, split, *, labels=False):
    root=resolve_root(root)
    if dataset not in ('devign','reveal') or split not in SPLITS: raise ValueError('Unsupported dataset/split')
    manifest=read_json(root/'data'/'manifest.json')
    item=manifest['datasets'][dataset][split]
    path=root/item['rows_path']
    if file_hash(path)!=item['rows_sha256']: raise ValueError(f'Dataset bytes changed: {path}')
    rows=list(iter_jsonl(path))
    if len(rows)!=item['count'] or row_identity(rows)!=item['identity']: raise ValueError('Dataset identity mismatch')
    if len({r['record_id'] for r in rows})!=len(rows): raise ValueError('Duplicate record IDs')
    for row in rows:
        if sha256(row['code'])!=row['code_sha256']: raise ValueError('Code hash mismatch')
        if not row['record_id'].startswith(f'{dataset}:{split}:'): raise ValueError('Wrong split in ID')
        if labels:
            if row.get('label') not in (0,1): raise ValueError('Missing binary label')
        else: row.pop('label',None)
    return rows

def load_embeddings(root,dataset,split,rows=None):
    root=resolve_root(root);rows=rows if rows is not None else load_rows(root,dataset,split)
    path=root/'cache'/'embeddings'/dataset/split
    meta=read_json(path/'manifest.json');index=list(iter_jsonl(path/'index.jsonl'))
    if meta['rows_identity']!=row_identity(rows) or row_identity(index)!=row_identity(rows):
        raise ValueError('Embedding row order/code hashes do not match dataset')
    for name,expected in meta['files'].items():
        if file_hash(path/name)!=expected: raise ValueError(f'Embedding checksum mismatch: {name}')
    matrix=np.load(path/'vectors.npy',mmap_mode='r',allow_pickle=False)
    if list(matrix.shape)!=meta['shape'] or matrix.dtype!=np.float32: raise ValueError('Invalid embedding shape/dtype')
    if len(matrix)!=len(rows): raise ValueError('Embedding count mismatch')
    for start in range(0,len(matrix),2048):
        if not np.isfinite(matrix[start:start+2048]).all(): raise ValueError('Non-finite cached embeddings')
    return matrix,meta

def embedding_recipe(meta):
    return {k:meta['encoder_spec'][k] for k in ('architecture','checkpoint','pooling','max_length','seed')}

def load_training_data(root,dataset):
    train=load_rows(root,dataset,'train',labels=True)
    valid=load_rows(root,dataset,'validation',labels=True)
    x,tm=load_embeddings(root,dataset,'train',train);v,vm=load_embeddings(root,dataset,'validation',valid)
    if embedding_recipe(tm)!=embedding_recipe(vm): raise ValueError('Train/validation encoder recipes differ')
    # Exclude exact train code duplicates from model-selection validation only.
    # Full GRACE validation/test cohorts remain intact for inference/evaluation.
    hashes={r['code_sha256'] for r in train}
    unique=[i for i,r in enumerate(valid) if r['code_sha256'] not in hashes]
    if not unique: raise ValueError('No independent validation code remains')
    return train,valid,x,v,tm,vm,unique
