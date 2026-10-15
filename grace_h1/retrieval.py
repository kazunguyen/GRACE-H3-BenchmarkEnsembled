from __future__ import annotations
import heapq,time
from pathlib import Path
import numpy as np
from .io import fingerprint,read_json,iter_jsonl,write_jsonl,row_identity,resolve_root,file_hash
from .reference import features,jaccard,jaccard_tokens,sequence_similarity

POOL_METHOD='full_label_blind_tokens_calls_guards_legacy_ties_v1'

def common_candidates(root,dataset,split,targets,train,size=50):
    root=resolve_root(root);by_id={r['record_id']:r for r in train};legacy={}
    key=fingerprint({'train':row_identity(train),'targets':row_identity(targets),'size':size,'method':POOL_METHOD})
    dest=root/'cache'/'candidates'/(key+'.jsonl')
    if dest.exists(): rows=list(iter_jsonl(dest))
    else:
        wanted={r['record_id']:r['code_sha256'] for r in targets}
        for file in sorted((root/'cache'/'legacy_candidates').glob('*.jsonl')):
            for row in iter_jsonl(file):
                if wanted.get(row.get('record_id'))==row.get('code_sha256'):
                    prev=legacy.get(row['record_id'])
                    if prev and prev['initial_candidate_ids']!=row['initial_candidate_ids']: raise ValueError('Conflicting prior candidates')
                    legacy[row['record_id']]=row
        feats=None;rows=[];started=time.perf_counter()
        for i,target in enumerate(targets):
            row=legacy.get(target['record_id'])
            if row is None:
                if feats is None: feats=[features(r['code']) for r in train]
                f=features(target['code']);rank=[]
                for candidate,g in zip(train,feats):
                    if candidate['code_sha256']==target['code_sha256']: continue
                    score=sum(w*jaccard(f[k],g[k]) for w,k in ((.55,'tokens'),(.30,'calls'),(.15,'guards')))
                    tie=candidate['record_id'].rsplit('#',1)[0] if dataset=='reveal' else candidate['record_id']
                    rank.append((score,tie,candidate['record_id']))
                ids=[r[2] for r in heapq.nsmallest(size,rank,key=lambda r:(-r[0],r[1]))]
                row={'record_id':target['record_id'],'code_sha256':target['code_sha256'],'initial_candidate_ids':ids,
                     'candidate_code_sha256':[by_id[rid]['code_sha256'] for rid in ids],'origin':POOL_METHOD}
            rows.append(row)
            if (i+1)%250==0: print(f'{dataset}/{split} common candidates {i+1}/{len(targets)} ({time.perf_counter()-started:.0f}s)',flush=True)
        write_jsonl(dest,rows)
    if len(rows)!=len(targets): raise ValueError('Candidate count differs')
    for target,row in zip(targets,rows):
        ids=row['initial_candidate_ids']
        if row['record_id']!=target['record_id'] or row['code_sha256']!=target['code_sha256']: raise ValueError('Candidate query identity differs')
        if len(ids)!=size or len(set(ids))!=size or any(rid not in by_id for rid in ids): raise ValueError('Invalid candidate IDs')
        hashes=[by_id[rid]['code_sha256'] for rid in ids]
        if hashes!=row['candidate_code_sha256'] or target['code_sha256'] in hashes: raise ValueError('Candidate code mismatch/leak')
    return rows

def shortlist(targets,train,common,train_vectors,query_vectors,top_k=5):
    index={r['record_id']:i for i,r in enumerate(train)};out=[]
    if len(query_vectors)!=len(targets) or len(train_vectors)!=len(train): raise ValueError('Vector count mismatch')
    for i,(target,row) in enumerate(zip(targets,common)):
        ids=row['initial_candidate_ids'];positions=np.asarray([index[rid] for rid in ids])
        distances=((train_vectors[positions]-query_vectors[i])**2).sum(1)
        ranked=sorted(zip(ids,distances.tolist()),key=lambda r:(r[1],r[0]))[:top_k]
        out.append({'record_id':target['record_id'],'code_sha256':target['code_sha256'],
                    'initial_candidate_ids':ids,'semantic_candidates':[{'record_id':rid,'squared_l2':d} for rid,d in ranked]})
    return out

def rerank(target,semantic,train_by_id,graphs,lexical_weight=.7):
    tg=graphs.get(target['code_sha256']);rank=[]
    for candidate in semantic['semantic_candidates']:
        demo=train_by_id[candidate['record_id']];dg=graphs.get(demo['code_sha256'])
        lexical=jaccard_tokens(target['code'],demo['code'])
        a=tg.get('ast_edge_tokens',[]);b=dg.get('ast_edge_tokens',[])
        syntax=sequence_similarity(a,b,2) if a and b else 0.0
        score=lexical_weight*lexical+(1-lexical_weight)*syntax
        rank.append({'record_id':demo['record_id'],'lexical':lexical,'syntax':syntax,'mixed':score,'squared_l2':candidate['squared_l2']})
    if not rank: raise ValueError('Empty semantic shortlist')
    rank.sort(key=lambda r:(-r['mixed'],r['record_id']))
    return rank[0],rank
