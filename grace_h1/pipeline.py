from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass,asdict
from concurrent.futures import ThreadPoolExecutor
import gc,subprocess,time,sys,importlib.metadata
import numpy as np
import yaml
from .io import resolve_root,read_json,write_json,write_jsonl,iter_jsonl,file_hash,fingerprint,row_identity,sha256
from .data import load_rows,load_embeddings,embedding_recipe
from .projection import project_embeddings,whitening
from .retrieval import common_candidates,shortlist,rerank
from .graphs import GraphStore
from .reference import raw_graph_context,fit_contracted_prompt
from .parser import parse_prediction
from .metrics import binary_metrics,paired_comparison

INSTRUCTION=('You are now an excellent programmer. You are conducting a function vulnerability detection task for C/C++ language. '
             'Determine whether the target function contains a software vulnerability. Use the target code, graph context, and labeled reference example as evidence.')

@dataclass(frozen=True)
class PipelineConfig:
    dataset: str='devign'
    split: str='test'
    arm: str='h1_supcon_hard'
    semantic_top_k: int=5
    lexical_weight: float=.7
    whitening_dimension: int=256
    graph_workers: int=2
    joern_home: str|None=None

    def validate(self):
        if self.dataset not in ('devign','reveal') or self.split not in ('validation','test'): raise ValueError('Unsupported dataset/split')
        if self.arm not in ('baseline_whitening','h1_ce','h1_supcon','h1_supcon_hard'): raise ValueError('Unknown arm')
        if not 1<=self.semantic_top_k<=50 or not 0<=self.lexical_weight<=1 or self.graph_workers<1: raise ValueError('Invalid retrieval config')

def verify_frozen(root,run_dir):
    root=resolve_root(root);run_dir=Path(run_dir);frozen=read_json(run_dir/'freeze.json')
    for rel,expected in frozen['files'].items():
        if file_hash(root/rel)!=expected: raise ValueError(f'Frozen input changed: {rel}')
    return read_json(run_dir/'run_manifest.json')

def prepare_run(root,config=None,head_manifest=None,checkpoint_callback=None,target_ids=None):
    root=resolve_root(root);cfg=config or PipelineConfig();cfg.validate();started=time.perf_counter()
    train=load_rows(root,cfg.dataset,'train',labels=True);targets=load_rows(root,cfg.dataset,cfg.split)
    if target_ids is not None:
        wanted=set(target_ids)
        if len(wanted)!=len(target_ids) or not wanted.issubset({r['record_id'] for r in targets}): raise ValueError('Invalid inspect IDs')
        targets=[r for r in targets if r['record_id'] in wanted]
        if not targets: raise ValueError('Empty inspect cohort must bypass prepare_run')
    tx,tm=load_embeddings(root,cfg.dataset,'train',train);qx,qm=load_embeddings(root,cfg.dataset,cfg.split,targets)
    recipe=embedding_recipe(tm)
    if embedding_recipe(qm)!=recipe: raise ValueError('Query/train encoder recipe mismatch')
    if cfg.arm!='baseline_whitening':
        if not head_manifest: raise ValueError('H1 needs a trained projection manifest')
        if head_manifest['dataset']!=cfg.dataset or head_manifest['train_identity']!=row_identity(train): raise ValueError('Wrong projection dataset')
        if cfg.arm!='h1_'+head_manifest['objective'] or head_manifest['encoder_recipe']!=recipe: raise ValueError('Wrong H1 objective/encoder recipe')
    sources={name:file_hash(Path(__file__).with_name(name)) for name in
             ('pipeline.py','reference.py','graphs.py','retrieval.py','projection.py','data.py')}
    contract_path=root/'configs'/'reference_contract.yaml';template_path=root/'configs'/'prompt_template.txt'
    contract=yaml.safe_load(contract_path.read_text(encoding='utf-8'));template=template_path.read_text(encoding='utf-8')
    if contract['retrieval_pool']['candidate_pool_size']!=50: raise ValueError('Expected common top50 contract')
    spec={'schema':'grace_h1_prepare_v1','config':asdict(cfg),'train':row_identity(train),'targets':row_identity(targets),
          'embedding_files':[tm['files'],qm['files']],'recipe':recipe,'head':head_manifest['projection_sha256'] if head_manifest else None,
          'data_manifest':file_hash(root/'data'/'manifest.json'),'contract':file_hash(contract_path),'template':file_hash(template_path),'source':sources}
    signature=fingerprint(spec);output=root/'runs'/cfg.dataset/cfg.split/cfg.arm/signature[:16]
    if (output/'freeze.json').exists():
        manifest=verify_frozen(root,output)
        if manifest['prepare_fingerprint']!=signature: raise ValueError('Run cache key mismatch')
        print(f'reuse frozen preparation: {output}',flush=True);return output
    output.mkdir(parents=True,exist_ok=True)
    common=common_candidates(root,cfg.dataset,cfg.split,targets,train)
    vkey=fingerprint({'train':tm['files'],'query':qm['files'],'arm':cfg.arm,'head':spec['head'],
                     'whitening_dimension':cfg.whitening_dimension,'projection_source':sources['projection.py']})
    vdir=root/'cache'/'projected'/vkey;vdir.mkdir(parents=True,exist_ok=True)
    vmeta=vdir/'manifest.json'
    if vmeta.exists():
        geom=read_json(vmeta)
        for name,h in geom['files'].items():
            if file_hash(vdir/name)!=h: raise ValueError('Projected vector cache checksum differs')
    else:
        if cfg.arm=='baseline_whitening': a,b,detail=whitening(tx,qx,cfg.whitening_dimension)
        else: a,b=project_embeddings(tx,head_manifest,root),project_embeddings(qx,head_manifest,root);detail={'fit':'train_only','dimension':a.shape[1]}
        for name,array in [('train.npy',a),('query.npy',b)]:
            temp=vdir/(name+'.tmp.npy');np.save(temp,array,allow_pickle=False);temp.replace(vdir/name)
        write_json(vmeta,{'spec':spec,'detail':detail,'files':{name:file_hash(vdir/name) for name in ('train.npy','query.npy')}})
        del a,b;gc.collect()
    tv=np.load(vdir/'train.npy',mmap_mode='r',allow_pickle=False);qv=np.load(vdir/'query.npy',mmap_mode='r',allow_pickle=False)
    sem=shortlist(targets,train,common,tv,qv,cfg.semantic_top_k);del tv,qv,tx,qx;gc.collect()
    by_id={r['record_id']:r for r in train};needed={r['code_sha256']:r for r in targets}
    for item in sem:
        for candidate in item['semantic_candidates']:
            row=by_id[candidate['record_id']];needed[row['code_sha256']]=row
    graphs=GraphStore(root,cfg.joern_home);graph_index=[]
    print(f'{cfg.dataset}/{cfg.arm}: {len(needed)} unique graph features required',flush=True)
    def ensure_summary(row):
        graph=graphs.ensure(row)
        return {'code_sha256':graph['code_sha256'],'feature_sha256':graph['feature_sha256'],
                'schema':graph['schema'],'nodes':len(graph['nodes']),'edges':len(graph['edges'])}
    last_checkpoint_built=-1
    with ThreadPoolExecutor(max_workers=cfg.graph_workers) as pool:
        for n,summary in enumerate(pool.map(ensure_summary,needed.values()),1):
            graph_index.append(summary)
            if n%500==0: print(f'graphs {n}/{len(needed)}',flush=True)
            if checkpoint_callback and n%500==0 and graphs.built!=last_checkpoint_built:
                checkpoint_callback();last_checkpoint_built=graphs.built
    trunc=contract['prompt_context_policy']['truncation_policy']
    policy=trunc|{'context_limit':contract['prompt_context_policy']['context_limit']}
    prompts=[];retrieval=[];available=0
    for target,item in zip(targets,sem):
        chosen,ranking=rerank(target,item,by_id,graphs,cfg.lexical_weight);demo=by_id[chosen['record_id']]
        graph=graphs.get(target['code_sha256']);available+=bool(graph['nodes'] and graph['edges'])
        graph_text,graph_meta=raw_graph_context(graph,trunc)
        selected={'record_id':demo['record_id'],'code_sha256':demo['code_sha256'],'code':demo['code'],'label':int(demo['label'])}
        prompt,meta=fit_contracted_prompt(template,target['code'],graph_text,[selected],INSTRUCTION,policy)
        prompts.append({'record_id':target['record_id'],'code_sha256':target['code_sha256'],'dataset':cfg.dataset,'split':cfg.split,
            'prompt':prompt,'prompt_sha256':sha256(prompt),'selected_demo_ids':[demo['record_id']],
            'graph_schema':graph['schema'],'graph_feature_sha256':graph['feature_sha256'],'graph_counts':graph_meta,'prompt_truncation':meta})
        retrieval.append(item|{'rerank':ranking,'selected_demo_id':demo['record_id'],'selected_demo_label':int(demo['label'])})
    paths=[output/'retrieval.jsonl',output/'prompts.jsonl',output/'graph_feature_index.jsonl']
    for path,rows in zip(paths,[retrieval,prompts,graph_index]): write_jsonl(path,rows)
    git=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
    versions={}
    for package in ('numpy','torch','rapidfuzz','PyYAML','google-genai'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    manifest={'schema':'grace_h1_run_v1','runtime':{'python':sys.version,'packages':versions},'prepare_fingerprint':signature,'inputs':spec,'target_count':len(targets),
              'target_identity':row_identity(targets),'train_count':len(train),'config':asdict(cfg),
              'head_manifest':head_manifest,'commit':git.stdout.strip() if git.returncode==0 else None,
              'graph_coverage':available/len(targets),'graph_reused':graphs.reused,'graph_built':graphs.built,
              'prepare_seconds':time.perf_counter()-started,'common_pool_sha256':fingerprint([(r['record_id'],r['initial_candidate_ids']) for r in common]),
              'graph_feature_identity':fingerprint(sorted((r['code_sha256'],r['feature_sha256']) for r in graph_index)),
              'query_labels_used_for_retrieval':False,'demo_count':1,'syntax_profile':'code_based_ast_edge_tokens_substitution_2',
              'prompt_budget_unit':'conservative UTF-8 bytes from baseline fitter; provider usage recorded at inference'}
    write_json(output/'run_manifest.json',manifest)
    locked=paths+[output/'run_manifest.json',contract_path,template_path,root/'data'/'manifest.json']
    write_json(output/'freeze.json',{'files':{p.relative_to(root).as_posix():file_hash(p) for p in locked}})
    print(f'prepared {len(prompts)} prompts; graph coverage {available/len(targets):.1%}: {output}',flush=True)
    if checkpoint_callback: checkpoint_callback()
    return output

def evaluate_run(root,run_dir):
    root=resolve_root(root);run_dir=Path(run_dir);manifest=verify_frozen(root,run_dir)
    cfg=manifest['config'];rows=load_rows(root,cfg['dataset'],cfg['split'],labels=True)
    pred=list(iter_jsonl(run_dir/'predictions.jsonl'));lookup={r['record_id']:r for r in pred}
    if len(lookup)!=len(pred) or set(lookup)!={r['record_id'] for r in rows}: raise ValueError('Predictions do not cover the full cohort')
    labels=[];values=[]
    for row in rows:
        p=lookup[row['record_id']]
        if p['code_sha256']!=row['code_sha256']: raise ValueError('Prediction code changed')
        labels.append(row['label']);values.append(p['prediction'])
    metrics=binary_metrics(labels,values)
    metrics['trivial_baselines']={'always_vulnerable':binary_metrics(labels,[1]*len(labels)),
                                 'always_safe':binary_metrics(labels,[0]*len(labels))}
    metrics['prediction_sha256']=file_hash(run_dir/'predictions.jsonl');write_json(run_dir/'evaluation.json',metrics)
    return metrics

def compare_runs(root,left_run,right_run,bootstrap=1000):
    root=resolve_root(root);a=verify_frozen(root,left_run);b=verify_frozen(root,right_run)
    for key in ('target_identity','common_pool_sha256'):
        if a[key]!=b[key]: raise ValueError(f'Not a paired comparison: {key}')
    for key in ('dataset','split','semantic_top_k','lexical_weight'):
        if a['config'][key]!=b['config'][key]: raise ValueError('Controls differ')
    if a['inputs']['contract']!=b['inputs']['contract'] or a['inputs']['recipe']!=b['inputs']['recipe']: raise ValueError('LLM/encoder controls differ')
    lp={r['record_id']:r for r in iter_jsonl(Path(left_run)/'prompts.jsonl')}
    rp={r['record_id']:r for r in iter_jsonl(Path(right_run)/'prompts.jsonl')}
    if set(lp)!=set(rp) or any(lp[rid]['graph_feature_sha256']!=rp[rid]['graph_feature_sha256'] for rid in lp):
        raise ValueError('Query graph evidence differs between arms')
    left={r['record_id']:r for r in iter_jsonl(Path(left_run)/'predictions.jsonl')}
    right={r['record_id']:r for r in iter_jsonl(Path(right_run)/'predictions.jsonl')}
    rows=load_rows(root,a['config']['dataset'],a['config']['split'],labels=True)
    if set(left)!=set(right) or set(left)!={r['record_id'] for r in rows}: raise ValueError('Paired cohort differs')
    comparison=paired_comparison([r['label'] for r in rows],[left[r['record_id']]['prediction'] for r in rows],
                                [right[r['record_id']]['prediction'] for r in rows],bootstrap=bootstrap)
    comparison.update(left=str(Path(left_run).relative_to(root)),right=str(Path(right_run).relative_to(root)))
    destination=root/'runs'/'comparisons'/(fingerprint(comparison)[:16]+'.json');write_json(destination,comparison)
    return comparison
