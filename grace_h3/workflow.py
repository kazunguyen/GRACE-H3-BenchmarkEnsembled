"""Route before GRACE preparation and evaluate the complete original cohort."""
from pathlib import Path
import argparse,json,time
from collections import Counter
from grace_h1.io import resolve_root,read_json,write_json,write_jsonl,iter_jsonl,file_hash,fingerprint,row_identity
from grace_h1.pipeline import prepare_run,PipelineConfig,verify_frozen
from grace_h1.inference import plan_inference,run_inference
from grace_h1.data import load_rows
from grace_h1.metrics import binary_metrics,paired_comparison
from .gate import fit_gate,gate_scores,source_identity

def head_for(root,dataset,arm):
    if arm=='baseline_whitening': return None
    if arm not in ('h1_ce','h1_supcon','h1_supcon_hard'): raise ValueError('Unsupported downstream arm')
    head=read_json(root/'artifacts'/'training_selection.json')['heads'][dataset][arm.removeprefix('h1_')]
    for name,h in head['source'].items():
        if file_hash(root/'grace_h1'/name)!=h: raise ValueError('Imported H1 training source differs')
    path=root/head['path'];path.resolve().relative_to(root/'artifacts'/'heads')
    if file_hash(path)!=head['projection_sha256']: raise ValueError('Imported H1 weights differ')
    return head

def verify_job(root,folder):
    root=resolve_root(root);folder=Path(folder);frozen=read_json(folder/'freeze_h3.json')
    for name,h in frozen['files'].items():
        if file_hash(root/name)!=h: raise ValueError(f'H3 frozen artifact changed: {name}')
    job=read_json(folder/'job.json')
    if job['source']!=source_identity(root): raise ValueError('H3 code changed; recreate CPU inputs/preparation')
    if job['data_manifest']!=file_hash(root/'data'/'manifest.json'): raise ValueError('H3 dataset changed')
    head=head_for(root,job['dataset'],job['config']['retrieval_arm'])
    if job['downstream_head_sha256']!=(head['projection_sha256'] if head else None): raise ValueError('Downstream H1 head changed')
    return job

def prepare_all(root,config,checkpoint_callback=None):
    root=resolve_root(root);results=[]
    if config['split'] not in ('test','validation') or len(set(config['datasets']))!=len(config['datasets']): raise ValueError('Invalid experiment cohort')
    for dataset in config['datasets']:
        gate=fit_gate(root,dataset,config['gate']);start=time.perf_counter()
        rows,scores=gate_scores(root,dataset,config['split'],gate);policy=gate['policy'];routing=[]
        for row,p in zip(rows,scores):
            branch='skip_low' if p<policy['low'] else ('direct_high' if policy['high'] is not None and p>=policy['high'] else 'inspect')
            routing.append({'record_id':row['record_id'],'code_sha256':row['code_sha256'],'score':p,'branch':branch,
                            'gate_prediction':int(p>=.5),'direct_prediction':0 if branch=='skip_low' else (1 if branch=='direct_high' else None)})
        route_seconds=time.perf_counter()-start
        head=head_for(root,dataset,config['retrieval_arm'])
        identity={'config':config,'dataset':dataset,'gate_fingerprint':gate['fingerprint'],'source':source_identity(root),
                  'data_manifest':file_hash(root/'data'/'manifest.json'),'target_identity':row_identity(rows),
                  'contract':file_hash(root/'configs'/'reference_contract.yaml'),'template':file_hash(root/'configs'/'prompt_template.txt'),
                  'downstream_head_sha256':head['projection_sha256'] if head else None}
        key=fingerprint(identity);folder=root/'runs'/'h3'/dataset/key[:16];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'freeze_h3.json').exists():
            verify_job(root,folder);results.append({'dataset':dataset,'path':folder.relative_to(root).as_posix()});continue
        write_jsonl(folder/'routing.jsonl',routing);write_json(folder/'gate_manifest.json',gate)
        pcfg=PipelineConfig(dataset=dataset,split=config['split'],arm=config['retrieval_arm'],graph_workers=config['graph_workers'])
        inspect_ids=[r['record_id'] for r in routing if r['branch']=='inspect']
        inspect_path=prepare_run(root,pcfg,head,checkpoint_callback,target_ids=inspect_ids) if inspect_ids else None
        reference=prepare_run(root,pcfg,head,checkpoint_callback) if config['full_reference'] else None
        if inspect_path and reference:
            a={r['record_id']:r for r in iter_jsonl(inspect_path/'prompts.jsonl')}
            b={r['record_id']:r for r in iter_jsonl(reference/'prompts.jsonl')}
            if any(a[k]['prompt_sha256']!=b[k]['prompt_sha256'] for k in a): raise ValueError('Inspect and full-reference prompts differ; paired routing effect is not isolated')
        job=identity|{'fingerprint':key,'inspect_path':inspect_path.relative_to(root).as_posix() if inspect_path else None,
                      'reference_path':reference.relative_to(root).as_posix() if reference else None,
                      'branch_counts':dict(Counter(r['branch'] for r in routing)),'n':len(rows),'routing_labels_used':False,
                      'gate_route_seconds':route_seconds,
                      'reference_is_benchmark_overhead':config['full_reference']}
        write_json(folder/'job.json',job)
        files=[folder/'routing.jsonl',folder/'gate_manifest.json',folder/'job.json',root/gate['weights'],root/'data'/'manifest.json',
               root/'configs'/'reference_contract.yaml',root/'configs'/'prompt_template.txt']
        if head: files.append(root/head['path'])
        for run in {p for p in (inspect_path,reference) if p}:
            files.append(run/'freeze.json');files.extend(root/p for p in read_json(run/'freeze.json')['files'])
        write_json(folder/'freeze_h3.json',{'files':{p.relative_to(root).as_posix():file_hash(p) for p in set(files)}})
        results.append({'dataset':dataset,'path':folder.relative_to(root).as_posix()})
        if checkpoint_callback: checkpoint_callback()
    write_json(root/'runs'/'h3_selection.json',{'config':config,'runs':results});return results

def selected_jobs(root,config):
    root=resolve_root(root);selection=read_json(root/'runs'/'h3_selection.json')
    if selection['config']!=config: raise ValueError('Experiment config differs; prepare matching CPU job')
    if len(selection['runs'])!=len(config['datasets']) or {r['dataset'] for r in selection['runs']}!=set(config['datasets']): raise ValueError('Missing dataset job')
    out=[]
    for row in selection['runs']:
        folder=root/row['path'];folder.resolve().relative_to(root/'runs'/'h3');job=verify_job(root,folder)
        if job['config']!=config: raise ValueError('Frozen job config differs')
        out.append((folder,job))
    return out

def preflight_all(root,config):
    root=resolve_root(root);out=[]
    for folder,job in selected_jobs(root,config):
        plan=plan_inference(root,root/job['inspect_path']) if job['inspect_path'] else {'queries':0,'new_unique_requests':0,'worst_case_attempts':0}
        out.append({'dataset':job['dataset'],'n':job['n'],'branches':job['branch_counts'],
                    'algorithmic_llm_query_ratio':job['branch_counts'].get('inspect',0)/job['n'],
                    'cascade':plan,'reference':plan_inference(root,root/job['reference_path']) if job['reference_path'] else None,
                    'note':'Reference is optional benchmark overhead; exact inspect prompts can reuse responses across arms. Preflight reference count is an upper bound before cascade runs.'})
    return out

def merge_predictions(root,folder,job):
    routes=list(iter_jsonl(folder/'routing.jsonl'));lookup={};path=root/job['inspect_path'] if job['inspect_path'] else None
    if path and (path/'predictions.jsonl').exists():
        raw=list(iter_jsonl(path/'predictions.jsonl'));lookup={r['record_id']:r for r in raw}
        wanted={r['record_id'] for r in routes if r['branch']=='inspect'}
        if len(lookup)!=len(raw) or set(lookup)!=wanted: raise ValueError('Inspect predictions have wrong IDs')
    merged=[]
    for r in routes:
        p=lookup.get(r['record_id'])
        if p and p['code_sha256']!=r['code_sha256']: raise ValueError('Inspect prediction code differs')
        value=p['prediction'] if p else r['direct_prediction']
        merged.append(r|{'prediction':value,'request_status':p['request_status'] if p else ('not_run' if r['branch']=='inspect' else 'gate_only'),
                         'prompt_sha256':p.get('prompt_sha256') if p else None})
    write_jsonl(folder/'predictions.jsonl',merged);return merged

def infer_all(root,config,allow_api=False):
    root=resolve_root(root);reports=[]
    for folder,job in selected_jobs(root,config):
        # Cascade first: its observed new requests are not hidden by reference prewarming.
        cascade=run_inference(root,root/job['inspect_path'],allow_api=allow_api,workers=config['inference_workers']) if job['inspect_path'] else {
            'status':'complete','queries':0,'new_unique_requests':0,'api_attempts_this_run':0,'provider_usage_new_requests':{},'wall_seconds':0}
        merge_predictions(root,folder,job)
        reference=run_inference(root,root/job['reference_path'],allow_api=allow_api,workers=config['inference_workers']) if job['reference_path'] else None
        # A full reference can fill exact inspect responses during a cache-only first pass.
        if cascade['status']=='prepared_cache_only' and job['inspect_path'] and reference and reference['status']=='complete':
            cascade=run_inference(root,root/job['inspect_path'],allow_api=False,workers=config['inference_workers']);merge_predictions(root,folder,job)
        report={'dataset':job['dataset'],'cascade':cascade,'reference_incremental':reference,
                'algorithmic_llm_query_ratio':job['branch_counts'].get('inspect',0)/job['n'],
                'branch_counts':job['branch_counts'],'n':job['n']}
        write_json(folder/'inference_summary.json',report);reports.append(report)
    return reports

def evaluate_all(root,config):
    root=resolve_root(root);reports=[]
    for folder,job in selected_jobs(root,config):
        rows=load_rows(root,job['dataset'],config['split'],labels=True);p=list(iter_jsonl(folder/'predictions.jsonl'))
        if row_identity(rows)!=row_identity(p) or len(rows)!=len(p): raise ValueError('Final predictions do not cover original cohort in order')
        y=[r['label'] for r in rows];values=[r['prediction'] for r in p]
        metrics=binary_metrics(y,values);metrics['specificity']=metrics['confusion_matrix_valid_only']['tn']/max(1,metrics['confusion_matrix_valid_only']['tn']+metrics['confusion_matrix_valid_only']['fp'])
        branches={}
        for branch in ('skip_low','direct_high','inspect'):
            ids=[i for i,r in enumerate(p) if r['branch']==branch]
            branches[branch]={'n':len(ids),'coverage':len(ids)/len(y),'metrics':binary_metrics([y[i] for i in ids],[values[i] for i in ids]) if ids else None,
                              'skip_fn':sum(y[i]==1 for i in ids) if branch=='skip_low' else None,
                              'high_fp':sum(y[i]==0 for i in ids) if branch=='direct_high' else None,
                              'skip_lost_positive_fraction':sum(y[i]==1 for i in ids)/max(1,sum(y)) if branch=='skip_low' else None}
        report={'dataset':job['dataset'],'cascade':metrics,'branches':branches,
                'gate_only':binary_metrics(y,[r['gate_prediction'] for r in p]),
                'algorithmic_call_saving':1-job['branch_counts'].get('inspect',0)/job['n'],
                'inference':read_json(folder/'inference_summary.json'),
                'status':'complete' if all(r['request_status'] not in ('not_run','exhausted') for r in p) else 'partial',
                'bound_scope':read_json(folder/'gate_manifest.json')['policy']['scope']}
        report['trivial_baselines']={'always_safe':binary_metrics(y,[0]*len(y)),'always_vulnerable':binary_metrics(y,[1]*len(y))}
        if len(set(y))==2:
            from sklearn.metrics import roc_auc_score,average_precision_score
            report['gate_score_auc']={'roc_auc':float(roc_auc_score(y,[r['score'] for r in p])),
                                      'average_precision':float(average_precision_score(y,[r['score'] for r in p])),
                                      'scope':'calibrated gate score only, not an AUC of the final discrete cascade'}
        gate=read_json(folder/'gate_manifest.json')
        report['offline_artifact_timings']={'gate_fit_seconds':gate['fit_seconds'],'gate_route_seconds':job['gate_route_seconds'],
            'gate_process_peak_rss_bytes':gate['process_peak_rss_bytes'],
            'inspect_prepare_seconds':verify_frozen(root,root/job['inspect_path'])['prepare_seconds'] if job['inspect_path'] else 0,
            'reference_prepare_seconds':verify_frozen(root,root/job['reference_path'])['prepare_seconds'] if job['reference_path'] else None,
            'note':'Stored artifact creation timings, not fresh-run latency after cache reuse; process peak is lifetime peak.'}
        if job['reference_path'] and (root/job['reference_path']/'predictions.jsonl').exists():
            raw=list(iter_jsonl(root/job['reference_path']/'predictions.jsonl'));lookup={r['record_id']:r for r in raw}
            if len(lookup)!=len(raw) or set(lookup)!={r['record_id'] for r in rows}: raise ValueError('Reference cohort differs')
            if any(lookup[r['record_id']]['code_sha256']!=r['code_sha256'] for r in rows): raise ValueError('Reference code differs')
            reference=[lookup[r['record_id']]['prediction'] for r in rows]
            report['full_reference']=binary_metrics(y,reference)
            report['paired_reference_to_cascade']=paired_comparison(y,reference,values,bootstrap=config['bootstrap'])
            report['paired_reference_to_gate_only']=paired_comparison(y,reference,[r['gate_prediction'] for r in p],bootstrap=config['bootstrap'])
            if report['inference']['reference_incremental']['status']!='complete': report['status']='partial'
        elif config['full_reference']: report['status']='partial'
        write_json(folder/'evaluation.json',report);reports.append(report)
    summary={'config':config,'status':'complete' if all(r['status']=='complete' for r in reports) else 'partial','datasets':reports,
             'note':'Routing saving is separate from cache saving and API retry counts; CP selection bounds do not guarantee test quality.'}
    write_json(root/'runs'/'experiment_summary.json',summary);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','preflight','infer','evaluate','all']);p.add_argument('--root',type=Path);p.add_argument('--allow-api',action='store_true')
    a=p.parse_args();root=resolve_root(a.root);config=read_json(root/'configs'/'experiment.json');result=None
    if a.stage in ('prepare','all'): result=prepare_all(root,config)
    if a.stage in ('preflight','all'): result=preflight_all(root,config)
    if a.stage in ('infer','all'): result=infer_all(root,config,a.allow_api)
    if a.stage in ('evaluate','all'): result=evaluate_all(root,config)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
