from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import inspect,os,re,threading,time
import numpy as np
import yaml
from .io import load_local_environment,resolve_root,read_json,write_json,write_jsonl,iter_jsonl,file_hash,fingerprint,sha256
from .pipeline import verify_frozen
from .parser import parse_prediction
from .fpt_provider import load_fpt_settings, fpt_identity, fpt_sdk


def request_spec(root):
    contract=yaml.safe_load((root/'configs'/'reference_contract.yaml').read_text(encoding='utf-8'))
    settings=load_fpt_settings()
    identity=fpt_identity(settings) if settings is not None else contract['backbone_llm']
    return {**identity,'inference':contract['inference_config']}

def request_key(prompt_sha256,spec):
    return fingerprint({'prompt_sha256':prompt_sha256,'inference_fingerprint':fingerprint(spec)})

def response_path(root,key): return root/'cache'/'responses'/(key+'.json')

def read_cached(root,key,prompt_hash,spec):
    path=response_path(root,key)
    if not path.exists(): return None
    row=read_json(path)
    if row.get('request_key')!=key or row.get('prompt_sha256')!=prompt_hash or row.get('request_spec')!=spec:
        raise ValueError('Response cache identity mismatch')
    if row.get('raw_response_sha256')!=sha256(row.get('raw_response','')): raise ValueError('Response cache content changed')
    return row if row.get('request_status')=='ok' else None

def import_prior_responses(root,spec):
    signature=fingerprint(spec);contract_sha=file_hash(root/'configs'/'reference_contract.yaml')
    imported=0
    for manifest_path in sorted((root/'cache'/'prior_runs').glob('*/*/*/run_manifest.json')):
        manifest=read_json(manifest_path)
        if manifest.get('contract_sha256')!=contract_sha: continue
        for path in sorted((manifest_path.parent/'inference').glob('*.jsonl')):
            # The source file was copied into this project. Never modify it.
            with path.open(encoding='utf-8') as handle:
                for line in handle:
                    if not line.strip(): continue
                    try:
                        import json
                        row=json.loads(line)
                    except ValueError:
                        if not line.endswith('\n'): break # an interrupted checkpoint tail
                        raise
                    if (row.get('request_status')!='ok' or row.get('inference_fingerprint')!=signature or
                        row.get('model')!=spec['model'] or not isinstance(row.get('prompt'),str) or
                        sha256(row['prompt'])!=row.get('prompt_sha256')): continue
                    key=request_key(row['prompt_sha256'],spec);dest=response_path(root,key)
                    if dest.exists(): continue
                    raw=row.get('raw_response','')
                    write_json(dest,{'request_key':key,'prompt_sha256':row['prompt_sha256'],'request_spec':spec,
                        'raw_response':raw,'raw_response_sha256':sha256(raw),'request_status':'ok',
                        'attempts':0,'usage':None,'elapsed_seconds':None,'source':'prior_run_exact_match',
                        'source_file':path.relative_to(root).as_posix()})
                    imported+=1
    return imported

def plan_inference(root,run_dir):
    root=resolve_root(root);run_dir=Path(run_dir);manifest=verify_frozen(root,run_dir)
    spec=request_spec(root);imported=import_prior_responses(root,spec)
    groups={};hits=0
    for prompt in iter_jsonl(run_dir/'prompts.jsonl'):
        if sha256(prompt['prompt'])!=prompt['prompt_sha256']: raise ValueError('Prompt bytes changed')
        key=request_key(prompt['prompt_sha256'],spec);groups.setdefault(key,[]).append(prompt)
        hits+=read_cached(root,key,prompt['prompt_sha256'],spec) is not None
    todo=[key for key,rows in groups.items() if read_cached(root,key,rows[0]['prompt_sha256'],spec) is None]
    report={'queries':manifest['target_count'],'unique_prompts':len(groups),'cached_queries':hits,
            'new_unique_requests':len(todo),'imported_prior_responses':imported,'request_spec':spec,
            'worst_case_attempts':len(todo)*int(re.search(r'max_attempts=(\d+)',spec['inference']['retry_policy'])[1])}
    write_json(run_dir/'inference_plan.json',report);return report

def run_inference(root,run_dir,*,allow_api=False,workers=4):
    root=resolve_root(root);load_local_environment(root);run_dir=Path(run_dir)
    if workers<1: raise ValueError('workers must be positive')
    plan=plan_inference(root,run_dir);spec=plan['request_spec'];groups={}
    for prompt in iter_jsonl(run_dir/'prompts.jsonl'):
        key=request_key(prompt['prompt_sha256'],spec);groups.setdefault(key,[]).append(prompt)
    todo=[(key,rows[0]) for key,rows in groups.items() if read_cached(root,key,rows[0]['prompt_sha256'],spec) is None]
    if todo and not allow_api:
        return plan|{'status':'prepared_cache_only','action':'run again with allow_api=True to call Vertex for missing exact prompts'}
    clients=[];client_lock=threading.Lock();local=threading.local()
    settings=load_fpt_settings()
    if todo and settings is not None:
        genai,types=fpt_sdk(settings)
        mode,project,location='vertexai',None,None
        if not settings['api_key']: raise RuntimeError('Set FPT_API in the shared LabThayTho .env')
    elif todo:
        project=os.environ.get('VERTEX_PROJECT') or os.environ.get('GOOGLE_CLOUD_PROJECT')
        location=os.environ.get('VERTEX_LOCATION') or os.environ.get('GOOGLE_CLOUD_LOCATION')
        if not project or not location: raise RuntimeError('Set VERTEX_PROJECT and VERTEX_LOCATION; use existing ADC credentials')
        from google import genai
        from google.genai import types
        parameters=inspect.signature(genai.Client).parameters
        mode='enterprise' if 'enterprise' in parameters else 'vertexai'
        if mode not in parameters: raise RuntimeError('Unsupported google-genai Client; install requirements.txt')
    inf=spec['inference'];attempt_limit=int(re.search(r'max_attempts=(\d+)',inf['retry_policy'])[1])
    delay=re.search(r'exponential_backoff=(\d+)s\.\.(\d+)s',inf['retry_policy'])
    if not delay: raise ValueError('Unsupported retry policy')
    def request(key,prompt):
        client=getattr(local,'client',None)
        raw='';error=None;usage=None;started=time.perf_counter()
        for attempt in range(1,attempt_limit+1):
            try:
                if client is None:
                    client=genai.Client(**{mode:True,'project':project,'location':location,
                                          'http_options':types.HttpOptions(timeout=120000)})
                    local.client=client
                    with client_lock: clients.append(client)
                response=client.models.generate_content(model=spec['model'],contents=prompt['prompt'],
                    config=types.GenerateContentConfig(temperature=inf['temperature'],seed=inf['seed'],max_output_tokens=inf['max_tokens']))
                raw=response.text or '';error=None
                u=getattr(response,'usage_metadata',None)
                if u is not None:
                    usage={name:getattr(u,name,None) for name in ('prompt_token_count','candidates_token_count','total_token_count')}
                break
            except Exception as exc:
                error=repr(exc)
                if getattr(exc,'code',None) in (400,401,403,404): break
                if attempt<attempt_limit: time.sleep(min(int(delay[2]),int(delay[1])*2**(attempt-1)))
        return {'request_key':key,'prompt_sha256':prompt['prompt_sha256'],'request_spec':spec,'raw_response':raw,
                'raw_response_sha256':sha256(raw),'request_status':'ok' if error is None else 'exhausted',
                'request_error':error,'attempts':attempt,'usage':usage,'elapsed_seconds':time.perf_counter()-started,'source':'new_api'}
    current={};started=time.perf_counter()
    try:
        if todo:
            with ThreadPoolExecutor(max_workers=min(workers,len(todo))) as pool:
                futures={pool.submit(request,key,p):key for key,p in todo}
                for n,future in enumerate(as_completed(futures),1):
                    row=future.result();key=row['request_key'];current[key]=row
                    write_json(run_dir/'requests'/(key+'.json'),row)
                    if row['request_status']=='ok': write_json(response_path(root,key),row)
                    if n%25==0 or n==len(todo): print(f'LLM unique requests {n}/{len(todo)}',flush=True)
    finally:
        for client in clients: client.close()
    predictions=[];responses=[]
    for key,rows in groups.items():
        response=current.get(key) or read_cached(root,key,rows[0]['prompt_sha256'],spec)
        if response is None: raise RuntimeError('Missing inference result')
        parsed=parse_prediction(response['raw_response']) if response['request_status']=='ok' else parse_prediction('')
        for p in rows:
            predictions.append({'record_id':p['record_id'],'code_sha256':p['code_sha256'],'prediction':parsed.prediction,
                                'parse_status':parsed.parse_status,'parse_method':parsed.parse_method,'request_key':key,
                                'request_status':response['request_status'],'reused':key not in current,
                                'prompt_sha256':p['prompt_sha256']})
            responses.append({'record_id':p['record_id'],'code_sha256':p['code_sha256'],**response})
    order={p['record_id']:i for i,p in enumerate(iter_jsonl(run_dir/'prompts.jsonl'))}
    predictions.sort(key=lambda p:order[p['record_id']]);responses.sort(key=lambda p:order[p['record_id']])
    write_jsonl(run_dir/'predictions.jsonl',predictions);write_jsonl(run_dir/'raw_responses.jsonl',responses)
    errors=sum(r['request_status']!='ok' for r in predictions)
    usage={};usage_coverage={}
    for key in ('prompt_token_count','candidates_token_count','total_token_count'):
        known=[r['usage'][key] for r in current.values() if r.get('usage') and r['usage'].get(key) is not None]
        usage[key]=sum(known) if known else (0 if not current else None)
        usage_coverage[key]=len(known)
    latencies=[r['elapsed_seconds'] for r in current.values()]
    report=plan|{'status':'complete' if not errors else 'partial_request_errors','request_error_queries':errors,
                 'api_attempts_this_run':sum(r['attempts'] for r in current.values()),'new_unique_requests':len(current),
                 'cached_queries':sum(p['reused'] for p in predictions),'provider_usage_new_requests':usage,
                 'usage_available_new_requests':sum(r.get('usage') is not None for r in current.values()),
                 'usage_coverage_per_field':usage_coverage,
                 'new_request_latency_p50_p95_seconds':np.quantile(latencies,[.5,.95]).tolist() if latencies else None,
                 'wall_seconds':time.perf_counter()-started,'predictions_sha256':file_hash(run_dir/'predictions.jsonl')}
    write_json(run_dir/'inference_summary.json',report);return report
