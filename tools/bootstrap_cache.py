"""Copy an independent cache snapshot; never modify the source project."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args();source=a.source.resolve()
    files=[source/'data'/'manifest.json']
    for d in ('devign','reveal'):
        for s in ('train','validation','test'):
            files.append(source/'data'/d/(s+'.jsonl'))
            files += [source/'cache'/'embeddings'/d/s/n for n in ('vectors.npy','index.jsonl','manifest.json')]
    for name in ('graphs','legacy_candidates','prior_runs','responses'):
        files += [f for f in (source/'cache'/name).rglob('*') if f.is_file() and '.tmp' not in f.name]
    def copy(f):
        rel=f.relative_to(source);out=ROOT/rel;out.parent.mkdir(parents=True,exist_ok=True);h=digest(f)
        if not out.exists() or digest(out)!=h:
            temp=out.with_name(out.name+'.import.tmp');shutil.copyfile(f,temp)
            if digest(temp)!=h: raise ValueError(f'Source changed while copying: {rel}')
            temp.replace(out)
        return {'path':rel.as_posix(),'bytes':out.stat().st_size,'sha256':h}
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(copy,sorted(set(files))),1):
            records.append(r)
            if i%2000==0: print('Copied/verified',i,'files',flush=True)
    index=ROOT/'cache'/'import_manifest.json';index.parent.mkdir(exist_ok=True)
    index.write_text(json.dumps({'source':str(source),'files':records},indent=2)+'\n',encoding='utf-8')
    summary={'source':str(source),'files':len(records),'bytes':sum(r['bytes'] for r in records),
             'graph_features':sum(r['path'].startswith('cache/graphs/') for r in records),
             'index_sha256':digest(index),'index':'cache/import_manifest.json','independent_copies':True}
    (ROOT/'provenance').mkdir(exist_ok=True)
    (ROOT/'provenance'/'cache_import.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    runtime=ROOT/'bundles'/'grace-h3-joern-runtime.zip';runtime.parent.mkdir(exist_ok=True)
    old=source/'bundles'/'grace-h1-joern-runtime.zip'
    if old.exists(): shutil.copyfile(old,runtime);summary['joern_runtime_sha256']=digest(runtime)
    env=source/'.env';dest=ROOT/'.env'
    if env.is_file() and not dest.exists():
        keep=[line for line in env.read_text(encoding='utf-8').splitlines() if line.split('=',1)[0].strip() in ('JOERN_HOME','VERTEX_PROJECT','VERTEX_LOCATION')]
        dest.write_text('\n'.join(keep)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
