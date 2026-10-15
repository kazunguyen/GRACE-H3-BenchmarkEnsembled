"""CPU input/output/checkpoint archives with per-file hashes and strict imports."""
from pathlib import Path,PurePosixPath
from concurrent.futures import ThreadPoolExecutor
import json,os,shutil,tempfile,zipfile
from grace_h1.io import resolve_root,read_json,write_json,file_hash
SCHEMA='grace_h3_bundle_v1'

def identity(root):
    return {p.relative_to(root).as_posix():file_hash(p) for package in ('grace_h1','grace_h3') for p in sorted((root/package).glob('*.py'))}

def create_bundle(root,destination,kind='cpu_inputs'):
    root=resolve_root(root);destination=Path(destination);files=[]
    job={'source':identity(root),'data_manifest':file_hash(root/'data'/'manifest.json'),'config':read_json(root/'configs'/'experiment.json')}
    if kind=='cpu_inputs':
        for package in ('grace_h1','grace_h3'): files+=list((root/package).glob('*.py'))
        files+=list(root.glob('requirements*.txt'))+list((root/'configs').glob('*'))+[root/'data'/'manifest.json']
        for d in ('devign','reveal'):
            for s in ('train','validation','test'):
                files.append(root/'data'/d/(s+'.jsonl'))
                files += [root/'cache'/'embeddings'/d/s/n for n in ('vectors.npy','index.jsonl','manifest.json')]
        graphs=[p for p in (root/'cache'/'graphs').rglob('*') if p.name in ('features.json','features.json.gz')]
        inventory=root/'cache'/'cpu_inventory.json'
        write_json(inventory,{'job':job,'base_graph_paths':[p.relative_to(root).as_posix() for p in sorted(graphs)]})
        files+=graphs+[inventory]
        for name in ('legacy_candidates','candidates','projected'):
            files += [p for p in (root/'cache'/name).rglob('*') if p.is_file() and '.tmp' not in p.name]
        files += [p for p in (root/'artifacts'/'gates').rglob('*') if p.is_file() and '.tmp' not in p.name]
        # Optional H1 heads are needed only when the user explicitly stacks H1 + H3.
        if job['config']['retrieval_arm']!='baseline_whitening':
            files += [p for p in (root/'artifacts'/'heads').rglob('*') if p.is_file() and '.tmp' not in p.name]
            files.append(root/'artifacts'/'training_selection.json')
    elif kind in ('cpu_output','cpu_checkpoint'):
        from .workflow import selected_jobs
        inventory=read_json(root/'cache'/'cpu_inventory.json')
        if inventory['job']!=job: raise ValueError('CPU input config/source/data changed; recreate input bundle')
        base=set(inventory['base_graph_paths'])
        for name in ('graphs','candidates','projected'):
            files += [p for p in (root/'cache'/name).rglob('*') if p.is_file() and '.tmp' not in p.name
                      and (name!='graphs' or (p.name in ('features.json','features.json.gz') and p.relative_to(root).as_posix() not in base))]
        files += [p for p in (root/'artifacts').rglob('*') if p.is_file() and '.tmp' not in p.name]
        for pattern in ('freeze.json','freeze_h3.json'):
            for frozen in sorted((root/'runs').rglob(pattern)):
                files.append(frozen)
                files.extend(root/p for p in read_json(frozen)['files'] if p.startswith(('runs/','artifacts/')))
        if kind=='cpu_output':
            selected_jobs(root,job['config']);files.append(root/'runs'/'h3_selection.json')
    else: raise ValueError('Unsupported export kind')
    files=sorted(set(files))
    def entry(p):
        if not p.is_file(): raise FileNotFoundError(p)
        return {'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p),'mode':p.stat().st_mode&0o777}
    with ThreadPoolExecutor(max_workers=4) as pool: entries=list(pool.map(entry,files))
    manifest={'schema':SCHEMA,'kind':kind,'job':job,'entries':entries,'no_credentials':True}
    destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.writestr('bundle_manifest.json',json.dumps(manifest,sort_keys=True,indent=2))
        for i,(p,e) in enumerate(zip(files,entries),1):
            z.write(p,e['path'])
            if i%5000==0: print('Bundled',i,'files',flush=True)
    os.replace(temp,destination)
    return {'path':str(destination),'kind':kind,'files':len(files),'bytes':destination.stat().st_size,'sha256':file_hash(destination)}

def import_bundle(root,archive,kind='cpu_output'):
    root=resolve_root(root);root.mkdir(parents=True,exist_ok=True);copied=reused=0
    with zipfile.ZipFile(archive) as z:
        m=json.loads(z.read('bundle_manifest.json'))
        legacy=kind in ('joern_runtime','trained_heads') and m.get('schema')=='grace_h1_bundle_v1'
        if m.get('kind')!=kind or (m.get('schema')!=SCHEMA and not legacy): raise ValueError('Wrong bundle kind/schema')
        if kind not in ('cpu_inputs','cpu_output','cpu_checkpoint','joern_runtime','trained_heads'): raise ValueError('Unsupported import kind')
        if kind in ('cpu_output','cpu_checkpoint'):
            expected={'source':identity(root),'data_manifest':file_hash(root/'data'/'manifest.json'),'config':read_json(root/'configs'/'experiment.json')}
            if m['job']!=expected: raise ValueError('CPU output config/source/data differs')
        entries=m['entries'];names=[e['path'] for e in entries]
        if len({n.casefold() for n in names})!=len(names) or sorted(z.namelist())!=sorted(names+['bundle_manifest.json']): raise ValueError('Duplicate/unexpected archive members')
        for e in entries:
            name=e['path'];p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or ':' in name or chr(92) in name: raise ValueError('Unsafe archive path')
            if kind in ('cpu_output','cpu_checkpoint') and not name.startswith(('cache/graphs/','cache/candidates/','cache/projected/','artifacts/','runs/')): raise ValueError('Unexpected CPU output member')
            if kind=='joern_runtime' and not name.startswith('tools/joern/joern-cli/'): raise ValueError('Unexpected Joern member')
            if kind=='trained_heads' and not name.startswith('artifacts/'): raise ValueError('Unexpected H1 head member')
            target=root/name;target.resolve().relative_to(root)
            if target.exists() and file_hash(target)!=e['sha256']: raise ValueError(f'Existing content differs: {name}; use a new experiment folder')
        with tempfile.TemporaryDirectory(prefix='h3-import-',dir=root) as scratch:
            for e in entries:
                p=Path(scratch)/e['path'];p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(e['path']) as src,p.open('wb') as out: shutil.copyfileobj(src,out,1024*1024)
                if p.stat().st_size!=e['bytes'] or file_hash(p)!=e['sha256']: raise ValueError('Bundle checksum mismatch')
                if os.name!='nt': p.chmod(e.get('mode',0o644)&0o777)
            for e in entries:
                target=root/e['path']
                if target.exists(): reused+=1;continue
                target.parent.mkdir(parents=True,exist_ok=True);os.replace(Path(scratch)/e['path'],target);copied+=1
    return {'kind':kind,'copied':copied,'reused_identical':reused}
