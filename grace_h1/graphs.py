from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import gzip,json,os,re,shutil,subprocess,tempfile,uuid
from .io import load_local_environment,sha256,file_hash,fingerprint,resolve_root
from .reference import parse_graphml

class GraphStore:
    def __init__(self,root,joern_home=None):
        self.root=resolve_root(root);load_local_environment(self.root)
        self.home=str(joern_home or os.environ.get('JOERN_HOME','/home/yungn/tools/joern/y/joern-cli'))
        self.reused=0;self.built=0

    def exists(self,code_hash):
        card=self.root/'cache'/'graphs'/code_hash
        return (card/'features.json.gz').is_file() or (card/'features.json').is_file()

    @lru_cache(maxsize=16)
    def get(self,code_hash):
        card=self.root/'cache'/'graphs'/code_hash;path=card/'features.json.gz'
        if not path.exists(): path=card/'features.json'
        raw=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
        graph=json.loads(raw);expected=graph.get('feature_sha256')
        if graph.get('code_sha256')!=code_hash or expected!=fingerprint({k:v for k,v in graph.items() if k!='feature_sha256'}):
            raise ValueError(f'Graph feature bytes/identity changed: {path}')
        if graph.get('schema') not in ('joern_legacy_text_v1','joern_whole_graph_v3','grace_h1_graph_v1'):
            raise ValueError('Unsupported graph feature schema')
        return graph

    def _run(self,name,args):
        tool=self.home.rstrip('/\\')+'/'+name
        flags=['-J-Xms128m','-J-Xmx2g','-J-XX:ActiveProcessorCount=4']
        def wsl(value):
            value=str(value)
            if re.match(r'^[A-Za-z]:[/\\]',value): return '/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
            return value
        command=(['wsl.exe','--exec',tool,*flags,*map(wsl,args)] if os.name=='nt' and tool.startswith('/')
                 else [tool,*flags,*map(str,args)])
        try: result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
        except FileNotFoundError as exc: raise RuntimeError('Joern missing. Set JOERN_HOME and use the documented WSL kernel.') from exc
        if result.returncode: raise RuntimeError(f'{name} failed: {(result.stderr+result.stdout)[-2000:]}')

    def ensure(self,row):
        h=row['code_sha256']
        if sha256(row['code'])!=h: raise ValueError('Code bytes changed before Joern')
        if self.exists(h):
            graph=self.get(h);self.reused+=1;return graph
        card=self.root/'cache'/'graphs'/h;card.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(card).free<1024**3: raise RuntimeError('Joern needs at least 1 GB free disk')
        # Linux uses native scratch; Windows+WSL can still use translated paths.
        with tempfile.TemporaryDirectory(prefix='grace-h1-') as temp:
            scratch=Path(temp);source=scratch/'target.c';binary=scratch/'target.bin';export=scratch/'export'
            source.write_bytes(row['code'].encode('utf-8'))
            self._run('joern-parse',[source,'--output',binary])
            self._run('joern-export',['--repr','all','--format','graphml','--out',export,binary])
            matches=list(export.glob('*.xml'))+list(export.glob('*.graphml'))
            if len(matches)!=1: raise RuntimeError('No unique whole-CPG GraphML export')
            graph=parse_graphml(matches[0])
            graph.update(schema='grace_h1_graph_v1',code_sha256=h,source_cpg_uncompressed_sha256=file_hash(binary),
                         graphml_sha256=file_hash(matches[0]),source='joern_parse_export_all',joern_home=self.home)
            graph['feature_sha256']=fingerprint(graph)
            temp_path=card/('features.'+uuid.uuid4().hex+'.tmp.gz')
            with gzip.open(temp_path,'wt',encoding='utf-8') as f: json.dump(graph,f,ensure_ascii=False,sort_keys=True)
            os.replace(temp_path,card/'features.json.gz')
        self.get.cache_clear();self.built+=1;return self.get(h)
