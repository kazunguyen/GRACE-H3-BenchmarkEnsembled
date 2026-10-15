from __future__ import annotations
import hashlib, json, os, uuid
from pathlib import Path

def sha256(value: bytes | str) -> str:
    return hashlib.sha256(value.encode('utf-8') if isinstance(value,str) else value).hexdigest()

def file_hash(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def fingerprint(value) -> str:
    return sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')))

def iter_jsonl(path: Path):
    with Path(path).open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try: yield json.loads(line)
                except ValueError as exc: raise ValueError(f'Invalid JSONL {path}:{i}') from exc

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def write_jsonl(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def row_identity(rows):
    return fingerprint([(r['record_id'],r['code_sha256']) for r in rows])

def resolve_root(root=None):
    return Path(root).expanduser().resolve() if root else Path(__file__).resolve().parents[1]


def load_local_environment(root):
    path=Path(root)/'.env'
    if path.is_file():
        from dotenv import load_dotenv
        load_dotenv(path,override=False)
    if os.name!='nt':
        value=os.environ.get('GOOGLE_APPLICATION_CREDENTIALS','')
        if len(value)>3 and value[1:3] in (':/',':\\'):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
