"""Stdlib-only stage bundles. Import only archives created by this project."""
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import tempfile
import zipfile

SCHEMA = 'h3v2_bundle_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def under(root, path):
    normalize = lambda p: Path(str(Path(p).resolve()).removeprefix(chr(92)*2+'?'+chr(92)))
    normalize(path).relative_to(normalize(root))


def identity(root):
    return {'source': {p.relative_to(root).as_posix(): digest(p)
                       for package in ('grace_h1', 'grace_h3v2')
                       for p in sorted((root/package).glob('*.py'))},
            'config': read(root/'configs/benchmark_v2.json'),
            'data_manifest': digest(root/'data/manifest.json'),
            'runtime': digest(root/'requirements-v2.txt'),
            'contract': digest(root/'configs/reference_contract.yaml'),
            'template': digest(root/'configs/prompt_template.txt')}


def walk(folder):
    return [p for p in folder.rglob('*') if p.is_file()
            and not p.name.endswith(('.tmp', '.copytmp')) and '__pycache__' not in p.parts]


def run_files(root):
    result = []
    for frozen in (root/'runs').rglob('freeze.json'):
        if '/h3v2/' in frozen.as_posix():
            continue
        result.append(frozen)
        result.extend(root/rel for rel in read(frozen)['files']
                      if rel.startswith(('runs/', 'artifacts/')))
        for name in ('predictions.jsonl', 'inference_summary.json'):
            p = frozen.parent/name
            if p.exists():
                result.append(p)
    result += walk(root/'runs/h3v2')
    return result


def create_bundle(root, destination, kind='inputs', phase='base'):
    root, destination = Path(root).resolve(), Path(destination)
    if kind not in ('inputs', 'output', 'checkpoint') or phase not in ('base', 'meta'):
        raise ValueError('Unknown bundle stage')
    job = identity(root)
    inventory = root/'cache/h3v2'/('input_inventory_'+phase+'.json')
    files = []
    if kind == 'inputs':
        for package in ('grace_h1', 'grace_h3v2'):
            files += list((root/package).glob('*.py'))
        files += [root/n for n in ('requirements-v2.txt', 'requirements.txt', 'requirements-cpu.txt')]
        files += [root/'configs'/n for n in ('benchmark_v2.json', 'reference_contract.yaml',
                                           'prompt_template.txt')]
        files += [root/'data/manifest.json']
        for dataset in job['config']['datasets']:
            for split in ('train', 'validation', 'test'):
                files += [root/'data'/dataset/(split+'.jsonl')]
                files += [root/'cache/embeddings'/dataset/split/n
                          for n in ('vectors.npy', 'index.jsonl', 'manifest.json')]
        graphs = [p for p in walk(root/'cache/graphs')
                  if p.name in ('features.json', 'features.json.gz')]
        inventory.parent.mkdir(parents=True, exist_ok=True)
        inventory.write_text(json.dumps({'job': job, 'base_graphs': [
            p.relative_to(root).as_posix() for p in sorted(graphs)]}, indent=2)+'\n',
                             encoding='utf-8')
        files += graphs + [inventory]
        for name in ('legacy_candidates', 'candidates', 'projected'):
            files += walk(root/'cache'/name)
        files += walk(root/'artifacts/h3v2') + run_files(root)
    else:
        if not inventory.exists() or read(inventory)['job'] != job:
            raise ValueError('Missing/mismatched input inventory; import matching CPU inputs first')
        if kind == 'output':
            from .workflow import load_stage
            load_stage(root, job['config'], 'base_selection' if phase == 'base' else 'meta_selection')
        base = set(read(inventory)['base_graphs'])
        files += [p for p in walk(root/'cache/graphs')
                  if p.name in ('features.json', 'features.json.gz')
                  and p.relative_to(root).as_posix() not in base]
        for name in ('candidates', 'projected'):
            files += walk(root/'cache'/name)
        files += walk(root/'artifacts/h3v2') + run_files(root)
        if kind == 'checkpoint':
            for name in ('fits', 'meta_fits'):
                files += walk(root/'cache/h3v2'/name)
    files = sorted(set(files))
    def entry(p):
        if not p.is_file():
            raise FileNotFoundError(p)
        return {'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                'sha256': digest(p), 'mode': p.stat().st_mode & 0o777}
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(entry, files))
    manifest = {'schema': SCHEMA, 'kind': kind, 'phase': phase, 'job': job, 'entries': entries}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.writestr('bundle_manifest.json', json.dumps(manifest, sort_keys=True, indent=2))
        for i, (p, e) in enumerate(zip(files, entries), 1):
            z.write(p, e['path'])
            if i % 5000 == 0:
                print('Bundled', i, 'files', flush=True)
    os.replace(temp, destination)
    return {'path': str(destination), 'phase': phase, 'kind': kind,
            'files': len(files), 'bytes': destination.stat().st_size, 'sha256': digest(destination)}


def import_bundle(root, archive, kind='output', phase='base'):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('bundle_manifest.json'))
        runtime = kind == 'joern_runtime'
        if runtime:
            if manifest.get('schema') != 'grace_h1_bundle_v1' or manifest.get('kind') != kind:
                raise ValueError('Wrong Joern archive')
        elif (manifest.get('schema') != SCHEMA or manifest.get('kind') != kind
              or manifest.get('phase') != phase):
            raise ValueError('Wrong H3-v2 bundle kind/phase')
        if kind in ('output', 'checkpoint') and manifest['job'] != identity(root):
            raise ValueError('CPU config/source/data/runtime differs')
        if kind not in ('inputs', 'output', 'checkpoint', 'joern_runtime'):
            raise ValueError('Unknown import kind')
        entries = manifest['entries']
        names = [e['path'] for e in entries]
        if len({n.casefold() for n in names}) != len(names):
            raise ValueError('Duplicate archive member')
        if sorted(z.namelist()) != sorted(names+['bundle_manifest.json']):
            raise ValueError('Unexpected archive members')
        for e in entries:
            name = e['path']
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or ':' in name or chr(92) in name:
                raise ValueError('Unsafe archive path')
            under(root, root/name)
            if runtime and not name.startswith('tools/joern/joern-cli/'):
                raise ValueError('Unexpected runtime member')
            if kind in ('output', 'checkpoint') and not name.startswith(
                    ('artifacts/h3v2/', 'cache/graphs/', 'cache/candidates/',
                     'cache/projected/', 'cache/h3v2/', 'runs/')):
                raise ValueError('Unexpected CPU result member')
            target = root/name
            if target.exists() and digest(target) != e['sha256']:
                raise ValueError(f'Existing content differs: {name}; use a fresh stage workspace')
        copied = reused = 0
        with tempfile.TemporaryDirectory(prefix='h3v2-import-', dir=root) as scratch:
            for e in entries:
                p = Path(scratch)/e['path']
                p.parent.mkdir(parents=True, exist_ok=True)
                with z.open(e['path']) as source, p.open('wb') as out:
                    shutil.copyfileobj(source, out, 1024*1024)
                if p.stat().st_size != e['bytes'] or digest(p) != e['sha256']:
                    raise ValueError('Bundle checksum differs')
                if os.name != 'nt':
                    p.chmod(e.get('mode', 0o644) & 0o777)
            for e in entries:
                target = root/e['path']
                if target.exists():
                    reused += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(Path(scratch)/e['path'], target)
                copied += 1
    return {'kind': kind, 'phase': phase, 'copied': copied, 'reused_identical': reused}
