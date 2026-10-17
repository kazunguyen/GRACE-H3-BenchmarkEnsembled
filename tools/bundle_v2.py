from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grace_h3v2.portable import create_bundle, import_bundle

p = argparse.ArgumentParser()
p.add_argument('action', choices=('inputs', 'output', 'checkpoint', 'import'))
p.add_argument('--phase', choices=('base', 'meta'), default='base')
p.add_argument('--kind', default='output')
p.add_argument('--path', type=Path, required=True)
a = p.parse_args()
result = (import_bundle(ROOT, a.path, a.kind, a.phase) if a.action == 'import'
          else create_bundle(ROOT, a.path, a.action, a.phase))
print(json.dumps(result, indent=2))
