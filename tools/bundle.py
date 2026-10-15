from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from grace_h3.portable import create_bundle,import_bundle
p=argparse.ArgumentParser();p.add_argument('action',choices=['cpu-inputs','cpu-output','cpu-checkpoint','import']);p.add_argument('--path',type=Path);p.add_argument('--kind',default='cpu_output')
a=p.parse_args()
if a.action=='import':
    if not a.path: p.error('--path is required')
    result=import_bundle(ROOT,a.path,a.kind)
else: result=create_bundle(ROOT,a.path or ROOT/'bundles'/('grace-h3-'+a.action+'.zip'),a.action.replace('-','_'))
print(json.dumps(result,indent=2))
