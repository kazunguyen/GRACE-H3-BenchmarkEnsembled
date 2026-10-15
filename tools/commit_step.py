"""Apply the user's size and conventional-commit rules to this isolated repo."""
from pathlib import Path
import argparse, re, subprocess
ROOT=Path(__file__).resolve().parents[1]
LIMIT=100*1024*1024

def git(*args, capture=False):
    return subprocess.run(['git','-C',str(ROOT),*args],check=True,
                          capture_output=capture,text=True,encoding='utf-8')

def paths(*args):
    return [p for p in git(*args,capture=True).stdout.split('\0') if p]

def sizes(names):
    oversized=[]
    for name in sorted(set(names)):
        p=ROOT/name
        size=p.stat().st_size if p.is_file() else 0
        print(f'{size:>12} bytes  {name}',flush=True)
        if size>LIMIT: oversized.append(name)
    return oversized

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['stage','commit'])
    ap.add_argument('--message');ap.add_argument('--paths',nargs='*');args=ap.parse_args()
    staged=paths('diff','--cached','--name-only','-z')
    if args.action=='stage':
        names=(args.paths if args.paths is not None else paths('ls-files','--modified','--others','--exclude-standard','-z'))+staged
        large=sizes(names)
        if large:
            ignore=ROOT/'.gitignore';text=ignore.read_text(encoding='utf-8')
            for name in large:
                if name in staged: git('restore','--staged','--',name)
                rule='/'+name.replace('\\','/')
                if rule not in text.splitlines(): text+='\n'+rule+'\n'
            ignore.write_text(text,encoding='utf-8')
            names=[n for n in names if n not in large]+['.gitignore']
        git('status','--short');git('diff','--stat')
        sizes(names) # Immediately before git add, including staged files.
        git('add','--',*sorted(set(names)))
        git('diff','--cached','--check');git('diff','--cached','--stat')
    else:
        if not args.message or not re.match(r'^(feat|fix|docs|style|refactor|perf|test|build|ci|chore)(\(.+\))?!?: .{1,50}$',args.message):
            raise ValueError('Use the copied commit-convention.md')
        if sizes(staged): raise ValueError('Oversized file is staged')
        git('diff','--cached','--check');git('commit','-m',args.message)
        git('status','--short')
if __name__=='__main__': main()
