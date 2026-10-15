"""CPU logistic gate, disjoint Platt calibration and finite-grid risk selection."""
from pathlib import Path
import os,time
import numpy as np
from scipy.special import expit
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn,scipy
from grace_h1.io import resolve_root,read_json,write_json,file_hash,fingerprint,row_identity
from grace_h1.data import load_rows,load_embeddings,embedding_recipe

def source_identity(root):
    root=resolve_root(root)
    return {p.relative_to(root).as_posix():file_hash(p) for package in ('grace_h1','grace_h3') for p in sorted((root/package).glob('*.py'))}

def upper(k,n,alpha):
    return 1.0 if not n or k==n else float(beta.ppf(1-alpha,k+1,n-k))

def lower(k,n,alpha):
    return 0.0 if not n or not k else float(beta.ppf(alpha,k,n-k+1))

def choose_policy(y,p,cfg,monotone=True):
    lows=cfg['low_grid'];highs=cfg['high_grid'] if cfg['enable_direct_high'] else []
    if not lows or any(not 0<t<.5 for t in lows) or any(not .5<t<1 for t in highs): raise ValueError('Use predefined low<.5 and high>.5 grids')
    if not 0<cfg['family_alpha']<1 or not 0<cfg['max_lost_positive_fraction']<1: raise ValueError('Invalid risk budget')
    if not 0<cfg['min_high_precision']<1 or not 0<cfg['max_high_false_positive_fraction']<1: raise ValueError('Invalid high-branch constraints')
    # High branch uses two bounds; allocate the family error across all tests.
    alpha=cfg['family_alpha']/(len(lows)+2*len(highs));positive=int((y==1).sum());safe=int((y==0).sum())
    low_rows=[];high_rows=[]
    for t in sorted(set(lows)):
        skip=p<t;fn=int((skip&(y==1)).sum());bound=upper(fn,positive,alpha)
        low_rows.append({'threshold':t,'count':int(skip.sum()),'fn':fn,'positive_denominator':positive,
                         'lost_positive_fraction_upper':bound,'eligible':bool(monotone and skip.any() and bound<=cfg['max_lost_positive_fraction'])})
    for t in sorted(set(highs)):
        high=p>=t;tp=int((high&(y==1)).sum());fp=int((high&(y==0)).sum())
        precision=lower(tp,int(high.sum()),alpha);fpr=upper(fp,safe,alpha)
        high_rows.append({'threshold':t,'count':int(high.sum()),'fp':fp,'precision_lower':precision,'safe_fpr_upper':fpr,
                          'eligible':bool(monotone and high.any() and precision>=cfg['min_high_precision'] and fpr<=cfg['max_high_false_positive_fraction'])})
    good=[r for r in low_rows if r['eligible']];best=max(good,key=lambda r:(r['count'],-r['threshold'])) if good else None
    good=[r for r in high_rows if r['eligible']];high=max(good,key=lambda r:(r['count'],r['threshold'])) if good else None
    return {'low':best['threshold'] if best else 0.0,'high':high['threshold'] if high else None,
            'direct_high_requested':cfg['enable_direct_high'],'alpha_per_bound':alpha,'family_alpha':cfg['family_alpha'],
            'monotone_calibration':monotone,'selection_low':low_rows,'selection_high':high_rows,
            'bound_method':'one-sided Clopper-Pearson, Bonferroni over the predefined finite grid',
            'scope':'nominal under iid/stationarity; no domain-shift or project-cluster guarantee'}

def calibration_metrics(y,p):
    p=np.clip(p,1e-12,1-1e-12);bins=np.minimum((p*10).astype(int),9);ece=0.0
    for b in range(10):
        mask=bins==b
        if mask.any(): ece+=float(mask.mean())*abs(float(y[mask].mean()-p[mask].mean()))
    return {'brier':float(np.mean((p-y)**2)),'nll':float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
            'ece_10_equal_width':ece,'n':len(y),'prevalence':float(y.mean())}

def fit_gate(root,dataset,cfg):
    root=resolve_root(root);start=time.perf_counter()
    if cfg['C']<=0 or cfg['max_iter']<1 or not 0<cfg['calibration_fraction']<1 or cfg['class_weight'] not in (None,'balanced'): raise ValueError('Invalid gate config')
    train=load_rows(root,dataset,'train',labels=True);valid=load_rows(root,dataset,'validation',labels=True)
    x,tm=load_embeddings(root,dataset,'train',train);v,vm=load_embeddings(root,dataset,'validation',valid)
    if embedding_recipe(tm)!=embedding_recipe(vm): raise ValueError('Gate train/validation encoder recipe differs')
    def groups(rows):
        result={}
        for i,r in enumerate(rows): result.setdefault(r['code_sha256'],[]).append(i)
        return result
    tg=groups(train);vg=groups(valid)
    ti=[ids[0] for ids in tg.values() if len({train[i]['label'] for i in ids})==1]
    vi=[ids[0] for h,ids in vg.items() if h not in tg and len({valid[i]['label'] for i in ids})==1]
    y=np.asarray([r['label'] for r in train]);vy=np.asarray([r['label'] for r in valid])
    if len(np.unique(y[ti]))!=2 or min(np.bincount(vy[vi],minlength=2))<4: raise ValueError('Insufficient independent validation groups for two partitions')
    ci,si=train_test_split(vi,train_size=cfg['calibration_fraction'],stratify=vy[vi],random_state=cfg['seed'])
    ci=sorted(ci);si=sorted(si)
    spec={'dataset':dataset,'config':cfg,'source':source_identity(root),'data_manifest':file_hash(root/'data'/'manifest.json'),
          'recipe':embedding_recipe(tm),'embedding_files':[tm['files'],vm['files']],
          'train_identity':row_identity(train),'validation_identity':row_identity(valid),'fit_indices':ti,'calibration_indices':ci,'selection_indices':si}
    key=fingerprint(spec);folder=root/'artifacts'/'gates'/dataset/key[:16];folder.mkdir(parents=True,exist_ok=True)
    final=folder/'manifest.json'
    if final.exists():
        m=read_json(final)
        if m['fingerprint']!=key or file_hash(root/m['weights'])!=m['weights_sha256']: raise ValueError('Gate artifact changed')
        return m
    a=np.asarray(x[ti],dtype=np.float64);mean=a.mean(0);scale=np.maximum(a.std(0),1e-5);a=(a-mean)/scale
    model=LogisticRegression(C=cfg['C'],class_weight=cfg['class_weight'],max_iter=cfg['max_iter'],tol=1e-5,solver='lbfgs',random_state=cfg['seed'])
    model.fit(a,y[ti]);del a
    if int(model.n_iter_.max())>=cfg['max_iter']: raise RuntimeError('Gate did not converge; increase max_iter before looking at test')
    weight=model.coef_[0];bias=float(model.intercept_[0])
    def raw(indices): return ((np.asarray(v[indices],dtype=np.float64)-mean)/scale)@weight+bias
    calibrator=LogisticRegression(C=1e6,max_iter=cfg['max_iter'],tol=1e-8,solver='lbfgs')
    calibrator.fit(raw(ci)[:,None],vy[ci])
    if int(calibrator.n_iter_.max())>=cfg['max_iter']: raise RuntimeError('Calibration did not converge')
    slope=float(calibrator.coef_[0,0]);intercept=float(calibrator.intercept_[0])
    p=expit(slope*raw(si)+intercept);policy=choose_policy(vy[si],p,cfg,monotone=slope>0)
    tmp=folder/'weights.tmp.npz';np.savez_compressed(tmp,mean=mean,scale=scale,weight=weight,bias=bias,calibration_slope=slope,calibration_intercept=intercept);os.replace(tmp,folder/'weights.npz')
    peak=None
    if os.name!='nt':
        import resource,sys
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    manifest={'schema':'grace_h3_gate_v1','fingerprint':key,'spec':spec,'policy':policy,
              'weights':(folder/'weights.npz').relative_to(root).as_posix(),'weights_sha256':file_hash(folder/'weights.npz'),
              'calibration':calibration_metrics(vy[ci],expit(slope*raw(ci)+intercept)),
              'selection_raw':calibration_metrics(vy[si],expit(raw(si))),'selection_calibrated':calibration_metrics(vy[si],p),
              'counts':{'train_total':len(train),'fit_unique_unambiguous':len(ti),'validation_total':len(valid),
                        'validation_eligible_unique':len(vi),'calibration':len(ci),'selection':len(si)},
              'fit_seconds':time.perf_counter()-start,'process_peak_rss_bytes':peak,
              'versions':{'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__},
              'note':'No query/test labels used for fitting/calibration/threshold selection; no guarantee under domain shift'}
    write_json(final,manifest);return manifest

def gate_scores(root,dataset,split,manifest):
    root=resolve_root(root);rows=load_rows(root,dataset,split);x,meta=load_embeddings(root,dataset,split,rows)
    if embedding_recipe(meta)!=manifest['spec']['recipe']: raise ValueError('Query gate encoder recipe differs')
    path=root/manifest['weights'];path.resolve().relative_to(root/'artifacts'/'gates')
    if file_hash(path)!=manifest['weights_sha256']: raise ValueError('Gate weights changed')
    with np.load(path,allow_pickle=False) as w:
        out=[]
        for start in range(0,len(rows),2048):
            raw=((np.asarray(x[start:start+2048],dtype=np.float64)-w['mean'])/w['scale'])@w['weight']+w['bias']
            out.extend(expit(w['calibration_slope']*raw+w['calibration_intercept']).tolist())
    if not np.isfinite(out).all(): raise ValueError('Non-finite gate scores')
    return rows,out
