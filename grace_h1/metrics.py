from __future__ import annotations
import math
import numpy as np

def binary_metrics(labels,predictions):
    y=np.asarray(labels,dtype=int);p=np.asarray([-1 if v is None else int(v) for v in predictions],dtype=int)
    if len(y)!=len(p) or not len(y) or not np.isin(y,[0,1]).all(): raise ValueError('Invalid evaluation arrays')
    valid=np.isin(p,[0,1]);a=y[valid];b=p[valid]
    tp=int(((a==1)&(b==1)).sum());tn=int(((a==0)&(b==0)).sum())
    fp=int(((a==0)&(b==1)).sum());fn=int(((a==1)&(b==0)).sum())
    div=lambda n,d:n/d if d else 0.0
    precision=div(tp,tp+fp);recall=div(tp,tp+fn)
    denominator=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return {'n':len(y),'valid_n':int(valid.sum()),'invalid_n':int((~valid).sum()),'invalid_rate':float((~valid).mean()),
        'confusion_matrix_valid_only':{'tp':tp,'tn':tn,'fp':fp,'fn':fn},
        'precision':precision,'recall':recall,'f1':div(2*precision*recall,precision+recall),
        'mcc':div(tp*tn-fp*fn,denominator),'balanced_accuracy':(recall+div(tn,tn+fp))/2,
        'accuracy_valid_only':div(tp+tn,int(valid.sum())), 'accuracy_all':div(tp+tn,len(y)),
        'recall_all_positives':div(tp,int((y==1).sum())),
        'predicted_positive_rate_all':float((p==1).mean()),'invalid_positive_n':int(((y==1)&~valid).sum()),
        'metric_scope':'binary metrics condition on valid labels; invalids retained and reported separately'}

def paired_comparison(labels,left,right,*,seed=42,bootstrap=1000):
    y=np.asarray(labels);a=np.asarray([-1 if x is None else x for x in left]);b=np.asarray([-1 if x is None else x for x in right])
    if len(y)!=len(a) or len(y)!=len(b): raise ValueError('Paired lengths differ')
    mask=np.isin(a,[0,1])&np.isin(b,[0,1]);yv=y[mask];av=a[mask];bv=b[mask]
    if not len(yv): return {'common_valid_n':0,'status':'no_common_valid_predictions'}
    ma=binary_metrics(yv,av);mb=binary_metrics(yv,bv);keys=('f1','mcc','recall','precision')
    delta={k:mb[k]-ma[k] for k in keys};rng=np.random.default_rng(seed);draws={k:[] for k in keys}
    for _ in range(bootstrap):
        idx=rng.integers(0,len(yv),len(yv));aa=binary_metrics(yv[idx],av[idx]);bb=binary_metrics(yv[idx],bv[idx])
        for k in keys: draws[k].append(bb[k]-aa[k])
    return {'common_valid_n':len(yv),'excluded_invalid_pair_n':int((~mask).sum()),'delta_right_minus_left':delta,
            'ci95_query_bootstrap':{k:np.quantile(v,[.025,.975]).tolist() for k,v in draws.items()},
            'bootstrap_repetitions':bootstrap,'note':'query bootstrap; repeated code/project dependence is not cluster-adjusted'}
