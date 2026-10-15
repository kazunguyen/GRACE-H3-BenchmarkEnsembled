"""Train a small supervised metric on frozen, cached encoder vectors."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import os,time
import numpy as np
from .io import resolve_root, fingerprint, file_hash, write_json, read_json
from .data import load_training_data,embedding_recipe
from .metrics import binary_metrics

@dataclass(frozen=True)
class TrainingConfig:
    objective: str = 'supcon_hard'
    dimension: int = 128
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    supcon_weight: float = 0.1
    hard_negative_weight: float = 0.1
    temperature: float = 0.1
    margin: float = 0.2
    class_weighted_ce: bool = True
    patience: int = 6
    seed: int = 42
    device: str = 'auto'

    def validate(self):
        if self.objective not in ('ce','supcon','supcon_hard'): raise ValueError('Unknown projection objective')
        if min(self.dimension,self.epochs,self.batch_size,self.patience)<1 or self.batch_size<4: raise ValueError('Invalid training sizes')
        if self.temperature<=0 or self.learning_rate<=0 or self.weight_decay<0: raise ValueError('Invalid training scales')
        if self.supcon_weight<0 or self.hard_negative_weight<0 or self.margin<0: raise ValueError('Invalid loss weights')

def contrastive_losses(z,labels,temperature,margin):
    import torch
    n=len(labels);eye=torch.eye(n,device=z.device,dtype=torch.bool)
    positive=labels[:,None].eq(labels[None,:]) & ~eye
    negative=labels[:,None].ne(labels[None,:])
    sim=z@z.T;logits=sim/temperature
    log_prob=logits-torch.logsumexp(logits.masked_fill(eye,float('-inf')),dim=1,keepdim=True)
    counts=positive.sum(1);active=counts.gt(0)
    # torch.where avoids 0 * -inf on the diagonal (which would produce NaN).
    log_positive=torch.where(positive,log_prob,torch.zeros_like(log_prob)).sum(1)
    sup=-(log_positive[active]/counts[active]).mean() if active.any() else z.sum()*0
    usable=active & negative.any(1)
    if usable.any():
        mean_positive=torch.where(positive,sim,torch.zeros_like(sim)).sum(1)/counts.clamp_min(1)
        hardest=sim.masked_fill(~negative,float('-inf')).max(1).values
        hard=torch.relu(margin+hardest[usable]-mean_positive[usable]).mean()
    else: hard=z.sum()*0
    return sup,hard

def normalization(x,indices,chunk=2048):
    count=0;total=np.zeros(x.shape[1],dtype=np.float64);squared=np.zeros_like(total)
    for start in range(0,len(indices),chunk):
        a=np.asarray(x[indices[start:start+chunk]],dtype=np.float64)
        count+=len(a);total+=a.sum(0);squared+=(a*a).sum(0)
    mean=total/count;variance=np.maximum(squared/count-mean*mean,0)
    return mean.astype(np.float32),np.maximum(np.sqrt(variance),1e-5).astype(np.float32)

def train_projection(root,dataset,config=None):
    import torch
    from torch import nn
    root=resolve_root(root);cfg=config or TrainingConfig();cfg.validate()
    train,valid,x,v,tm,vm,vi=load_training_data(root,dataset)
    # Conflicting labels for identical code do not form impossible hard pairs.
    groups={}
    for i,row in enumerate(train): groups.setdefault(row['code_sha256'],[]).append(i)
    ti=[idxs[0] for idxs in groups.values() if len({train[i]['label'] for i in idxs})==1]
    ti=np.asarray(ti,dtype=np.int64);vi=np.asarray(vi,dtype=np.int64)
    y=np.asarray([r['label'] for r in train]);vy=np.asarray([r['label'] for r in valid])
    if len(np.unique(y[ti]))!=2 or len(np.unique(vy[vi]))!=2: raise ValueError('Need both classes for training/validation')
    sources={name:file_hash(Path(__file__).with_name(name)) for name in ('projection.py','data.py','metrics.py')}
    signature=fingerprint({'dataset':dataset,'config':asdict(cfg),'train':tm['rows_identity'],'validation':vm['rows_identity'],
                           'recipe':embedding_recipe(tm),'data_manifest':file_hash(root/'data'/'manifest.json'),'training_indices':ti.tolist(),'selection_indices':vi.tolist(),'source':sources})
    folder=root/'artifacts'/'heads'/dataset/cfg.objective/signature[:16];folder.mkdir(parents=True,exist_ok=True)
    final=folder/'manifest.json'
    if final.exists():
        m=read_json(final)
        if m['training_fingerprint']!=signature or file_hash(folder/'projection.npz')!=m['projection_sha256']: raise ValueError('Projection cache changed')
        print(f'{dataset}/{cfg.objective}: reuse {folder}',flush=True);return m
    device=('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device=='auto' else cfg.device
    torch.manual_seed(cfg.seed);np.random.seed(cfg.seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(cfg.seed);torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    mean,scale=normalization(x,ti)
    class Head(nn.Module):
        def __init__(self):
            super().__init__();self.projection=nn.Linear(x.shape[1],cfg.dimension);self.classifier=nn.Linear(cfg.dimension,2)
        def forward(self,e):
            z=nn.functional.normalize(self.projection(e),dim=1,eps=1e-8)
            return z,self.classifier(z)
    model=Head().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    counts=np.bincount(y[ti],minlength=2);weights=(len(ti)/(2*counts)) if cfg.class_weighted_ce else np.ones(2)
    ce=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    best=-float('inf');start=0;stale=0;history=[];checkpoint=folder/'last.pt';started=time.perf_counter()
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=True)
        if saved['fingerprint']!=signature: raise ValueError('Training resume fingerprint mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['epoch']+1;best=saved['best'];stale=saved['stale'];history=saved['history']
    def tensor(matrix,indices):
        a=(np.asarray(matrix[indices],dtype=np.float32)-mean)/scale
        return torch.from_numpy(a).to(device)
    for epoch in range(start,cfg.epochs):
        if stale>=cfg.patience: break
        model.train();order=np.random.default_rng(cfg.seed+epoch).permutation(ti);loss_total=0.0
        for offset in range(0,len(order),cfg.batch_size):
            ids=order[offset:offset+cfg.batch_size]
            if len(ids)<2: continue
            labels=torch.tensor(y[ids],device=device,dtype=torch.long)
            z,logits=model(tensor(x,ids));loss=ce(logits,labels)
            if cfg.objective!='ce':
                sup,hard=contrastive_losses(z,labels,cfg.temperature,cfg.margin)
                loss=loss+cfg.supcon_weight*sup
                if cfg.objective=='supcon_hard': loss=loss+cfg.hard_negative_weight*hard
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            loss_total+=float(loss.detach())*len(ids)
        model.eval();pred=[]
        with torch.no_grad():
            for offset in range(0,len(vi),cfg.batch_size):
                _,logits=model(tensor(v,vi[offset:offset+cfg.batch_size]));pred.extend(logits.argmax(1).cpu().tolist())
        metrics=binary_metrics(vy[vi],pred);history.append({'epoch':epoch+1,'loss':loss_total/len(order),'validation':metrics})
        if metrics['mcc']>best:
            best=metrics['mcc'];stale=0
            arrays={'mean':mean,'scale':scale,'weight':model.projection.weight.detach().cpu().numpy(),
                    'bias':model.projection.bias.detach().cpu().numpy(),
                    'classifier_weight':model.classifier.weight.detach().cpu().numpy(),
                    'classifier_bias':model.classifier.bias.detach().cpu().numpy()}
            temp=folder/'projection.tmp.npz';np.savez_compressed(temp,**arrays);os.replace(temp,folder/'projection.npz')
            write_json(folder/'best.json',{'epoch':epoch+1,'metrics':metrics})
        else: stale+=1
        saved={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'stale':stale,
               'history':history,'fingerprint':signature}
        torch.save(saved,folder/'last.tmp.pt');os.replace(folder/'last.tmp.pt',checkpoint)
        write_json(folder/'history.json',history)
        print(f'{dataset}/{cfg.objective} epoch {epoch+1}: loss={history[-1]["loss"]:.4f}, val MCC={metrics["mcc"]:.4f}',flush=True)
        if stale>=cfg.patience: break
    if not (folder/'projection.npz').exists(): raise RuntimeError('No trained projection checkpoint')
    meta={'schema':'grace_h1_projection_v1','dataset':dataset,'objective':cfg.objective,'config':asdict(cfg),
          'training_fingerprint':signature,'encoder_recipe':embedding_recipe(tm),'source':sources,
          'train_identity':tm['rows_identity'],'validation_identity':vm['rows_identity'],
          'train_rows':len(train),'training_unique_unambiguous_rows':len(ti),'validation_selection_rows':len(vi),
          'validation_excluded_train_duplicates':len(valid)-len(vi),'selection':'best validation head MCC at argmax; proxy for retrieval',
          'best':read_json(folder/'best.json'),'path':(folder/'projection.npz').relative_to(root).as_posix(),
          'projection_sha256':file_hash(folder/'projection.npz'),'device':device,'torch':torch.__version__,
          'parameters':sum(p.numel() for p in model.parameters()),'numpy':np.__version__,
          'gpu_name':torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
          'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
          'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else None,
          'train_seconds_this_session':time.perf_counter()-started}
    write_json(final,meta);return meta

def project_embeddings(matrix,manifest,root,batch_size=2048):
    root=resolve_root(root);path=root/manifest['path']
    if file_hash(path)!=manifest['projection_sha256']: raise ValueError('Projection weights changed')
    with np.load(path,allow_pickle=False) as weights:
        mean,scale,w,b=[weights[k] for k in ('mean','scale','weight','bias')]
    if matrix.shape[1]!=len(mean): raise ValueError('Embedding dimension differs from projection')
    out=np.empty((len(matrix),len(b)),dtype=np.float32)
    for start in range(0,len(matrix),batch_size):
        e=(np.asarray(matrix[start:start+batch_size],dtype=np.float32)-mean)/scale
        z=e@w.T+b;out[start:start+len(e)]=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)
    return out

def whitening(train,queries,dimension=256):
    # Fit on train only. Query/test never enters covariance or model selection.
    mean=np.asarray(train,dtype=np.float64).mean(0);center=np.asarray(train,dtype=np.float64)-mean
    cov=center.T@center/max(len(train)-1,1);values,vectors=np.linalg.eigh(cov)
    order=np.argsort(values)[::-1][:min(dimension,train.shape[1])]
    w=vectors[:,order]/np.sqrt(np.maximum(values[order],1e-8))[None,:]
    def transform(x):
        z=(np.asarray(x,dtype=np.float64)-mean)@w
        return (z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)).astype(np.float32)
    return transform(train),transform(queries),{'fit':'train_only','dimension':len(order)}
