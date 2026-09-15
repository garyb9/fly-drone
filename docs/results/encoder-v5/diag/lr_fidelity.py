import sys, numpy as np, torch
from fly_drone.encoder import LearnedEncoder, v4_targets
d=np.load('runs/v5/clone/data.npz'); fl=d['flight']
u=np.unique(fl); rng=np.random.default_rng(72); held=rng.choice(u,max(1,int(len(u)*0.1)),replace=False)
ids=np.flatnonzero(np.isin(fl,held))
stacks=d['stacks'][ids]  # npz re-reads the whole array on every key access: load once
enc=LearnedEncoder.load(sys.argv[1]); out=[]
with torch.no_grad():
    for i in range(0,len(ids),512):
        x=torch.as_tensor(stacks[i:i+512],dtype=torch.float32)/255.
        out.append((torch.tanh(enc.mu(enc.extractor({'eyes':x})))+1).numpy())
p=np.concatenate(out); t=v4_targets(d['cues'][ids])
for a,name in ((0,'mi1'),(2,'tm3'),(4,'lc4'),(6,'lplc2')):
    dt=t[:,a]-t[:,a+1]; dp=p[:,a]-p[:,a+1]
    print(f'{name}: per-ch r L {np.corrcoef(t[:,a],p[:,a])[0,1]:.3f}  L-R r {np.corrcoef(dt,dp)[0,1]:.3f}  gain {dp.std()/dt.std():.2f}  rmse {np.sqrt(np.mean((dt-dp)**2)):.3f}  |pred diff| when v4 diff==0 {np.abs(dp[dt==0]).mean():.3f}')
