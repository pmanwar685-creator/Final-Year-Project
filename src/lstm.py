"""Multi-task LSTM (SOH + RUL) from sliding windows of per-cycle features; leave-one-battery-out."""
import numpy as np, pandas as pd, torch, torch.nn as nn, warnings, sys, json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae
import train as T
warnings.filterwarnings('ignore'); torch.set_num_threads(8)
df, eol = T.df, T.eol; bats = list(eol)
BASE = T.base; W = 15

def windows(d, cols, w=W):
    X = d[cols].values; out = []
    for k in range(len(X)):
        idx = np.clip(np.arange(k-w+1, k+1), 0, None); out.append(X[idx])
    return np.stack(out).astype(np.float32)

class Net(nn.Module):
    def __init__(s, nf, h=32):
        super().__init__(); s.lstm = nn.LSTM(nf, h, batch_first=True); s.do = nn.Dropout(0.2)
        s.head = nn.Sequential(nn.Linear(h, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(s, x): o, _ = s.lstm(x); return s.head(s.do(o[:, -1]))

def fit_predict(tr, te, cols, seeds=5, epochs=150):
    sc = StandardScaler().fit(tr[cols]); 
    def prep(d): d = d.copy(); d[cols] = sc.transform(d[cols]); return d
    trs = [prep(tr[tr.battery==b]) for b in tr.battery.unique()]
    Xtr = np.concatenate([windows(g, cols) for g in trs]); 
    ytr = np.concatenate([np.c_[(g.soh.values-85)/10, np.log1p(g.rul.values)/5] for g in trs]).astype(np.float32)
    Xte = windows(prep(te), cols); preds = []
    for s in range(seeds):
        torch.manual_seed(s); m = Net(len(cols)); opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=1e-2)
        Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
        for ep in range(epochs):
            m.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 64):
                b = perm[i:i+64]; x = Xt[b] + 0.05*torch.randn_like(Xt[b])
                loss = nn.functional.smooth_l1_loss(m(x), yt[b], beta=0.1); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): preds.append(m(torch.tensor(Xte)).numpy())
    p = np.mean(preds, 0)
    return p[:,0]*10+85, np.clip(np.expm1(p[:,1]*5), 0, None)

if __name__ == '__main__':
    cols = BASE if len(sys.argv) < 2 else BASE+['cycle']
    res = []
    for tb in bats:
        tr, te = df[df.battery!=tb], df[df.battery==tb]
        soh, rul = fit_predict(tr, te, cols)
        m = (te.cycle <= te.eol).values
        r = (tb, mae(te.soh, soh), mae(te.rul[m], rul[m])); res.append(r); print(r, flush=True)
    print('mean SOH MAE %.3f  RUL MAE %.2f' % (np.mean([r[1] for r in res]), np.mean([r[2] for r in res])))
