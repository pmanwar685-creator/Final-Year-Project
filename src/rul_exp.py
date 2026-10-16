import numpy as np, pandas as pd, warnings
from train import *
warnings.filterwarnings('ignore')
soh_pred = {}
for b in eol:
    tr, te = df[df.battery!=b], df[df.battery==b]
    m = models()['Ridge'].fit(tr[FEATS], tr.soh)
    soh_pred[b] = m.predict(te[FEATS])
    print(b, 'SOH MAE', mae(te.soh, soh_pred[b]))

def extrap(s, k, win, kind, thr=70.0, ema=0.3):
    # s: predicted soh array (causal up to idx k); returns predicted RUL at cycle index k
    y = pd.Series(s[:k+1]).ewm(alpha=ema).mean().values
    lo = max(0, k-win+1); x = np.arange(lo, k+1); yy = y[lo:k+1]
    if len(x) < 5: return np.nan
    if kind == 'lin':
        a, b0 = np.polyfit(x, yy, 1)
        if a >= -1e-6: return 200.
        xc = (thr-b0)/a
    elif kind == 'quad':
        p = np.polyfit(x, yy, 2)
        r = np.roots(p - np.array([0,0,thr])); r = r[np.isreal(r)].real; r = r[r >= k]
        xc = r.min() if len(r) else k+200
    elif kind == 'exp':
        yl = np.log(np.clip(yy - 50, 1e-3, None))
        a, b0 = np.polyfit(x, yl, 1)
        if a >= -1e-6: return 200.
        xc = (np.log(thr-50)-b0)/a
    return max(xc - k, 0)

for kind in ('lin','quad','exp'):
  for win in (20, 40, 80):
    for ema in (0.2, 0.5):
        out = []
        for b in eol:
            te = df[df.battery==b].reset_index(drop=True)
            p = np.array([extrap(soh_pred[b], k, win, kind, ema=ema) for k in range(len(te))])
            msk = (te.cycle <= te.eol) & (te.cycle >= 10)
            out.append(mae(te.rul[msk], np.clip(np.nan_to_num(p[msk], nan=50), 0, 200)))
        print(kind, win, ema, np.round(out,1), round(np.mean(out),2))
