"""LSTM over trajectory features derived from (nested out-of-fold) Ridge SOH predictions."""
import numpy as np, pandas as pd, sys
from sklearn.metrics import mean_absolute_error as mae
import lstm as L
from rul_exp2 import oof_soh, traj_feats
df, bats = L.df, L.bats
cols = ['soh_s','drop','slope20','slope40','slope_all','gap']
res = []
for tb in bats:
    sp = oof_soh(tb); parts = []
    for b in bats:
        g = df[df.battery==b].reset_index(drop=True)
        t = traj_feats(sp[b], g.cycle.values)
        parts.append(pd.concat([g[['battery','cycle','soh','rul','eol']], t[cols]], axis=1))
    d = pd.concat(parts); tr, te = d[d.battery!=tb], d[d.battery==tb]
    soh, rul = L.fit_predict(tr, te, cols, seeds=5, epochs=120)
    m = (te.cycle <= te.eol).values
    r = (tb, mae(te.soh, soh), mae(te.rul[m], rul[m])); res.append(r); print(r, flush=True)
print('mean SOH MAE %.3f  RUL MAE %.2f' % (np.mean([r[1] for r in res]), np.mean([r[2] for r in res])))
