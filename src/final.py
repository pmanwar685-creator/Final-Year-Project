"""Final SOH + RUL pipeline. Two protocols:
 A) leave-one-battery-out (strict, unseen battery)
 B) chronological split per battery (train first 60% cycles of all 4 batteries, test remaining)
Outputs: results.json, plots/*.png, models/*.joblib"""
import json, os, numpy as np, pandas as pd, joblib, warnings
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error, r2_score
from rul_exp2 import traj_feats, TF
import train as T
warnings.filterwarnings('ignore')
df, FEATS, eol = T.df, T.FEATS, T.eol
bats = list(eol); RF = ['gap','slope20','slope40','slope_all']
os.makedirs('results/plots', exist_ok=True); os.makedirs('results/models', exist_ok=True)

soh_m = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3,3,20)))
rul_m = lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
def metr(y, p): return dict(MAE=float(mae(y,p)), RMSE=float(np.sqrt(mean_squared_error(y,p))), R2=float(r2_score(y,p)))

def fit_predict(train_idx_fn, test_b, mode):
    """Return (soh_pred, rul_pred, test_df) for test battery; train set defined by mode."""
    d = df.copy()
    if mode == 'LOBO': trmask = d.battery != test_b
    else: trmask = d.cycle <= (d.groupby('battery').cycle.transform('max')*0.6).astype(int)
    trb = [b for b in bats if (b != test_b if mode=='LOBO' else True)]
    # out-of-fold SOH for training batteries (stacking)
    soh = np.zeros(len(d))
    for c in trb:
        m_in = trmask & (d.battery != c) if mode=='LOBO' else trmask & (d.battery != c)
        if mode == 'CHRONO':   # in-battery: use own early cycles + others' early cycles
            m_in = trmask
        soh[(d.battery==c).values] = soh_m().fit(d[m_in][FEATS], d[m_in].soh).predict(d[d.battery==c][FEATS])
    if mode == 'LOBO':
        soh[(d.battery==test_b).values] = soh_m().fit(d[trmask][FEATS], d[trmask].soh).predict(d[d.battery==test_b][FEATS])
    d['soh_pred'] = soh
    TF_ = {b: traj_feats(d[d.battery==b].soh_pred.values, d[d.battery==b].cycle.values) for b in bats}
    for b in bats: TF_[b]['rul'] = d[d.battery==b].rul.values; TF_[b]['cycle_']=d[d.battery==b].cycle.values; TF_['eol']=None
    return d, TF_

results = {'LOBO': {}, 'CHRONO': {}}
fig, axs = plt.subplots(2, 4, figsize=(18, 7))
# ---- Protocol A
for i, tb in enumerate(bats):
    d, TFs = fit_predict(None, tb, 'LOBO')
    tr = pd.concat([TFs[b] for b in bats if b != tb]); te = TFs[tb]
    rul = rul_m().fit(tr[RF], np.log1p(tr.rul)); p = np.clip(np.expm1(rul.predict(te[RF])), 0, None)
    ds = d[d.battery==tb]; m = (ds.cycle <= ds.eol).values
    results['LOBO'][tb] = {'SOH': metr(ds.soh, ds.soh_pred), 'RUL': metr(ds.rul[m], p[m])}
    axs[0,i].plot(ds.cycle, ds.soh, 'k', label='true'); axs[0,i].plot(ds.cycle, ds.soh_pred, 'r--', label='pred')
    axs[0,i].axhline(70, c='gray', ls=':'); axs[0,i].set_title(f'{tb} SOH (MAE {results["LOBO"][tb]["SOH"]["MAE"]:.2f}%)')
    axs[1,i].plot(ds.cycle[m], ds.rul[m], 'k'); axs[1,i].plot(ds.cycle[m], p[m], 'r--')
    axs[1,i].set_title(f'{tb} RUL (MAE {results["LOBO"][tb]["RUL"]["MAE"]:.1f} cyc)')
axs[0,0].legend(); plt.tight_layout(); plt.savefig('results/plots/lobo_results.png', dpi=110); plt.close()

# ---- Protocol B: chronological 60/40 within every battery
d, TFs = fit_predict(None, None, 'CHRONO')
maxc = d.groupby('battery').cycle.transform('max'); trm = (d.cycle <= (maxc*0.6).astype(int)).values
tr = pd.concat([TFs[b][(d[d.battery==b].cycle <= int(d[d.battery==b].cycle.max()*0.6)).values] for b in bats])
rul = rul_m().fit(tr[RF], np.log1p(tr.rul))
fig, axs = plt.subplots(2, 4, figsize=(18, 7))
for i, b in enumerate(bats):
    ds = d[d.battery==b]; t = (ds.cycle > int(ds.cycle.max()*0.6)).values
    p = np.clip(np.expm1(rul.predict(TFs[b][RF])), 0, None)
    m = t & (ds.cycle <= ds.eol).values
    results['CHRONO'][b] = {'SOH': metr(ds.soh[t], ds.soh_pred[t]), 'RUL': metr(ds.rul[m], p[m]) if m.sum()>2 else None}
    axs[0,i].plot(ds.cycle, ds.soh, 'k'); axs[0,i].plot(ds.cycle[t], ds.soh_pred[t], 'r--'); axs[0,i].set_title(f'{b} SOH test (MAE {results["CHRONO"][b]["SOH"]["MAE"]:.2f}%)')
    axs[1,i].plot(ds.cycle[m], ds.rul[m], 'k'); axs[1,i].plot(ds.cycle[m], p[m], 'r--')
    axs[1,i].set_title(f'{b} RUL test'+(f' (MAE {results["CHRONO"][b]["RUL"]["MAE"]:.1f})' if results['CHRONO'][b]['RUL'] else ''))
plt.tight_layout(); plt.savefig('results/plots/chrono_results.png', dpi=110); plt.close()

for pr in results:
    for t in ('SOH','RUL'):
        v = [results[pr][b][t] for b in bats if results[pr][b][t]]
        results[pr][f'mean_{t}'] = {k: float(np.mean([x[k] for x in v])) for k in ('MAE','RMSE','R2')}
json.dump(results, open('results/results.json','w'), indent=1)
print(json.dumps({p:{k:v for k,v in r.items()} for p,r in results.items()}, indent=1))

# ---- final deployable models: trained on all 4 batteries
soh_f = soh_m().fit(df[FEATS], df.soh)
oof = np.zeros(len(df))
for c in bats:
    mk = df.battery != c; oof[(df.battery==c).values] = soh_m().fit(df[mk][FEATS], df[mk].soh).predict(df[df.battery==c][FEATS])
d2 = df.copy(); d2['soh_pred'] = oof
tr = pd.concat([traj_feats(d2[d2.battery==b].soh_pred.values, d2[d2.battery==b].cycle.values).assign(rul=d2[d2.battery==b].rul.values) for b in bats])
rul_f = rul_m().fit(tr[RF], np.log1p(tr.rul))
joblib.dump({'soh': soh_f, 'rul': rul_f, 'soh_features': FEATS, 'rul_features': RF}, 'results/models/soh_rul_models.joblib')
