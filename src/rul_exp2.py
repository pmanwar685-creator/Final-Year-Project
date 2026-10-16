import numpy as np, pandas as pd, warnings
from train import *
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')
bats = list(eol)
def soh_model(): return models()['Ridge']
def oof_soh(test_b):
    """SOH preds for every battery, none produced by a model that saw test_b."""
    out = {}
    others = [b for b in bats if b != test_b]
    for c in others:
        tr = df[df.battery.isin([x for x in others if x != c])]
        out[c] = soh_model().fit(tr[FEATS], tr.soh).predict(df[df.battery==c][FEATS])
    tr = df[df.battery.isin(others)]
    out[test_b] = soh_model().fit(tr[FEATS], tr.soh).predict(df[df.battery==test_b][FEATS])
    return out

def traj_feats(s, cyc):
    s = pd.Series(s); sm = s.ewm(alpha=0.3).mean()
    f = pd.DataFrame({'cycle': cyc, 'soh_s': sm.values})
    f['drop'] = sm.iloc[0:5].mean() - f.soh_s  # fade since start
    for w in (20, 40):
        sl = []
        for k in range(len(sm)):
            lo = max(0, k-w+1)
            sl.append(np.polyfit(np.arange(lo,k+1), sm.values[lo:k+1], 1)[0] if k-lo >= 4 else np.nan)
        f[f'slope{w}'] = sl
    f['slope_all'] = -f['drop'] / np.maximum(f.cycle, 1)
    f['gap'] = f.soh_s - 70
    f['rul_lin'] = (f.gap / np.clip(-f.slope_all, 1e-3, None)).clip(upper=250)
    f['rul_lin40'] = (f.gap / np.clip(-f.slope40, 1e-3, None)).clip(upper=250)
    return f.bfill().fillna(0)

TF = ['cycle','soh_s','drop','slope20','slope40','slope_all','gap','rul_lin','rul_lin40']
def run(model_fn, feats, logt=True, name=''):
    out = []
    for tb in bats:
        sp = oof_soh(tb)
        T = {b: traj_feats(sp[b], df[df.battery==b].cycle.values) for b in bats}
        for b in bats: T[b]['rul'] = df[df.battery==b].rul.values; T[b]['eol']=df[df.battery==b].eol.values
        tr = pd.concat([T[b] for b in bats if b != tb]); te = T[tb]
        y = np.log1p(tr.rul) if logt else tr.rul
        p = model_fn().fit(tr[feats], y).predict(te[feats])
        p = np.expm1(p) if logt else p
        m = (te.cycle <= te.eol)
        out.append(mae(te.rul[m], np.clip(p[m],0,None)))
    print(f'{name:35s}', np.round(out,1), round(np.mean(out),2))
if __name__ == '__main__':
    for fs_name, fs in [('all',TF), ('no cycle',[f for f in TF if f!='cycle']), ('lin only',['rul_lin','rul_lin40','gap']), ('gap+slopes',['gap','slope20','slope40','slope_all'])]:
        for mn, mf in [('Ridge', lambda: make_pipeline(StandardScaler(), Ridge(1.0))),
                       ('ET', lambda: ExtraTreesRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0)),
                       ('GBR', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0))]:
            for logt in (True, False):
                run(mf, fs, logt, f'{fs_name}/{mn}/log={logt}')
