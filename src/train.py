"""Leave-one-battery-out evaluation of SOH and RUL regressors on NASA B0005/6/7/18."""
import numpy as np, pandas as pd, warnings, sys
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error, r2_score
warnings.filterwarnings('ignore')

NOMINAL, EOL_CAP = 2.0, 1.4
df = pd.read_csv('results/features.csv')
df['soh'] = df.capacity / NOMINAL * 100
eol = {}
for b, g in df.groupby('battery'):
    hit = g[g.capacity <= EOL_CAP + 0.001]
    eol[b] = int(hit.cycle.iloc[0]) if len(hit) else int(g.cycle.max())
df['eol'] = df.battery.map(eol)
df['rul'] = (df.eol - df.cycle).clip(lower=0)
print('EOL cycle per battery:', eol)

# smooth features over a short trailing window (reduces measurement noise, causal)
base = [c for c in df.columns if c not in ('battery','cycle','capacity','soh','eol','rul')]
def add_rolling(d, w=5):
    out = d.copy()
    for c in base:
        out[c+'_ma'] = d.groupby('battery')[c].transform(lambda s: s.rolling(w, min_periods=1).mean())
    return out
df = add_rolling(df)
FEATS = base + [c+'_ma' for c in base] + ['cycle']

def models():
    return {
      'Ridge': make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3,3,20))),
      'SVR': make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.01, gamma='scale')),
      'ExtraTrees': ExtraTreesRegressor(500, min_samples_leaf=2, n_jobs=-1, random_state=0),
      'GBR': GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0),
    }

def evaluate(target, feats, scale=1.0):
    res = {}
    for name in models():
        rows = []
        for test_b in eol:
            tr, te = df[df.battery != test_b], df[df.battery == test_b]
            if target == 'rul': te = te[te.cycle <= te.eol]
            m = models()[name].fit(tr[feats], tr[target])
            p = m.predict(te[feats])
            if target == 'soh': pass
            else: p = np.clip(p, 0, None)
            rows.append((test_b, mae(te[target], p), np.sqrt(mean_squared_error(te[target], p)), r2_score(te[target], p)))
        res[name] = pd.DataFrame(rows, columns=['bat','MAE','RMSE','R2']).set_index('bat')
    return res

if __name__ == '__main__':
    for target in ('soh','rul'):
        print(f'\n===== {target.upper()} (leave-one-battery-out) =====')
        r = evaluate(target, FEATS)
        for n, t in r.items():
            print(f'{n:11s} MAE={t.MAE.mean():.3f} RMSE={t.RMSE.mean():.3f} R2={t.R2.mean():.3f} | ' + ' '.join(f'{b}:{t.MAE[b]:.2f}' for b in t.index))
