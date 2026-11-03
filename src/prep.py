"""Extract per-discharge-cycle features from NASA .mat files -> features.csv"""
import os, numpy as np, pandas as pd, scipy.io

def load(bid):
    d = scipy.io.loadmat(f'{bid}.mat', simplify_cells=True)[bid]['cycle']
    return d

def feats(bid):
    cyc = load(bid)
    rows = []
    last_charge = None
    n = 0
    for c in cyc:
        t = c['type']
        d = c['data']
        if t == 'charge':
            last_charge = d
        elif t == 'discharge':
            n += 1
            v = np.atleast_1d(d['Voltage_measured']); cur = np.atleast_1d(d['Current_measured'])
            T = np.atleast_1d(d['Temperature_measured']); tm = np.atleast_1d(d['Time'])
            cap = float(np.atleast_1d(d['Capacity'])[0])
            r = dict(battery=bid, cycle=n, capacity=cap)
            r['dis_time'] = tm[-1]
            r['dis_vmin'] = v.min(); r['dis_vmean'] = v.mean()
            r['dis_tmax'] = T.max(); r['dis_tmean'] = T.mean(); r['dis_trise'] = T.max()-T[0]
            r['dis_imean'] = cur.mean()
            # time to reach voltage levels (3.8,3.6,3.4,3.2)
            for lv in (3.9, 3.7, 3.5, 3.3):
                idx = np.where(v <= lv)[0]
                r[f't_v{lv}'] = tm[idx[0]] if len(idx) else tm[-1]
            # time of temperature peak
            r['t_tpeak'] = tm[np.argmax(T)]
            # voltage slope in mid region
            r['v_drop_100'] = v[0]-v[min(len(v)-1, 20)]
            if last_charge is not None:
                cv = np.atleast_1d(last_charge['Voltage_measured']); ci = np.atleast_1d(last_charge['Current_measured'])
                ct = np.atleast_1d(last_charge['Time']); cT = np.atleast_1d(last_charge['Temperature_measured'])
                r['ch_time'] = ct[-1]
                r['ch_tmax'] = cT.max()
                # CC phase end: first time current drops below 1.4A (CC=1.5A)
                idx = np.where(ci < 1.45)[0]
                r['ch_cc_time'] = ct[idx[0]] if len(idx) else ct[-1]
                r['ch_cv_time'] = ct[-1]-r['ch_cc_time']
                r['ch_ah'] = np.trapz(ci, ct)/3600
            rows.append(r)
    return pd.DataFrame(rows)

if __name__ == '__main__':
    df = pd.concat([feats(b) for b in ['B0005','B0006','B0007','B0018']], ignore_index=True)
    os.makedirs('results', exist_ok=True); df.to_csv('results/features.csv', index=False)
    print(df.groupby('battery').agg(n=('cycle','size'), c0=('capacity','first'), cend=('capacity','last'), cmin=('capacity','min')))
    print(df.isna().sum()[lambda s: s>0])
