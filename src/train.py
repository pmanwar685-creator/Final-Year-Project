"""Prepare battery-level SOH data and README-censored RUL labels."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


NOMINAL_CAPACITY = 2.0
MIN_VALID_CAPACITY = 0.5
MAX_VALID_CAPACITY = 2.5
MIN_RUL_CYCLES = 10
EOL_CONFIRMATION_CYCLES = 3
ENDPOINT_TOLERANCE_AH = 0.08

# Batteries with no README-stated EOL capacity, or whose tests stopped after a
# software crash, are intentionally omitted from supervised RUL labels.
EOL_CAPACITY = {
    **{f"B{number:04d}": 1.4 for number in (5, 6, 7, 18)},
    **{f"B{number:04d}": 1.6 for number in (33, 34, 36, 38, 39, 40)},
    **{f"B{number:04d}": 1.4 for number in range(41, 49)},
    **{f"B{number:04d}": 1.4 for number in range(53, 57)},
}

raw_df = pd.read_csv("results/features.csv")
raw_df["capacity"] = pd.to_numeric(raw_df["capacity"], errors="coerce")
raw_df = raw_df.sort_values(["battery", "cycle"]).reset_index(drop=True)

EXCLUDED = {"battery", "cycle", "capacity", "soh", "eol", "rul"}
FEATURES = [column for column in raw_df.columns if column not in EXCLUDED]

df = raw_df[
    raw_df["capacity"].between(MIN_VALID_CAPACITY, MAX_VALID_CAPACITY)
].dropna(subset=FEATURES).copy()
df["soh"] = df["capacity"] / NOMINAL_CAPACITY * 100
df = df.reset_index(drop=True)
FEATS = FEATURES


def _consecutive_run(group, predicate, start=0):
    capacity = group["capacity"].to_numpy()
    cycles = group["cycle"].to_numpy()
    width = EOL_CONFIRMATION_CYCLES
    for index in range(start, len(group) - width + 1):
        if not np.all(np.diff(cycles[index:index + width]) == 1):
            continue
        if np.all(predicate(capacity[index:index + width])):
            return index
    return None


def _rul_interval(battery, group):
    cutoff = EOL_CAPACITY.get(battery)
    if cutoff is None:
        return None

    valid = group[
        group["capacity"].between(MIN_VALID_CAPACITY, MAX_VALID_CAPACITY)
    ].sort_values("cycle").reset_index(drop=True)
    start_index = _consecutive_run(valid, lambda values: values > cutoff)
    if start_index is None:
        return None

    end_index = _consecutive_run(
        valid, lambda values: values <= cutoff, start=start_index
    )
    if end_index is not None:
        eol_cycle = int(valid.loc[end_index + EOL_CONFIRMATION_CYCLES - 1, "cycle"])
    else:
        last = valid.iloc[-1]
        if (
            last["cycle"] != group["cycle"].max()
            or abs(last["capacity"] - cutoff) > ENDPOINT_TOLERANCE_AH
        ):
            return None
        eol_cycle = int(last["cycle"])

    start_cycle = int(valid.loc[start_index, "cycle"])
    if eol_cycle - start_cycle + 1 < MIN_RUL_CYCLES:
        return None
    return start_cycle, eol_cycle


eol = {}
rul_parts = []
for battery, group in raw_df.groupby("battery", sort=True):
    interval = _rul_interval(battery, group)
    if interval is None:
        continue
    start_cycle, end_cycle = interval
    eol[battery] = end_cycle
    labeled = df[
        (df["battery"] == battery)
        & df["cycle"].between(start_cycle, end_cycle)
    ].copy()
    labeled["feature_index"] = labeled.index
    labeled["eol"] = end_cycle
    labeled["rul"] = (end_cycle - labeled["cycle"]).clip(lower=0)
    if not labeled.empty:
        rul_parts.append(labeled)

rul_df = pd.concat(rul_parts, ignore_index=True) if rul_parts else pd.DataFrame()
RUL_BATTERIES = sorted(rul_df["battery"].unique()) if not rul_df.empty else []


def models():
    return {
        "Ridge": make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 20))
        ),
        "SVR": make_pipeline(
            StandardScaler(),
            SVR(C=10, epsilon=0.01, gamma="scale"),
        ),
        "ExtraTrees": ExtraTreesRegressor(
            300, min_samples_leaf=3, n_jobs=-1, random_state=0
        ),
        "GBR": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            random_state=0,
        ),
    }


if __name__ == "__main__":
    print(f"SOH rows: {len(df)} across {df.battery.nunique()} batteries")
    print(f"RUL rows: {len(rul_df)} across {len(RUL_BATTERIES)} complete batteries")
    print("RUL endpoints:", eol)
    print("Batteries excluded from RUL:", sorted(set(raw_df.battery) - set(RUL_BATTERIES)))
