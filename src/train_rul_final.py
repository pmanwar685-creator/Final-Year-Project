"""Clean RUL benchmark and final model using observed end-of-life labels only.

B0007 is right-censored at 1.4 Ah and is never used as a supervised RUL label.
The reported score is leave-one-battery-out across B0005, B0006 and B0018.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

EOL_CYCLE = {"B0005": 125, "B0006": 109, "B0018": 97}
RANDOM_STATE = 42


def model(name):
    settings = dict(n_estimators=700, min_samples_leaf=2, max_features=0.8, n_jobs=-1, random_state=RANDOM_STATE)
    if name == "ExtraTrees":
        return ExtraTreesRegressor(**settings)
    if name == "RandomForest":
        return RandomForestRegressor(**settings)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.03, loss="huber", random_state=RANDOM_STATE)
    raise ValueError(name)


def score(y, p):
    y, p = np.asarray(y), np.asarray(p)
    nonzero = y > 0
    return {
        "MAE_cycles": float(mean_absolute_error(y, p)),
        "RMSE_cycles": float(np.sqrt(mean_squared_error(y, p))),
        "MedianAE_cycles": float(np.median(np.abs(y - p))),
        "MaxAE_cycles": float(np.max(np.abs(y - p))),
        "R2": float(r2_score(y, p)),
        "Bias_cycles": float(np.mean(p - y)),
        "MAPE_nonzero_RUL_pct": float(np.mean(np.abs((p[nonzero] - y[nonzero]) / y[nonzero])) * 100),
    }


def data_and_features():
    data = pd.read_csv("results/features.csv")
    data = data[data.battery.isin(EOL_CYCLE)].copy()
    # Trailing means are causal: cycle k uses only measurements up to cycle k.
    raw = [c for c in data.columns if c not in {"battery", "capacity", "cycle"}]
    for column in raw:
        data[f"{column}_ma5"] = data.groupby("battery")[column].transform(
            lambda values: values.rolling(5, min_periods=1).mean()
        )
    data["eol"] = data.battery.map(EOL_CYCLE)
    data = data[data.cycle <= data.eol].copy()
    data["rul"] = data.eol - data.cycle
    features = [c for c in data.columns if c not in {"battery", "capacity", "eol", "rul"}]
    if data[features].isna().any().any():
        raise ValueError("Unexpected missing feature values")
    return data, features


def main():
    data, features = data_and_features()
    cells, names = list(EOL_CYCLE), ("ExtraTrees", "RandomForest", "GradientBoosting")
    output = {
        "protocol": "LOBO across observed-EOL cells; held-out battery excluded from every fit",
        "eol_threshold_ah": 1.4,
        "included_cells": cells,
        "right_censored_excluded_cell": "B0007",
        "n_labelled_cycles": int(len(data)),
        "n_features": len(features),
        "models": {},
    }
    for name in names:
        folds = {}
        for held_out in cells:
            train, test = data[data.battery != held_out], data[data.battery == held_out]
            fitted = model(name).fit(train[features], train.rul)
            prediction = np.clip(fitted.predict(test[features]), 0, None)
            folds[held_out] = score(test.rul, prediction)
        metric_names = list(next(iter(folds.values())))
        output["models"][name] = {"folds": folds, "mean": {
            metric: float(np.mean([folds[cell][metric] for cell in cells])) for metric in metric_names
        }}
    best = min(names, key=lambda name: output["models"][name]["mean"]["MAE_cycles"])
    final = model(best).fit(data[features], data.rul)
    os.makedirs("results/models", exist_ok=True)
    joblib.dump({"model": final, "model_name": best, "features": features,
                 "eol_threshold_ah": 1.4, "training_cells": cells,
                 "right_censored_excluded_cell": "B0007"},
                "results/models/rul_observed_eol_model.joblib")
    output["selected_model"] = best
    output["selection_note"] = "Lowest mean MAE among three predeclared models; add independent cells for a final external test."
    with open("results/rul_observed_eol_results.json", "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
