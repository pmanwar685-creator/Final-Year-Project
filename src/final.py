"""Train and evaluate SOH/RUL models on the extended NASA battery dataset."""
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train as T


warnings.filterwarnings("ignore")
df = T.df
rul_df = T.rul_df
soh_batteries = sorted(df["battery"].unique())
rul_batteries = T.RUL_BATTERIES
RUL_FEATURES = ["gap", "slope20", "slope40", "slope_all"]

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/models", exist_ok=True)


def soh_model():
    return make_pipeline(
        StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 20))
    )


def rul_model():
    return GradientBoostingRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        random_state=0,
    )


def trajectory_features(soh, cycles):
    smoothed = pd.Series(soh).ewm(alpha=0.3, adjust=False).mean().to_numpy()
    cycles = np.asarray(cycles)
    slopes = {}
    for window in (20, 40):
        values = np.zeros(len(smoothed))
        for index in range(len(smoothed)):
            start = max(0, index - window + 1)
            if index - start >= 4:
                x = cycles[start:index + 1]
                values[index] = np.polyfit(x, smoothed[start:index + 1], 1)[0]
        slopes[f"slope{window}"] = values

    baseline = float(np.mean(smoothed[:min(5, len(smoothed))]))
    return pd.DataFrame({
        "cycle": cycles,
        "soh_s": smoothed,
        "gap": smoothed - 70,
        "slope20": slopes["slope20"],
        "slope40": slopes["slope40"],
        "slope_all": -(baseline - smoothed) / np.maximum(cycles, 1),
    })


def metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def evaluate_soh_lobo():
    predictions = {}
    results = {}
    for battery in soh_batteries:
        train = df[df["battery"] != battery]
        test = df[df["battery"] == battery]
        model = soh_model().fit(train[T.FEATS], train["soh"])
        prediction = model.predict(test[T.FEATS])
        predictions[battery] = pd.Series(prediction, index=test.index)
        results[battery] = metrics(test["soh"], prediction)
    return predictions, results


def lobo_rul_predictions(test_battery):
    trajectories = {}
    for battery in rul_batteries:
        rows = rul_df[rul_df["battery"] == battery]
        feature_rows = df.loc[rows["feature_index"].to_numpy()]
        if battery == test_battery:
            train = df[df["battery"] != test_battery]
        else:
            train = df[~df["battery"].isin([test_battery, battery])]
        model = soh_model().fit(train[T.FEATS], train["soh"])
        predicted_soh = model.predict(feature_rows[T.FEATS])
        trajectory = trajectory_features(predicted_soh, rows["cycle"])
        trajectory["rul"] = rows["rul"].to_numpy()
        trajectories[battery] = trajectory

    results = {}
    for battery in rul_batteries:
        train = pd.concat(
            [trajectories[other] for other in rul_batteries if other != battery],
            ignore_index=True,
        )
        test = trajectories[battery]
        model = rul_model().fit(train[RUL_FEATURES], np.log1p(train["rul"]))
        prediction = np.clip(np.expm1(model.predict(test[RUL_FEATURES])), 0, None)
        results[battery] = metrics(test["rul"], prediction)
    return trajectories, results


def evaluate_chronological():
    max_cycle = df.groupby("battery")["cycle"].transform("max")
    training_mask = df["cycle"] <= (max_cycle * 0.6).astype(int)
    model = soh_model().fit(df[training_mask][T.FEATS], df[training_mask]["soh"])
    soh_predictions = model.predict(df[T.FEATS])
    predictions = pd.Series(soh_predictions, index=df.index)

    soh_results = {}
    for battery in soh_batteries:
        test = df[(df["battery"] == battery) & ~training_mask]
        if test.empty:
            continue
        soh_results[battery] = metrics(
            test["soh"], predictions.loc[test.index]
        )

    trajectories = {}
    training_parts = []
    for battery in rul_batteries:
        rows = rul_df[rul_df["battery"] == battery]
        trajectory = trajectory_features(
            predictions.loc[rows["feature_index"].to_numpy()], rows["cycle"]
        )
        trajectory["rul"] = rows["rul"].to_numpy()
        split_cycle = int(rows["cycle"].max() * 0.6)
        train_rows = trajectory[trajectory["cycle"] <= split_cycle]
        test_rows = trajectory[trajectory["cycle"] > split_cycle]
        if not train_rows.empty:
            training_parts.append(train_rows)
        trajectories[battery] = (trajectory, test_rows)

    if not training_parts:
        raise ValueError("No RUL rows remain for chronological training.")
    training = pd.concat(training_parts, ignore_index=True)
    model = rul_model().fit(training[RUL_FEATURES], np.log1p(training["rul"]))
    rul_results = {}
    for battery, (_, test) in trajectories.items():
        if test.empty:
            continue
        prediction = np.clip(np.expm1(model.predict(test[RUL_FEATURES])), 0, None)
        rul_results[battery] = metrics(test["rul"], prediction)
    return soh_results, rul_results


def save_metric_plots(path, title, values, label):
    columns = min(4, max(1, len(values)))
    rows = int(np.ceil(len(values) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.5 * columns, 2.8 * rows), squeeze=False
    )
    for axis, (battery, metric) in zip(axes.flat, values.items()):
        axis.bar(["MAE", "RMSE"], [metric["MAE"], metric["RMSE"]])
        axis.set_title(f"{battery} {label}")
    for axis in axes.flat[len(values):]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def mean_metrics(values):
    return {
        key: float(np.mean([metric[key] for metric in values.values()]))
        for key in ("MAE", "RMSE", "R2")
    } if values else None


def main():
    if len(soh_batteries) < 2:
        raise ValueError("At least two batteries with valid capacities are required.")
    if len(rul_batteries) < 2:
        raise ValueError("At least two complete, README-labeled batteries are required for RUL.")

    _, soh_lobo = evaluate_soh_lobo()
    _, rul_lobo = lobo_rul_predictions(rul_batteries[0])

    chronological_soh, chronological_rul = evaluate_chronological()
    results = {
        "LOBO": {
            "SOH": soh_lobo,
            "RUL": rul_lobo,
            "mean_SOH": mean_metrics(soh_lobo),
            "mean_RUL": mean_metrics(rul_lobo),
        },
        "CHRONO": {
            "SOH": chronological_soh,
            "RUL": chronological_rul,
            "mean_SOH": mean_metrics(chronological_soh),
            "mean_RUL": mean_metrics(chronological_rul),
        },
        "data": {
            "soh_batteries": soh_batteries,
            "rul_batteries": rul_batteries,
            "rul_eol_cycles": T.eol,
            "rul_eol_capacity_ah": {
                battery: T.EOL_CAPACITY[battery] for battery in rul_batteries
            },
            "rul_excluded_batteries": sorted(
                set(df["battery"]) - set(rul_batteries)
            ),
            "minimum_capacity_ah": T.MIN_VALID_CAPACITY,
            "maximum_capacity_ah": T.MAX_VALID_CAPACITY,
        },
    }
    with open("results/results.json", "w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)

    save_metric_plots(
        "results/plots/lobo_soh.png", "Leave-one-battery-out SOH MAE/RMSE",
        soh_lobo, "SOH",
    )
    save_metric_plots(
        "results/plots/lobo_rul.png", "Leave-one-battery-out RUL MAE/RMSE",
        rul_lobo, "RUL",
    )
    save_metric_plots(
        "results/plots/chrono_soh.png", "Chronological SOH MAE/RMSE",
        chronological_soh, "SOH",
    )
    save_metric_plots(
        "results/plots/chrono_rul.png", "Chronological RUL MAE/RMSE",
        chronological_rul, "RUL",
    )

    final_soh = soh_model().fit(df[T.FEATS], df["soh"])
    oof_trajectories = []
    for battery in rul_batteries:
        rows = rul_df[rul_df["battery"] == battery]
        feature_rows = df.loc[rows["feature_index"].to_numpy()]
        training = df[df["battery"] != battery]
        predicted_soh = soh_model().fit(
            training[T.FEATS], training["soh"]
        ).predict(feature_rows[T.FEATS])
        trajectory = trajectory_features(predicted_soh, rows["cycle"])
        trajectory["rul"] = rows["rul"].to_numpy()
        oof_trajectories.append(trajectory)
    rul_training = pd.concat(oof_trajectories, ignore_index=True)
    final_rul = rul_model().fit(
        rul_training[RUL_FEATURES], np.log1p(rul_training["rul"])
    )
    joblib.dump(
        {
            "soh": final_soh,
            "rul": final_rul,
            "soh_features": T.FEATS,
            "rul_features": RUL_FEATURES,
            "nominal_capacity_ah": T.NOMINAL_CAPACITY,
            "rul_eol_capacity_ah": results["data"]["rul_eol_capacity_ah"],
            "rul_batteries": rul_batteries,
        },
        "results/models/soh_rul_models.joblib",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
