"""Extract per-discharge-cycle features from all NASA battery MAT files."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    ROOT,
    ROOT / "3. BatteryAgingARC_25-44",
    ROOT / "2. BatteryAgingARC_25_26_27_28_P1",
    ROOT / "4. BatteryAgingARC_45_46_47_48",
    ROOT / "5. BatteryAgingARC_49_50_51_52",
    ROOT / "6. BatteryAgingARC_53_54_55_56",
)


def battery_files():
    files = {}
    for folder in SOURCE_DIRS:
        for path in sorted(folder.glob("B*.mat")):
            files.setdefault(path.stem, path)
    if not files:
        raise FileNotFoundError("No NASA battery MAT files were found.")
    return files


def vector(data, key):
    values = np.asarray(data[key], dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError(f"MAT cycle is missing measurements for {key}")
    return values


def load(bid):
    path = battery_files()[bid]
    return np.atleast_1d(scipy.io.loadmat(path, simplify_cells=True)[bid]["cycle"])


def feats(bid):
    cycles = load(bid)
    rows = []
    last_charge = None
    discharge_number = 0

    for cycle in cycles:
        cycle_type = cycle["type"]
        data = cycle["data"]
        if cycle_type == "charge":
            last_charge = data
            continue
        if cycle_type != "discharge":
            continue

        discharge_number += 1
        voltage = vector(data, "Voltage_measured")
        current = vector(data, "Current_measured")
        temperature = vector(data, "Temperature_measured")
        time = vector(data, "Time")
        capacity_values = np.asarray(data.get("Capacity", []), dtype=float).reshape(-1)
        capacity = float(capacity_values[0]) if capacity_values.size else np.nan

        row = {
            "battery": bid,
            "cycle": discharge_number,
            "capacity": capacity,
            "ambient_temperature": float(cycle["ambient_temperature"]),
            "dis_time": time[-1],
            "dis_vmin": voltage.min(),
            "dis_vmean": voltage.mean(),
            "dis_tmax": temperature.max(),
            "dis_tmean": temperature.mean(),
            "dis_trise": temperature.max() - temperature[0],
            "dis_imean": current.mean(),
        }
        for level in (3.9, 3.7, 3.5, 3.3):
            reached = np.flatnonzero(voltage <= level)
            row[f"t_v{level}"] = time[reached[0]] if reached.size else time[-1]
        row["t_tpeak"] = time[np.argmax(temperature)]
        row["v_drop_100"] = voltage[0] - voltage[min(len(voltage) - 1, 20)]

        if last_charge is not None:
            charge_current = vector(last_charge, "Current_measured")
            charge_time = vector(last_charge, "Time")
            charge_temperature = vector(last_charge, "Temperature_measured")
            row["ch_time"] = charge_time[-1]
            row["ch_tmax"] = charge_temperature.max()
            cc_end = np.flatnonzero(charge_current < 1.45)
            row["ch_cc_time"] = charge_time[cc_end[0]] if cc_end.size else charge_time[-1]
            row["ch_cv_time"] = charge_time[-1] - row["ch_cc_time"]
            row["ch_ah"] = np.trapezoid(charge_current, charge_time) / 3600

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    files = battery_files()
    features = pd.concat([feats(battery) for battery in sorted(files)], ignore_index=True)
    os.makedirs(ROOT / "results", exist_ok=True)
    features.to_csv(ROOT / "results" / "features.csv", index=False)
    print(
        features.groupby("battery").agg(
            n=("cycle", "size"),
            c0=("capacity", "first"),
            cend=("capacity", "last"),
            cmin=("capacity", "min"),
        )
    )
    print("Missing values:")
    print(features.isna().sum()[lambda values: values > 0])
