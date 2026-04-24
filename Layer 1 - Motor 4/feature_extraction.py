#!/usr/bin/env python3
"""Extract statistical features from NLN-EMP vibration CSVs (Motor 2 and Motor 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


PROJECT_ROOT = Path("/Users/eng.luai/Desktop/ISE 619 Project")
MOTOR_2_ROOT = PROJECT_ROOT / "MOTOR 2"
MOTOR_4_ROOT = PROJECT_ROOT / "MOTOR 4 - 70"
OUTPUT_CSV = PROJECT_ROOT / "features_dataset.csv"

FEATURE_NAMES = (
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "std",
    "mean_abs",
)


def build_folder_to_label() -> dict[str, str]:
    m: dict[str, str] = {}
    for i in (1, 2, 3):
        m[f"healthy {i}"] = "Normal"
    for prefix in ("bearing bpfi", "bearing bpfo", "bearing pump"):
        for n in (1, 2, 3):
            m[f"{prefix} {n}"] = "Bearing Fault"
    m["bearing bsf"] = "Bearing Fault"
    m["bearing contaminated"] = "Bearing Fault"
    for i in (1, 2, 3):
        m[f"impeller {i}"] = "Impeller Damage"
    m["loose foot motor"] = "Looseness"
    m["loose foot pump"] = "Looseness"
    m["soft foot 1"] = "Soft Foot"
    m["soft foot 2"] = "Soft Foot"
    for i in (1, 2, 3, 4, 5):
        m[f"align angular {i}"] = "Misalignment"
    for i in (1, 2, 3, 4):
        m[f"align parallel {i}"] = "Misalignment"
        m[f"align combination {i}"] = "Misalignment"
    for i in (1, 3, 4):
        m[f"cavitation suction {i}"] = "Cavitation"
    for i in (1, 2, 3, 4, 5):
        m[f"cavitation discharge {i}"] = "Cavitation"
    for i in (1, 2, 3, 4, 5, 6):
        m[f"unbalance motor {i}"] = "Unbalance"
    for i in (1, 2, 3):
        m[f"unbalance pump {i}"] = "Unbalance"
    m["coupling 1"] = "Coupling Degradation"
    m["coupling 2"] = "Coupling Degradation"
    m["coupling 2D"] = "Coupling Degradation"
    m["coupling 3"] = "Coupling Degradation"
    return m


FOLDER_TO_LABEL = build_folder_to_label()


def window_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c != "time"]
    # Numeric window indices stored as strings in CSV headers
    def sort_key(x: str) -> int:
        try:
            return int(x)
        except ValueError:
            return -1

    cols_sorted = sorted(cols, key=sort_key)
    if any(sort_key(c) < 0 for c in cols_sorted):
        raise ValueError(f"Unexpected non-integer columns besides 'time': {df.columns.tolist()}")
    return cols_sorted


def compute_features_matrix(x: np.ndarray) -> dict[str, np.ndarray]:
    """Compute features for each column. x shape (n_time, n_windows)."""
    rms = np.sqrt(np.mean(x * x, axis=0))
    abs_x = np.abs(x)
    peak = np.max(abs_x, axis=0)
    p2p = np.max(x, axis=0) - np.min(x, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        crest = np.where(rms > 1e-15, peak / rms, np.nan)
    std = np.std(x, axis=0, ddof=1)
    mean_abs = np.mean(abs_x, axis=0)
    kurt = kurtosis(x, axis=0, bias=False)
    sk = skew(x, axis=0, bias=False)
    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": p2p,
        "crest_factor": crest,
        "kurtosis": kurt,
        "skewness": sk,
        "std": std,
        "mean_abs": mean_abs,
    }


def load_channel_array(csv_path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    win_cols = window_feature_columns(df)
    arr = df[win_cols].to_numpy(dtype=np.float64)
    return arr, win_cols


def process_condition_folder(
    motor_name: str,
    folder_path: Path,
    csv_template: str,
) -> pd.DataFrame:
    folder_name = folder_path.name
    if folder_name not in FOLDER_TO_LABEL:
        raise KeyError(f"No label mapping for folder: {folder_name!r}")

    label = FOLDER_TO_LABEL[folder_name]
    channel_paths = [
        folder_path / csv_template.format(folder=folder_name, ch=c) for c in range(1, 6)
    ]
    if not all(p.is_file() for p in channel_paths):
        missing = [str(p) for p in channel_paths if not p.is_file()]
        raise FileNotFoundError(f"Missing channel file(s) in {folder_path}: {missing}")

    arrays: list[np.ndarray] = []
    win_cols_ref: list[str] | None = None
    for p in channel_paths:
        arr, win_cols = load_channel_array(p)
        if win_cols_ref is None:
            win_cols_ref = win_cols
        elif win_cols != win_cols_ref:
            raise ValueError(f"Window columns mismatch: {p} vs first channel")
        arrays.append(arr)

    n_win = arrays[0].shape[1]
    per_channel = [compute_features_matrix(arr) for arr in arrays]

    feature_rows: list[dict[str, float | str]] = []
    for w_idx in range(n_win):
        row: dict[str, float | str] = {}
        for ch_i, feats in enumerate(per_channel, start=1):
            for fname in FEATURE_NAMES:
                row[f"ch{ch_i}_{fname}"] = float(feats[fname][w_idx])
        row["label"] = label
        row["motor"] = motor_name
        row["folder"] = folder_name
        feature_rows.append(row)

    return pd.DataFrame(feature_rows)


def main() -> None:
    motors = [
        ("Motor 2", MOTOR_2_ROOT, "Vibration_Motor-2_75_time-{folder}-ch{ch}.csv"),
        ("Motor 4", MOTOR_4_ROOT, "Vibration_Motor-4_70_time-{folder}-ch{ch}.csv"),
    ]

    all_parts: list[pd.DataFrame] = []
    for motor_name, root, template in motors:
        if not root.is_dir():
            raise FileNotFoundError(f"Motor root not found: {root}")
        for folder_path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not folder_path.is_dir():
                continue
            df_part = process_condition_folder(motor_name, folder_path, template)
            if not df_part.empty:
                all_parts.append(df_part)

    feat_cols = [f"ch{c}_{f}" for c in range(1, 6) for f in FEATURE_NAMES]
    out = pd.concat(all_parts, ignore_index=True)
    out = out[feat_cols + ["label", "motor", "folder"]]

    out.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Shape (rows, columns): {out.shape}")
    print("\nClass distribution (label):")
    print(out["label"].value_counts().sort_index().to_string())
    print("\nRows per motor:")
    print(out["motor"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
