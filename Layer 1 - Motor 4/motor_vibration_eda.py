"""
Motor Vibration EDA Script
Analyzes vibration data for Motor 2 and Motor 4 across multiple fault types.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MOTOR_CONFIGS = {
    2: {
        "path": "/Users/eng.luai/Desktop/ISE 619 Project/MOTOR 2/",
        "file_pattern": "Vibration_Motor-2_75_time-{folder}-ch{ch}.csv",
    },
    4: {
        "path": "/Users/eng.luai/Desktop/ISE 619 Project/MOTOR 4 - 70/",
        "file_pattern": "Vibration_Motor-4_70_time-{folder}-ch{ch}.csv",
    },
}

CHANNELS = [1, 2, 3, 4, 5]
SAMPLING_RATE = 20_000   # Hz
WINDOW_DURATION = 12     # seconds

PLOT_OUTPUT = "/Users/eng.luai/Desktop/ISE 619 Project/eda_signal_comparison.png"

# Six subplots: (motor_number, folder_name)
SUBPLOT_CASES = [
    (2, "healthy 1"),
    (2, "bearing bpfo 3"),
    (2, "impeller 3"),
    (4, "healthy 1"),
    (4, "cavitation discharge 5"),
    (4, "align angular 5"),
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def build_filename(motor: int, folder: str, ch: int) -> str:
    pattern = MOTOR_CONFIGS[motor]["file_pattern"]
    return pattern.format(folder=folder, ch=ch)


def load_csv(filepath: str):
    """Load a CSV and return (DataFrame, error_message)."""
    if not os.path.exists(filepath):
        return None, f"File not found: {filepath}"
    if os.path.getsize(filepath) == 0:
        return None, f"Empty file: {filepath}"
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None, f"CSV has no data rows: {filepath}"
        return df, None
    except Exception as exc:
        return None, f"Read error ({exc}): {filepath}"


def measurement_columns(df: pd.DataFrame):
    """Return all column names that are measurement windows (not 'time')."""
    return [c for c in df.columns if c != "time"]


def scan_motor(motor: int) -> dict:
    """
    Scan all subfolders for a given motor.
    Returns a dict keyed by folder name with scan results.
    """
    base_path = MOTOR_CONFIGS[motor]["path"]
    results = {}

    if not os.path.isdir(base_path):
        print(f"  [WARNING] Motor {motor} base path does not exist: {base_path}")
        return results

    subfolders = sorted(
        [
            d
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        ]
    )

    for folder in subfolders:
        entry = {
            "folder": folder,
            "motor": motor,
            "csv_files_found": 0,
            "measurement_windows": None,
            "has_nan": False,
            "corrupted_files": [],
        }

        for ch in CHANNELS:
            fname = build_filename(motor, folder, ch)
            fpath = os.path.join(base_path, folder, fname)
            df, err = load_csv(fpath)

            if err:
                entry["corrupted_files"].append((fname, err))
                continue

            entry["csv_files_found"] += 1

            if ch == 1:
                # Derive measurement windows from channel 1
                mcols = measurement_columns(df)
                entry["measurement_windows"] = len(mcols)

                # Check for NaN values across all measurement columns
                nan_found = df[mcols].isnull().any().any()
                if "time" in df.columns:
                    nan_found = nan_found or df["time"].isnull().any()
                entry["has_nan"] = bool(nan_found)

        results[folder] = entry

    return results


# ─────────────────────────────────────────────
# Main scanning logic
# ─────────────────────────────────────────────
def print_scan_results(all_results: dict):
    total_samples = {2: 0, 4: 0}

    for motor in [2, 4]:
        print("=" * 60)
        print(f"MOTOR {motor}")
        print("=" * 60)

        motor_results = all_results.get(motor, {})
        if not motor_results:
            print("  No subfolders found or base path missing.\n")
            continue

        for folder, entry in sorted(motor_results.items()):
            windows = entry["measurement_windows"]
            windows_str = str(windows) if windows is not None else "N/A (ch1 unavailable)"
            nan_str = "YES ⚠" if entry["has_nan"] else "no"
            corrupted = entry["corrupted_files"]

            print(f"\n  Folder            : {folder}")
            print(f"  Motor             : {entry['motor']}")
            print(f"  CSV files found   : {entry['csv_files_found']} / {len(CHANNELS)}")
            print(f"  Measurement windows (from ch1): {windows_str}")
            print(f"  NaN values present: {nan_str}")

            if corrupted:
                print(f"  ⚠  Flagged files ({len(corrupted)}):")
                for fname, reason in corrupted:
                    print(f"       • {fname} — {reason}")

            # Accumulate sample count:
            #   windows × samples_per_window × channels found
            if windows is not None and entry["csv_files_found"] > 0:
                samples = (
                    windows
                    * WINDOW_DURATION
                    * SAMPLING_RATE
                    * entry["csv_files_found"]
                )
                total_samples[motor] += samples

        print()

    print("=" * 60)
    print("TOTAL SAMPLE COUNTS")
    print("=" * 60)
    for motor in [2, 4]:
        print(f"  Motor {motor}: {total_samples[motor]:,} samples")
    print()


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def load_first_window(motor: int, folder: str):
    """
    Load ch1 CSV for the given motor/folder and return
    (time_array, signal_array) for the first measurement window.
    Returns (None, None) if unavailable.
    """
    base_path = MOTOR_CONFIGS[motor]["path"]
    fname = build_filename(motor, folder, ch=1)
    fpath = os.path.join(base_path, folder, fname)

    df, err = load_csv(fpath)
    if err:
        print(f"  [PLOT WARNING] Cannot load data for Motor {motor} / '{folder}': {err}")
        return None, None

    mcols = measurement_columns(df)
    if not mcols:
        print(f"  [PLOT WARNING] No measurement columns in Motor {motor} / '{folder}'.")
        return None, None

    first_col = mcols[0]
    signal = df[first_col].values

    if "time" in df.columns:
        time = df["time"].values
    else:
        time = np.arange(len(signal)) / SAMPLING_RATE

    return time, signal


def create_comparison_plot(all_results: dict):
    print("Creating signal comparison figure …")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        "Vibration Signal Comparison — First Measurement Window (ch1)",
        fontsize=14,
        fontweight="bold",
    )
    axes_flat = axes.flatten()

    for idx, (motor, folder) in enumerate(SUBPLOT_CASES):
        ax = axes_flat[idx]
        time, signal = load_first_window(motor, folder)

        if time is None or signal is None:
            ax.text(
                0.5,
                0.5,
                f"Data unavailable\n(Motor {motor} / {folder})",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="red",
                fontsize=10,
            )
            ax.set_title(f"Motor {motor} — {folder}", fontsize=10, color="red")
        else:
            ax.plot(time, signal, linewidth=0.5, color="steelblue")
            ax.set_title(f"Motor {motor} — {folder}", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Acceleration (g)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Ensure output directory exists
    out_dir = os.path.dirname(PLOT_OUTPUT)
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(PLOT_OUTPUT, dpi=150)
    print(f"  Plot saved to: {PLOT_OUTPUT}")
    plt.close(fig)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  MOTOR VIBRATION — EXPLORATORY DATA ANALYSIS")
    print("=" * 60 + "\n")

    all_results = {}
    for motor in [2, 4]:
        print(f"Scanning Motor {motor} …")
        all_results[motor] = scan_motor(motor)
        print(f"  Found {len(all_results[motor])} subfolder(s).\n")

    print_scan_results(all_results)
    create_comparison_plot(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
