"""
scenario_builder.py
Build 4 scenario CSV files (520 columns each) for pipeline presentation.

Columns 0–39:  NLN-EMP statistical features (from features_dataset.csv)
Columns 40–519: XJTU-SY bearing vibration features (30 snapshots × 16 stats)

Notes on actual data layout vs original spec
─────────────────────────────────────────────
• features_dataset.csv / layer2_model.pkl live in  "Layer 2 - Motor 4 - XGBoost/"
  (not the project root).  feature_cols are derived directly from the CSV so that
  loading layer2_model.pkl (which requires xgboost) is optional.
• "Coupling Failure" label does not exist in the dataset; "Coupling Degradation"
  is used instead.
• xte2_3.npy has shape (533, 32768, 2) — raw vibration (32 768 time samples,
  2 channels) per snapshot, not (533, 16).  This script extracts 8 time-domain
  statistics per channel → 16 features per snapshot, then takes a 30-snapshot
  consecutive window → 30 × 16 = 480 values.  This matches the expected block
  size and mirrors the same feature family used in the NLN-EMP layer.
• The XJTU testing-data directory name contains two spaces before "XJTU".
"""

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)          # one level up → project root
LAYER1_PKL   = os.path.join(PROJECT_ROOT, "Layer 1 - Motor 4", "layer1_model.pkl")
LAYER2_DIR   = os.path.join(PROJECT_ROOT, "Layer 2 - Motor 4 - XGBoost")
FEATURES_CSV = os.path.join(LAYER2_DIR, "features_dataset.csv")
MODEL_PKL    = os.path.join(LAYER2_DIR, "layer2_model.pkl")

XJTU_DIR  = os.path.join(PROJECT_ROOT, "Layer 3", "Layer 3 Testing data -  XJTU")
XTE_PATH  = os.path.join(XJTU_DIR, "xte2_3.npy")
YTE_PATH  = os.path.join(XJTU_DIR, "yte2_3.npy")

# ── Scenario definitions ───────────────────────────────────────────────────────
# l1_expected: +1 = normal (pick highest decision_function),
#              -1 = anomaly (pick lowest decision_function)
SCENARIOS = [
    dict(name="scenario_healthy",          label="Normal",               xjtu_start=5,   l1_expected=1),
    dict(name="scenario_anomaly_early",    label="Misalignment",         xjtu_start=160, l1_expected=-1),
    dict(name="scenario_fault_critical",   label="Cavitation",           xjtu_start=490, l1_expected=-1),
    dict(name="scenario_imminent_failure", label="Coupling Degradation", xjtu_start=500, l1_expected=-1),
]

WINDOW_LEN = 30   # consecutive snapshots per scenario
N_XJTU     = WINDOW_LEN * 16   # 480


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — NLN-EMP BLOCK
# ══════════════════════════════════════════════════════════════════════════════

print("Loading features_dataset.csv …")
df = pd.read_csv(FEATURES_CSV)

# Derive feature_cols from the CSV (same result as loading from pkl bundle,
# but avoids the xgboost dependency).
feature_cols = [c for c in df.columns if c not in ("label", "motor", "folder")]
assert len(feature_cols) == 40, f"Expected 40 feature cols, got {len(feature_cols)}"

# Optionally load feature_cols from the pkl bundle (requires xgboost installed).
try:
    bundle = joblib.load(MODEL_PKL)
    if isinstance(bundle, dict) and "feature_cols" in bundle:
        feature_cols = bundle["feature_cols"]
        print(f"  feature_cols loaded from pkl bundle ({len(feature_cols)} cols)")
    else:
        print("  pkl loaded but no 'feature_cols' key — using CSV-derived cols")
except Exception as exc:
    print(f"  pkl load skipped ({exc.__class__.__name__}: {exc})")
    print("  Using CSV-derived feature_cols instead")

print(f"  feature_cols ({len(feature_cols)}): {feature_cols[:5]} … {feature_cols[-3:]}")
print()

# ── Reproduce Layer 2 80/20 stratified split → use test split only ────────────
from sklearn.model_selection import train_test_split

_, X_test, _, y_test = train_test_split(
    df[feature_cols], df['label'],
    test_size=0.2, stratify=df['label'], random_state=42
)

# Rebuild a test-only dataframe with the label column re-attached
test_df = X_test.copy()
test_df['label'] = y_test

print(f"Test split: {len(test_df)} rows (20% of {len(df)})")
print("  Rows per class in test split:")
for cls, cnt in sorted(test_df['label'].value_counts().items()):
    print(f"    {cls:<25} {cnt}")
print()

# ── Layer 1 — run predictions on test rows only ───────────────────────────────
print("Loading Layer 1 model and scoring test rows …")
l1_pipeline = joblib.load(LAYER1_PKL)

X_test_vals = test_df[feature_cols].values
l1_preds    = l1_pipeline.predict(X_test_vals)           # +1 or -1
l1_scores   = l1_pipeline.decision_function(X_test_vals) # higher → more normal

test_df = test_df.copy()
test_df['_l1_pred']  = l1_preds
test_df['_l1_score'] = l1_scores

normal_pct  = (l1_preds == 1).mean() * 100
anomaly_pct = (l1_preds == -1).mean() * 100
print(f"  {len(test_df)} test rows scored — normal: {normal_pct:.1f}%  anomaly: {anomaly_pct:.1f}%")
print()

# ── Pick the most-confident row for each scenario ─────────────────────────────
# For normal  (+1): highest  decision_function score (farthest from boundary, safe side)
# For anomaly (-1): lowest   decision_function score (most confidently anomalous)
nln_rows  = {}
l1_info   = {}          # (pred, score) per scenario name

for sc in SCENARIOS:
    lbl      = sc["label"]
    expected = sc["l1_expected"]

    label_df = test_df[test_df["label"] == lbl]
    if label_df.empty:
        raise ValueError(f"Label '{lbl}' not found in test split. "
                         f"Available: {test_df['label'].unique().tolist()}")

    # Prefer rows where Layer 1 already agrees with expected class
    correct_df = label_df[label_df["_l1_pred"] == expected]
    pool = correct_df if not correct_df.empty else label_df

    if expected == 1:
        idx = pool["_l1_score"].idxmax()   # most confidently normal
    else:
        idx = pool["_l1_score"].idxmin()   # most confidently anomalous

    row   = test_df.loc[idx, feature_cols].values.astype(float)
    pred  = int(test_df.loc[idx, "_l1_pred"])
    score = float(test_df.loc[idx, "_l1_score"])

    nln_rows[sc["name"]] = row
    l1_info[sc["name"]]  = (pred, score)

    pred_label = "NORMAL" if pred == 1 else "ANOMALY"
    print(f"  NLN-EMP  [{sc['name']}]  label='{lbl}'  "
          f"L1={pred_label} (score={score:.4f})  "
          f"sample values: {row[:3].round(4)} …")

print()


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — XJTU-SY BLOCK
# ══════════════════════════════════════════════════════════════════════════════

print("Loading XJTU numpy arrays …")
xte = np.load(XTE_PATH)   # (533, 32768, 2)
yte = np.load(YTE_PATH)   # (533, 1)
print(f"  xte2_3 shape: {xte.shape}")
print(f"  yte2_3 shape: {yte.shape}")
print()


def extract_16_features(snapshot: np.ndarray) -> np.ndarray:
    """
    Extract 16 time-domain statistics from one raw bearing snapshot.

    snapshot : ndarray shape (T, 2)  — T time samples, 2 vibration channels
    returns  : ndarray shape (16,)   — 8 stats × 2 channels
    """
    feats = []
    for ch in range(snapshot.shape[1]):
        sig = snapshot[:, ch].astype(float)
        rms          = np.sqrt(np.mean(sig ** 2))
        peak         = np.max(np.abs(sig))
        peak_to_peak = np.max(sig) - np.min(sig)
        crest_factor = peak / (rms + 1e-12)
        from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
        kurt     = scipy_kurtosis(sig, bias=False)
        skewness = scipy_skew(sig, bias=False)
        std      = np.std(sig, ddof=1)
        mean_abs = np.mean(np.abs(sig))
        feats.extend([rms, peak, peak_to_peak, crest_factor, kurt, skewness, std, mean_abs])
    return np.array(feats, dtype=float)


def extract_window(xte_array: np.ndarray, yte_array: np.ndarray, start: int) -> tuple:
    """
    Extract a 30-snapshot window, compute per-snapshot features, flatten.

    Returns (flat_array shape (480,), mean_rul float)
    """
    window_x = xte_array[start: start + WINDOW_LEN]   # (30, 32768, 2)
    window_y = yte_array[start: start + WINDOW_LEN]   # (30, 1)
    mean_rul = float(window_y.mean())

    feature_rows = np.stack([extract_16_features(window_x[i])
                             for i in range(WINDOW_LEN)])   # (30, 16)
    flat = feature_rows.flatten()                           # (480,)
    assert flat.shape == (N_XJTU,), f"Expected ({N_XJTU},) got {flat.shape}"
    return flat, mean_rul


print("Extracting XJTU windows …")
xjtu_rows = {}
rul_values = {}
for sc in SCENARIOS:
    start = sc["xjtu_start"]
    flat, mean_rul = extract_window(xte, yte, start)
    xjtu_rows[sc["name"]] = flat
    rul_values[sc["name"]] = mean_rul
    print(f"  XJTU  [{sc['name']}]  start={start}  "
          f"mean RUL={mean_rul:.2f}  flat shape={flat.shape}")

print()


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — COMBINE AND SAVE
# ══════════════════════════════════════════════════════════════════════════════

xjtu_cols = [f"xjtu_{i}" for i in range(N_XJTU)]
all_cols   = list(feature_cols) + xjtu_cols   # 40 + 480 = 520

print("Saving scenario CSV files …")
saved_shapes = {}
for sc in SCENARIOS:
    name  = sc["name"]
    row   = np.concatenate([nln_rows[name], xjtu_rows[name]])   # (520,)
    out   = pd.DataFrame([row], columns=all_cols)
    path  = os.path.join(PROJECT_ROOT, f"{name}.csv")
    out.to_csv(path, index=False)
    saved_shapes[name] = out.shape
    print(f"  Saved: {path}  shape={out.shape}")

print()


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

header = (
    f"{'Scenario':<26}| {'NLN-EMP Label':<22}| {'L1 Pred':^9}| "
    f"{'L1 Score':^10}| {'XJTU Start':^11}| {'Mean RUL':^10}| {'CSV Shape'}"
)
sep = (
    "-" * 26 + "+" + "-" * 23 + "+" + "-" * 10 + "+"
    + "-" * 11 + "+" + "-" * 12 + "+" + "-" * 11 + "+" + "-" * 10
)
print(header)
print(sep)
for sc in SCENARIOS:
    name  = sc["name"]
    label = sc["label"]
    start = sc["xjtu_start"]
    rul   = rul_values[name]
    shape = saved_shapes[name]
    pred, score = l1_info[name]
    pred_str = "NORMAL" if pred == 1 else "ANOMALY"
    print(
        f"{name:<26}| {label:<22}| {pred_str:^9}| "
        f"{score:^10.4f}| {start:^11}| {rul:^10.1f}| {shape}"
    )

print()
print("Scenario builder complete — 4 files saved to project root")
