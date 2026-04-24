import os
import numpy as np
import joblib
from scipy.stats import kurtosis, skew

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─── Seeds ────────────────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
TRAIN_PATH  = "/Users/eng.luai/Desktop/ISE 619 Project/Layer 3/Layer 3 training Load 1/"
TEST_PATH   = "/Users/eng.luai/Desktop/ISE 619 Project/Layer 3/Layer 3 testing Load 1/"
OUTPUT_PATH = "/Users/eng.luai/Desktop/ISE 619 Project/Layer 3/"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — LOAD DATA")
print("=" * 60)

xtr1_b1 = np.load(os.path.join(TRAIN_PATH, "xtr1_1.npy"))
xtr1_b2 = np.load(os.path.join(TRAIN_PATH, "xtr1_2.npy"))
ytr1_b1 = np.load(os.path.join(TRAIN_PATH, "ytr1_1.npy")).flatten()
ytr1_b2 = np.load(os.path.join(TRAIN_PATH, "ytr1_2.npy")).flatten()

xte3 = np.load(os.path.join(TEST_PATH, "xte1_3.npy"))
xte4 = np.load(os.path.join(TEST_PATH, "xte1_4.npy"))
xte5 = np.load(os.path.join(TEST_PATH, "xte1_5.npy"))
yte3 = np.load(os.path.join(TEST_PATH, "yte1_3.npy")).flatten()
yte4 = np.load(os.path.join(TEST_PATH, "yte1_4.npy")).flatten()
yte5 = np.load(os.path.join(TEST_PATH, "yte1_5.npy")).flatten()

print(f"  xtr1_1: {xtr1_b1.shape}   ytr1_1: {ytr1_b1.shape}")
print(f"  xtr1_2: {xtr1_b2.shape}   ytr1_2: {ytr1_b2.shape}")
print(f"  xte1_3: {xte3.shape}   yte1_3: {yte3.shape}")
print(f"  xte1_4: {xte4.shape}   yte1_4: {yte4.shape}")
print(f"  xte1_5: {xte5.shape}   yte1_5: {yte5.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 2 — FEATURE EXTRACTION")
print("=" * 60)

def extract_features(X):
    """
    X : (n_snapshots, n_samples, 2)
    Returns : (n_snapshots, 16)  — 8 features × 2 channels
    """
    n = X.shape[0]
    feats = np.zeros((n, 16), dtype=np.float32)
    for i in range(n):
        for ch in range(2):
            sig = X[i, :, ch].astype(np.float64)
            rms   = np.sqrt(np.mean(sig ** 2))
            peak  = np.max(np.abs(sig))
            p2p   = np.max(sig) - np.min(sig)
            crest = peak / (rms + 1e-10)
            kurt  = kurtosis(sig, fisher=True)
            sk    = skew(sig)
            std   = np.std(sig)
            mav   = np.mean(np.abs(sig))
            base  = ch * 8
            feats[i, base:base + 8] = [rms, peak, p2p, crest, kurt, sk, std, mav]
    return feats

print("  Extracting features — this may take a few minutes …")
F_tr1 = extract_features(xtr1_b1); print(f"  Bearing 1 (train): {F_tr1.shape}")
F_tr2 = extract_features(xtr1_b2); print(f"  Bearing 2 (train): {F_tr2.shape}")
F_te3 = extract_features(xte3);    print(f"  Bearing 3 (test):  {F_te3.shape}")
F_te4 = extract_features(xte4);    print(f"  Bearing 4 (test):  {F_te4.shape}")
F_te5 = extract_features(xte5);    print(f"  Bearing 5 (test):  {F_te5.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SLIDING WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 3 — SLIDING WINDOW  (window=30, stride=1)")
print("=" * 60)

WINDOW = 30
STRIDE = 1

def sliding_windows(F, y, window=30, stride=1):
    """Return X (n_windows, window, 16) and y (n_windows,)."""
    X_wins, y_wins = [], []
    for start in range(0, len(F) - window + 1, stride):
        end = start + window
        X_wins.append(F[start:end])
        y_wins.append(y[end - 1])
    return np.array(X_wins, dtype=np.float32), np.array(y_wins, dtype=np.float32)

Xw_tr1, yw_tr1 = sliding_windows(F_tr1, ytr1_b1, WINDOW, STRIDE)
Xw_tr2, yw_tr2 = sliding_windows(F_tr2, ytr1_b2, WINDOW, STRIDE)

X_test3, y_test3 = sliding_windows(F_te3, yte3, WINDOW, STRIDE)
X_test4, y_test4 = sliding_windows(F_te4, yte4, WINDOW, STRIDE)
X_test5, y_test5 = sliding_windows(F_te5, yte5, WINDOW, STRIDE)

X_train = np.vstack([Xw_tr1, Xw_tr2])
y_train = np.concatenate([yw_tr1, yw_tr2])

print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
print(f"  X_test3: {X_test3.shape}   y_test3: {y_test3.shape}")
print(f"  X_test4: {X_test4.shape}   y_test4: {y_test4.shape}")
print(f"  X_test5: {X_test5.shape}   y_test5: {y_test5.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — NORMALIZE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 4 — NORMALIZE FEATURES")
print("=" * 60)

n_tr, w, f = X_train.shape
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, f)).reshape(n_tr, w, f)

def scale_test(X):
    n, ww, ff = X.shape
    return scaler.transform(X.reshape(-1, ff)).reshape(n, ww, ff)

X_test3_s = scale_test(X_test3)
X_test4_s = scale_test(X_test4)
X_test5_s = scale_test(X_test5)

scaler_path = os.path.join(OUTPUT_PATH, "layer3_scaler_load1.pkl")
joblib.dump(scaler, scaler_path)
print(f"  Scaler saved → {scaler_path}")

y_train_s = y_train / 100.0
y_test3_s = y_test3 / 100.0
y_test4_s = y_test4 / 100.0
y_test5_s = y_test5 / 100.0

print(f"  RUL scaled to [0, 1]. Example y_train range: [{y_train_s.min():.3f}, {y_train_s.max():.3f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — BUILD LSTM MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 5 — BUILD LSTM MODEL")
print("=" * 60)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, f)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1,  activation="linear"),
], name="Layer3_LSTM_Load1")

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"],
)
model.summary()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 6 — TRAIN")
print("=" * 60)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                     patience=7, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_train_scaled, y_train_s,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    shuffle=True,
    callbacks=callbacks,
    verbose=1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 7 — EVALUATE")
print("=" * 60)

test_sets = [
    (3, X_test3_s, y_test3),
    (4, X_test4_s, y_test4),
    (5, X_test5_s, y_test5),
]

all_rmse, all_mae = [], []
predictions = {}

for b_num, X_s, y_true in test_sets:
    y_pred_scaled = model.predict(X_s, verbose=0).flatten()
    y_pred = np.clip(y_pred_scaled * 100.0, 0, 100)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    all_rmse.append(rmse)
    all_mae.append(mae)
    predictions[b_num] = (y_pred, y_true)
    print(f"  Bearing {b_num} — RMSE: {rmse:.4f}   MAE: {mae:.4f}")

overall_rmse = np.mean(all_rmse)
overall_mae  = np.mean(all_mae)
print(f"\n  Overall RMSE (avg across bearings 3-5): {overall_rmse:.4f}")
print(f"  Overall MAE  (avg across bearings 3-5): {overall_mae:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — MAINTENANCE TIER MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 8 — MAINTENANCE TIER MAPPING")
print("=" * 60)

def get_maintenance_tier(rul):
    if rul > 60:
        return "HEALTHY", "green"
    elif rul >= 20:
        return "MONITOR", "orange"
    else:
        return "REPLACE", "red"

tier_counts = {"HEALTHY": 0, "MONITOR": 0, "REPLACE": 0}
all_tiers = {}

for b_num, (y_pred, y_true) in predictions.items():
    tiers = [get_maintenance_tier(r)[0] for r in y_pred]
    all_tiers[b_num] = tiers
    for t in tiers:
        tier_counts[t] += 1

    print(f"\n  Bearing {b_num} — Sample predictions (5 steps):")
    print(f"  {'Step':>6}  {'Pred RUL':>9}  {'Actual RUL':>10}  {'Tier'}")
    print(f"  {'-'*45}")
    indices = np.linspace(0, len(y_pred) - 1, 5, dtype=int)
    for idx in indices:
        tier, _ = get_maintenance_tier(y_pred[idx])
        print(f"  {idx:>6}  {y_pred[idx]:>9.2f}  {y_true[idx]:>10.2f}  {tier}")

print(f"\n  Tier distribution across all test bearings:")
for tier, count in tier_counts.items():
    print(f"    {tier:8s}: {count}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 9 — PLOTS")
print("=" * 60)

# Plot 1 — Training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Layer 3 LSTM — Training History (Load 1 / Condition 1)",
             fontsize=12, fontweight="bold")

ax1.plot(history.history["loss"],     label="Train Loss",  color="steelblue")
ax1.plot(history.history["val_loss"], label="Val Loss",    color="tomato", linestyle="--")
ax1.set_title("Loss (MSE)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history["mae"],     label="Train MAE",   color="steelblue")
ax2.plot(history.history["val_mae"], label="Val MAE",     color="tomato", linestyle="--")
ax2.set_title("MAE")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MAE")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
p1 = os.path.join(OUTPUT_PATH, "layer3_training_history_load1.png")
plt.savefig(p1, dpi=150)
plt.close()
print(f"  Plot 1 saved → {p1}")

# Plot 2 — Predicted vs Actual RUL
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle("Layer 3 LSTM — Predicted vs Actual RUL (Load 1 / Condition 1)",
             fontsize=12, fontweight="bold")

for ax, b_num in zip(axes, [3, 4, 5]):
    y_pred, y_true = predictions[b_num]
    ax.plot(y_true, color="steelblue",  linewidth=1.0, label="Actual RUL")
    ax.plot(y_pred, color="tomato",     linewidth=1.0, linestyle="--", label="Predicted RUL")
    ax.axhline(60, color="green", linestyle="--", linewidth=0.8, alpha=0.7, label="RUL=60")
    ax.axhline(20, color="red",   linestyle="--", linewidth=0.8, alpha=0.7, label="RUL=20")
    ax.set_title(f"Bearing {b_num} – Condition 1", fontsize=9)
    ax.set_xlabel("Snapshot index", fontsize=8)
    ax.set_ylabel("RUL (0–100)", fontsize=8)
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
p2 = os.path.join(OUTPUT_PATH, "layer3_rul_predictions_load1.png")
plt.savefig(p2, dpi=150)
plt.close()
print(f"  Plot 2 saved → {p2}")

# Plot 3 — Tier distribution
tiers_ordered = ["HEALTHY", "MONITOR", "REPLACE"]
counts = [tier_counts[t] for t in tiers_ordered]
colors = ["green", "orange", "red"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(tiers_ordered, counts, color=colors, edgecolor="white", linewidth=0.5)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Maintenance Tier Distribution — Load 1, Test Bearings 3, 4, 5",
             fontsize=10, fontweight="bold")
ax.set_ylabel("Number of Predictions")
ax.set_xlabel("Tier")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
p3 = os.path.join(OUTPUT_PATH, "layer3_tier_distribution_load1.png")
plt.savefig(p3, dpi=150)
plt.close()
print(f"  Plot 3 saved → {p3}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10 — SAVE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("STEP 10 — SAVE MODEL")
print("=" * 60)

model_path = os.path.join(OUTPUT_PATH, "layer3_model_load1.h5")
model.save(model_path)
print(f"  Model saved → {model_path}")

# ─── Final summary ────────────────────────────────────────────────────────────
rmse3, mae3 = all_rmse[0], all_mae[0]
rmse4, mae4 = all_rmse[1], all_mae[1]
rmse5, mae5 = all_rmse[2], all_mae[2]

print()
print("=" * 60)
print("  LAYER 3 — CONDITION 1 RESULTS (35 Hz / 12 kN)")
print("=" * 60)
print(f"Bearing 3  →  RMSE: {rmse3:.2f}  |  MAE: {mae3:.2f}")
print(f"Bearing 4  →  RMSE: {rmse4:.2f}  |  MAE: {mae4:.2f}")
print(f"Bearing 5  →  RMSE: {rmse5:.2f}  |  MAE: {mae5:.2f}")
print("-" * 60)
print(f"OVERALL    →  RMSE: {overall_rmse:.2f}  |  MAE: {overall_mae:.2f}")
print("=" * 60)
print("Tier counts:")
print(f"  HEALTHY : {tier_counts['HEALTHY']}")
print(f"  MONITOR : {tier_counts['MONITOR']}")
print(f"  REPLACE : {tier_counts['REPLACE']}")
print("=" * 60)
print()
print("Layer 3 Load 1 complete.")
