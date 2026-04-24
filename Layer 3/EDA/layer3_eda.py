import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

TRAIN_DIR = "/Users/eng.luai/Desktop/ISE 619 Project/Layer 3/Layer 3 Training data -  XJTU/"
TEST_DIR  = "/Users/eng.luai/Desktop/ISE 619 Project/Layer 3/Layer 3 Testing data -  XJTU/"
OUT_DIR   = "/Users/eng.luai/Desktop/ISE 619 Project/Layer 3/"

# ── Load arrays ──────────────────────────────────────────────────────────────
xtr1 = np.load(os.path.join(TRAIN_DIR, "xtr2_1.npy"))
xtr2 = np.load(os.path.join(TRAIN_DIR, "xtr2_2.npy"))
ytr1 = np.load(os.path.join(TRAIN_DIR, "ytr2_1.npy"))
ytr2 = np.load(os.path.join(TRAIN_DIR, "ytr2_2.npy"))

boundaries = {}
for i in range(1, 6):
    boundaries[i] = np.load(os.path.join(TRAIN_DIR, f"xtr2_b{i}.npy"))

xte3 = np.load(os.path.join(TEST_DIR, "xte2_3.npy"))
xte4 = np.load(os.path.join(TEST_DIR, "xte2_4.npy"))
xte5 = np.load(os.path.join(TEST_DIR, "xte2_5.npy"))
yte3 = np.load(os.path.join(TEST_DIR, "yte2_3.npy"))
yte4 = np.load(os.path.join(TEST_DIR, "yte2_4.npy"))
yte5 = np.load(os.path.join(TEST_DIR, "yte2_5.npy"))

arrays = {
    "xtr2_1": xtr1, "xtr2_2": xtr2,
    "ytr2_1": ytr1, "ytr2_2": ytr2,
    **{f"xtr2_b{i}": boundaries[i] for i in range(1, 6)},
    "xte2_3": xte3, "xte2_4": xte4, "xte2_5": xte5,
    "yte2_3": yte3, "yte2_4": yte4, "yte2_5": yte5,
}

# ── 1. Shape inspection ───────────────────────────────────────────────────────
print("=" * 60)
print("1. SHAPE INSPECTION")
print("=" * 60)
for name, arr in arrays.items():
    print(f"  {name:12s}  shape={str(arr.shape):25s}  dtype={arr.dtype}")

print()
print("  Snapshots per bearing:")
for label, arr in [("Bearing 1 (train)", xtr1), ("Bearing 2 (train)", xtr2),
                   ("Bearing 3 (test)",  xte3),  ("Bearing 4 (test)",  xte4),
                   ("Bearing 5 (test)",  xte5)]:
    print(f"    {label}: {arr.shape[0]} snapshots")

# ── 2. NaN / Inf check ───────────────────────────────────────────────────────
print()
print("=" * 60)
print("2. NaN / INF CHECK")
print("=" * 60)
for name, arr in arrays.items():
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan == 0 and n_inf == 0:
        print(f"  {name:12s}  CLEAN")
    else:
        print(f"  {name:12s}  NaN={n_nan}  Inf={n_inf}")

# ── 3. RUL label summary ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("3. RUL LABEL SUMMARY")
print("=" * 60)
rul_map = {
    "Bearing 1 (train)": ytr1,
    "Bearing 2 (train)": ytr2,
    "Bearing 3 (test)":  yte3,
    "Bearing 4 (test)":  yte4,
    "Bearing 5 (test)":  yte5,
}
for label, y in rul_map.items():
    flat = y.flatten()
    print(f"  {label}")
    print(f"    Steps={len(flat):6d}  Min={flat.min():.2f}  Max={flat.max():.2f}  Mean={flat.mean():.2f}")

# ── helper: extract scalar boundary index ────────────────────────────────────
def scalar_boundary(b_arr):
    """Return the degradation boundary as a plain int regardless of array shape."""
    flat = b_arr.flatten()
    return int(flat[0])

# ── 4. Degradation trajectory plots ─────────────────────────────────────────
print()
print("=" * 60)
print("4. DEGRADATION TRAJECTORY PLOTS")
print("=" * 60)

bearing_data = [
    (1, ytr1,  True,  scalar_boundary(boundaries[1])),
    (2, ytr2,  True,  scalar_boundary(boundaries[2])),
    (3, yte3,  False, None),
    (4, yte4,  False, None),
    (5, yte5,  False, None),
]

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle("RUL Trajectories – Condition 2 (XJTU)", fontsize=13, fontweight="bold")

for ax, (b_num, y, is_train, boundary) in zip(axes, bearing_data):
    flat = y.flatten()
    ax.plot(flat, color="steelblue", linewidth=0.9)
    if is_train and boundary is not None:
        ax.axvline(x=boundary, color="red", linestyle="--", linewidth=1.2, label=f"Boundary={boundary}")
        ax.legend(fontsize=7)
    ax.set_title(f"Bearing {b_num} – Condition 2", fontsize=9)
    ax.set_xlabel("Snapshot index (min)", fontsize=8)
    ax.set_ylabel("RUL", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "layer3_eda_rul_trajectories.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"  Saved → {out_path}")

# ── 5. Vibration signal comparison ───────────────────────────────────────────
print()
print("=" * 60)
print("5. VIBRATION SIGNAL COMPARISON  (Bearing 1)")
print("=" * 60)

# Determine horizontal channel index
# Expected shape: (snapshots, samples, channels) or (snapshots, samples)
if xtr1.ndim == 3:
    # channel 0 is typically horizontal
    first_snap = xtr1[0, :, 0]
    last_snap  = xtr1[-1, :, 0]
    ch_label   = "Channel 0 (horizontal)"
elif xtr1.ndim == 2:
    first_snap = xtr1[0, :]
    last_snap  = xtr1[-1, :]
    ch_label   = "Signal"
else:
    first_snap = xtr1[0].flatten()
    last_snap  = xtr1[-1].flatten()
    ch_label   = "Signal"

print(f"  xtr1 shape: {xtr1.shape}  → using {ch_label}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Bearing 1 – Condition 2: Early vs Near-Failure Snapshot", fontsize=11, fontweight="bold")

ax1.plot(first_snap, color="steelblue", linewidth=0.6)
ax1.set_title("Snapshot 0  (early life)", fontsize=9)
ax1.set_xlabel("Sample index", fontsize=8)
ax1.set_ylabel("Acceleration", fontsize=8)
ax1.tick_params(labelsize=7)
ax1.grid(True, alpha=0.3)

ax2.plot(last_snap, color="tomato", linewidth=0.6)
ax2.set_title(f"Snapshot {xtr1.shape[0]-1}  (near failure)", fontsize=9)
ax2.set_xlabel("Sample index", fontsize=8)
ax2.set_ylabel("Acceleration", fontsize=8)
ax2.tick_params(labelsize=7)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "layer3_eda_signal_comparison.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"  Saved → {out_path}")

# ── 6. RUL distribution ───────────────────────────────────────────────────────
print()
print("=" * 60)
print("6. RUL DISTRIBUTION")
print("=" * 60)

all_rul = np.concatenate([y.flatten() for y in [ytr1, ytr2, yte3, yte4, yte5]])

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(all_rul, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
ax.set_title("RUL Distribution – All Bearings (Condition 2)", fontsize=11, fontweight="bold")
ax.set_xlabel("RUL (minutes)", fontsize=9)
ax.set_ylabel("Count", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "layer3_eda_rul_distribution.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"  Saved → {out_path}")
print(f"  Total RUL values: {len(all_rul):,}")
print(f"  Combined min={all_rul.min():.2f}  max={all_rul.max():.2f}  mean={all_rul.mean():.2f}")

print()
print("Layer 3 EDA complete")
