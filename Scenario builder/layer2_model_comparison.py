import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconf")

# Resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "Layer 2 - Motor 4 - XGBoost", "features_dataset.csv")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ── 1. Load & map ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

def map_label(lbl):
    s = str(lbl).strip().lower()
    if s == "normal":      return "Normal"
    if "align" in s:       return "Misalignment"
    if "unbalance" in s:   return "Unbalance"
    if "coupling" in s:    return "Coupling Failure"
    if "cavitation" in s:  return "Cavitation"
    return None

df["fault_class"] = df["label"].map(map_label)
df = df.dropna(subset=["fault_class"])

drop_cols = {"label", "fault_class", "motor", "folder"}
feature_cols = [c for c in df.columns if c not in drop_cols]
assert len(feature_cols) == 40, f"Expected 40 features, got {len(feature_cols)}"

X = df[feature_cols].values
y = df["fault_class"].values

# ── 2. Same 80/20 stratified split as Layer 2 ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

class_names = sorted(np.unique(y))

# ── 3. Define models ──────────────────────────────────────────────────────────
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":           Pipeline([("scaler", StandardScaler()),
                                ("clf",    SVC(kernel="rbf", C=1.0, gamma="scale"))]),
    "KNN":           Pipeline([("scaler", StandardScaler()),
                                ("clf",    KNeighborsClassifier(n_neighbors=5))]),
    "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                      ("clf",    LogisticRegression(max_iter=1000,
                                                                     random_state=42))]),
}

# ── 4. Train, evaluate, collect results ──────────────────────────────────────
results   = {}   # name → {"accuracy": float, "f1_per_class": dict, "cm": array}
sep = "=" * 60

for name, model in models.items():
    print(f"\n{sep}\n{name}\n{sep}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=class_names,
                                   output_dict=True)
    print(classification_report(y_test, y_pred, target_names=class_names))

    f1_per_class = {cls: report[cls]["f1-score"] for cls in class_names}
    cm = confusion_matrix(y_test, y_pred, labels=class_names, normalize="true")

    results[name] = {"accuracy": acc, "f1_per_class": f1_per_class, "cm": cm}

# ── 5. 2×2 confusion matrix figure ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for ax, (name, res) in zip(axes, results.items()):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=res["cm"] * 100,
        display_labels=class_names
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".1f")
    ax.set_title(f"{name}  (acc = {res['accuracy']:.4f})", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=30)

fig.suptitle("Normalized Confusion Matrices — Layer 2 Model Comparison\n(% of true label)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "layer2_comparison_confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: layer2_comparison_confusion_matrices.png")

# ── 6. Accuracy bar chart ─────────────────────────────────────────────────────
model_names = list(results.keys())
accuracies  = [results[n]["accuracy"] for n in model_names]
colors      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(model_names, accuracies, color=colors, width=0.5, edgecolor="black", linewidth=0.6)
ax.set_ylim(min(accuracies) - 0.05, 1.02)
ax.set_ylabel("Overall Accuracy")
ax.set_title("Overall Accuracy — Layer 2 Model Comparison", fontweight="bold")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{acc:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "layer2_comparison_accuracy.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: layer2_comparison_accuracy.png")

# ── 7. Summary table ──────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY TABLE")
print(f"{'=' * 70}")

header_cls = "  ".join(f"{c:<18}" for c in class_names)
print(f"{'Model':<22}  {'Accuracy':>8}  {header_cls}")
print("-" * 70)

for name, res in results.items():
    f1s = "  ".join(f"{res['f1_per_class'][c]:<18.4f}" for c in class_names)
    print(f"{name:<22}  {res['accuracy']:>8.4f}  {f1s}")

print(f"\nClasses: {class_names}")
print("\nModel comparison complete — files saved to project root")
