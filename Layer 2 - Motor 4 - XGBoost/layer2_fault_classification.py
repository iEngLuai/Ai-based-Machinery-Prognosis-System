import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconf")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("features_dataset.csv")

# ── 2. Map label → fault_class ───────────────────────────────────────────────
def map_label(lbl):
    s = str(lbl).strip().lower()
    if s == "normal":
        return "Normal"
    if "align" in s:
        return "Misalignment"
    if "unbalance" in s:
        return "Unbalance"
    if "coupling" in s:
        return "Coupling Failure"
    if "cavitation" in s:
        return "Cavitation"
    return None  # bearing, impeller, looseness, soft foot → dropped

df["fault_class"] = df["label"].map(map_label)

# Drop unmatched rows
before = len(df)
df = df.dropna(subset=["fault_class"])
after = len(df)
print(f"Rows before mapping: {before}  |  After dropping unmatched: {after}  |  Dropped: {before - after}")

# ── 3. Class distribution ─────────────────────────────────────────────────────
print("\nClass distribution:")
print(df["fault_class"].value_counts())

# ── 4. Features and target ───────────────────────────────────────────────────
drop_cols = {"label", "fault_class", "motor", "folder"}
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df["fault_class"]

print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

# ── 5. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ── 6. Encode labels ─────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(y_train)
y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)
class_names  = le.classes_

# ── 7. Train XGBoost ─────────────────────────────────────────────────────────
model = XGBClassifier(
    eval_metric="mlogloss",
    random_state=42,
)
model.fit(X_train, y_train_enc)

# ── 8. Evaluate ───────────────────────────────────────────────────────────────
y_pred_enc = model.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ── 9. Confusion matrix (normalised, %) ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=class_names, normalize="true")
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm * 100, display_labels=class_names)
disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".1f")
ax.set_title("Confusion Matrix (% of true label)")
plt.tight_layout()
plt.savefig("layer2_confusion_matrix.png", dpi=150)
plt.close()
print("Saved: layer2_confusion_matrix.png")

# ── 10. SHAP explanations ─────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Beeswarm summary plot — multi-class: shap_values is (n_samples, n_features, n_classes)
# shap.summary_plot expects (n_samples, n_features) for multi-output; use mean abs across classes
if isinstance(shap_values, list):
    shap_arr = np.array(shap_values)          # (n_classes, n_samples, n_features)
    shap_mean = np.abs(shap_arr).mean(axis=0) # (n_samples, n_features)
else:
    # newer shap returns (n_samples, n_features, n_classes)
    shap_mean = np.abs(shap_values).mean(axis=-1) if shap_values.ndim == 3 else shap_values

plt.figure()
shap.summary_plot(shap_mean, X_test, feature_names=feature_cols, show=False, plot_type="violin")
plt.tight_layout()
plt.savefig("layer2_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: layer2_shap_beeswarm.png")

# Top-10 features bar chart
mean_abs_shap = np.abs(shap_mean).mean(axis=0)
top10_idx     = np.argsort(mean_abs_shap)[::-1][:10]
top10_feats   = [feature_cols[i] for i in top10_idx]
top10_vals    = mean_abs_shap[top10_idx]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(top10_feats[::-1], top10_vals[::-1], color="steelblue")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Top 10 Features — Mean Absolute SHAP Value")
plt.tight_layout()
plt.savefig("layer2_shap_top10.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: layer2_shap_top10.png")

# ── 11. Save model ────────────────────────────────────────────────────────────
joblib.dump({"model": model, "label_encoder": le, "feature_cols": feature_cols},
            "layer2_model.pkl")
print("\nLayer 2 complete — model saved to layer2_model.pkl")
