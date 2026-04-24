#!/usr/bin/env python3
"""Layer 1: Isolation Forest anomaly detection on extracted vibration features.

Evaluates multiple contamination values and three training scenarios:
  A — Train on all healthy 1+2, test on all motors
  B — Train on Motor 2 healthy 1+2, test on Motor 2 only
  C — Train on Motor 4 healthy 1+2, test on Motor 4 only

Metrics: Accuracy, Precision, Recall (TPR), F1, AUC-ROC, TNR, FPR,
         Confusion Matrix, per-fault TPR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path("/Users/eng.luai/Desktop/ISE 619 Project")
FEATURES_CSV = PROJECT_ROOT / "features_dataset.csv"
FIGURE_PATH = PROJECT_ROOT / "layer1_results.png"
MODEL_PATH = PROJECT_ROOT / "layer1_model.pkl"

CONTAMINATION_VALUES = [0.01, 0.02, 0.03, 0.05, 0.10]
TNR_FLOOR = 85.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ScenarioData:
    name: str
    label: str
    X_train: np.ndarray
    X_test_normal: np.ndarray
    X_test_fault: np.ndarray
    fault_labels: pd.Series          # aligned with X_test_fault rows


@dataclass
class EvalResult:
    contamination: float
    # core sklearn binary metrics (normal=0, anomaly=1)
    accuracy: float
    precision: float
    recall: float                    # == TPR
    f1: float
    auc_roc: float
    tnr: float
    fpr: float
    fnr: float
    conf_matrix: np.ndarray          # [[TN, FP], [FN, TP]]
    tpr_by_type: dict[str, float]
    # raw scores and arrays for plotting
    score_normal: np.ndarray
    score_fault: np.ndarray
    threshold: float
    fpr_curve: np.ndarray            # for ROC plot
    tpr_curve: np.ndarray
    pipeline: Pipeline
    n_train: int
    n_test_normal: int
    n_test_fault: int


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def build_pipeline(contamination: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(contamination=contamination, random_state=42)),
        ]
    )


def evaluate(
    pipeline: Pipeline,
    X_test_normal: np.ndarray,
    X_test_fault: np.ndarray,
    fault_labels: pd.Series,
    contamination: float,
    n_train: int,
) -> EvalResult:
    """Fit assumed done. Returns full EvalResult."""
    # IsolationForest: predict 1 (inlier/normal), -1 (outlier/anomaly)
    pred_normal = pipeline.predict(X_test_normal)   # expect mostly +1
    pred_fault  = pipeline.predict(X_test_fault)    # expect mostly -1
    score_normal = pipeline.decision_function(X_test_normal)  # higher = more normal
    score_fault  = pipeline.decision_function(X_test_fault)
    threshold = float(pipeline.named_steps["iforest"].offset_)

    # Recode to binary: normal=0, anomaly=1
    y_true  = np.concatenate([np.zeros(len(pred_normal)), np.ones(len(pred_fault))])
    y_pred  = np.concatenate([
        (pred_normal == -1).astype(int),
        (pred_fault  == -1).astype(int),
    ])
    # decision_function: higher is more normal → negate for "anomaly score"
    scores_all = np.concatenate([-score_normal, -score_fault])

    accuracy  = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))
    auc_roc   = float(roc_auc_score(y_true, scores_all))

    cm = confusion_matrix(y_true, y_pred)          # [[TN,FP],[FN,TP]]
    tn, fp, fn, tp = cm.ravel()

    tnr = 100.0 * tn / (tn + fp) if (tn + fp) else float("nan")
    fpr = 100.0 * fp / (tn + fp) if (tn + fp) else float("nan")
    fnr = 100.0 * fn / (fn + tp) if (fn + tp) else float("nan")

    fpr_curve, tpr_curve, _ = roc_curve(y_true, scores_all)

    # per-fault TPR
    fa = fault_labels.reset_index(drop=True)
    tpr_by_type: dict[str, float] = {}
    for ft in sorted(fa.unique()):
        m = fa == ft
        n_ft = int(m.sum())
        tpr_by_type[ft] = (
            100.0 * float(np.sum(pred_fault[m.to_numpy()] == -1)) / n_ft
            if n_ft else float("nan")
        )

    return EvalResult(
        contamination=contamination,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
        tnr=tnr,
        fpr=fpr,
        fnr=fnr,
        conf_matrix=cm,
        tpr_by_type=tpr_by_type,
        score_normal=score_normal,
        score_fault=score_fault,
        threshold=threshold,
        fpr_curve=fpr_curve,
        tpr_curve=tpr_curve,
        pipeline=pipeline,
        n_train=n_train,
        n_test_normal=len(pred_normal),
        n_test_fault=len(pred_fault),
    )


def sweep_contamination(sd: ScenarioData) -> list[EvalResult]:
    results: list[EvalResult] = []
    for c in CONTAMINATION_VALUES:
        pipe = build_pipeline(c)
        pipe.fit(sd.X_train)
        results.append(
            evaluate(pipe, sd.X_test_normal, sd.X_test_fault, sd.fault_labels, c, len(sd.X_train))
        )
    return results


def pick_best(results: list[EvalResult]) -> EvalResult:
    """Highest TPR (Recall) among those with TNR >= TNR_FLOOR; fallback to highest TPR."""
    candidates = [r for r in results if r.tnr >= TNR_FLOOR]
    pool = candidates if candidates else results
    return max(pool, key=lambda r: r.recall)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_contamination_table(scenario_name: str, results: list[EvalResult]) -> None:
    best = pick_best(results)
    print(f"\n{'='*78}")
    print(f"  Contamination sweep — {scenario_name}")
    print(f"{'='*78}")
    print(f"  {'Contam':>7}  {'Acc%':>6}  {'Prec%':>6}  {'Rec%':>6}  "
          f"{'F1%':>6}  {'AUC':>6}  {'TNR%':>6}  {'FPR%':>6}  {'FNR%':>6}")
    print(f"  {'-'*74}")
    for r in results:
        marker = " *" if r is best else ""
        print(
            f"  {r.contamination:>7.2f}  "
            f"{r.accuracy*100:>6.2f}  "
            f"{r.precision*100:>6.2f}  "
            f"{r.recall*100:>6.2f}  "
            f"{r.f1*100:>6.2f}  "
            f"{r.auc_roc:>6.4f}  "
            f"{r.tnr:>6.2f}  "
            f"{r.fpr:>6.2f}  "
            f"{r.fnr:>6.2f}{marker}"
        )
    print("  * = selected best (highest Recall with TNR >= 85%, else highest Recall)")


def print_best_detail(scenario_name: str, r: EvalResult) -> None:
    print(f"\n--- {scenario_name} best (contamination={r.contamination}) ---")
    print(f"  Train samples        : {r.n_train}")
    print(f"  Test normal          : {r.n_test_normal}")
    print(f"  Test fault           : {r.n_test_fault}")
    print(f"  Accuracy             : {r.accuracy*100:.2f}%")
    print(f"  Precision (anomaly)  : {r.precision*100:.2f}%")
    print(f"  Recall / TPR         : {r.recall*100:.2f}%")
    print(f"  F1 Score             : {r.f1*100:.2f}%")
    print(f"  AUC-ROC              : {r.auc_roc:.4f}")
    print(f"  True Negative Rate   : {r.tnr:.2f}%")
    print(f"  False Positive Rate  : {r.fpr:.2f}%")
    print(f"  False Negative Rate  : {r.fnr:.2f}%")
    tn, fp, fn, tp = r.conf_matrix.ravel()
    print(f"  Confusion Matrix     :")
    print(f"                          Predicted Normal  Predicted Anomaly")
    print(f"    Actual Normal      :  {tn:>16}  {fp:>17}")
    print(f"    Actual Anomaly     :  {fn:>16}  {tp:>17}")
    print(f"  Decision threshold   : {r.threshold:.6f}")
    print(f"  TPR by fault type:")
    for ft, v in sorted(r.tpr_by_type.items()):
        print(f"    {ft:<30} {v:6.2f}%")


def print_comparison_table(scenario_names: list[str], bests: list[EvalResult]) -> None:
    print(f"\n{'='*78}")
    print("  FINAL COMPARISON TABLE (best contamination per scenario)")
    print(f"{'='*78}")
    metrics = [
        ("Contamination",    lambda r: f"{r.contamination:.2f}"),
        ("Accuracy (%)",     lambda r: f"{r.accuracy*100:.2f}"),
        ("Precision (%)",    lambda r: f"{r.precision*100:.2f}"),
        ("Recall/TPR (%)",   lambda r: f"{r.recall*100:.2f}"),
        ("F1 Score (%)",     lambda r: f"{r.f1*100:.2f}"),
        ("AUC-ROC",          lambda r: f"{r.auc_roc:.4f}"),
        ("TNR (%)",          lambda r: f"{r.tnr:.2f}"),
        ("FPR (%)",          lambda r: f"{r.fpr:.2f}"),
        ("FNR (%)",          lambda r: f"{r.fnr:.2f}"),
        ("Train samples",    lambda r: str(r.n_train)),
        ("Test normal",      lambda r: str(r.n_test_normal)),
        ("Test fault",       lambda r: str(r.n_test_fault)),
    ]
    col_w = 22
    header = f"  {'Metric':<24}" + "".join(f"{n:>{col_w}}" for n in scenario_names)
    print(header)
    print(f"  {'-'*74}")
    for name, fn in metrics:
        row = f"  {name:<24}" + "".join(f"{fn(r):>{col_w}}" for r in bests)
        print(row)
    print(f"{'='*78}")


# ---------------------------------------------------------------------------
# Plots: 3 rows × 3 columns
# ---------------------------------------------------------------------------

def make_plots(
    scenario_names: list[str],
    bests: list[EvalResult],
) -> None:
    n_rows = len(bests)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5 * n_rows))
    fig.suptitle("Layer 1 — Isolation Forest results per scenario", fontsize=13, fontweight="bold")

    for row, (sname, r) in enumerate(zip(scenario_names, bests)):
        ax_hist, ax_bar, ax_roc = axes[row]

        # ── column 1: score histogram ──────────────────────────────────────
        all_scores = np.concatenate([r.score_normal, r.score_fault])
        rng = (float(np.min(all_scores)), float(np.max(all_scores)))
        ax_hist.hist(r.score_normal, bins=40, range=rng, alpha=0.55,
                     color="green", density=True, label="Normal (healthy 3)")
        ax_hist.hist(r.score_fault,  bins=40, range=rng, alpha=0.55,
                     color="red",   density=True, label="Fault (non-Normal)")
        ax_hist.axvline(r.threshold, color="black", linestyle="--", linewidth=1.4,
                        label=f"Threshold ({r.threshold:.3f})")
        ax_hist.set_xlabel("Decision score (higher = more normal)")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(f"{sname}  |  Score distributions\n"
                          f"contam={r.contamination}  TNR={r.tnr:.1f}%  TPR={r.recall*100:.1f}%")
        ax_hist.legend(fontsize=7)

        # ── column 2: TPR per fault type bar chart ─────────────────────────
        fault_types = sorted(r.tpr_by_type.keys())
        x_pos = np.arange(len(fault_types))
        bars  = [r.tpr_by_type[ft] for ft in fault_types]
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(fault_types)))
        ax_bar.bar(x_pos, bars, color=colors, edgecolor="black", linewidth=0.5)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(fault_types, rotation=38, ha="right", fontsize=7)
        ax_bar.set_ylabel("True Positive Rate (%)")
        ax_bar.set_ylim(0, 115)
        ax_bar.axhline(100, color="gray", linestyle=":", linewidth=0.8)
        ax_bar.set_title(f"{sname}  |  TPR per fault type")
        for xi, val in zip(x_pos, bars):
            ax_bar.text(xi, val + 2, f"{val:.0f}", ha="center", va="bottom", fontsize=6.5)

        # ── column 3: ROC curve ────────────────────────────────────────────
        ax_roc.plot(r.fpr_curve, r.tpr_curve, color="darkorange", lw=1.8,
                    label=f"AUC = {r.auc_roc:.3f}")
        ax_roc.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"{sname}  |  ROC Curve  (AUC={r.auc_roc:.3f})")
        ax_roc.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_scenarios(df: pd.DataFrame, feature_cols: list[str]) -> list[ScenarioData]:
    def Xn(mask: pd.Series) -> np.ndarray:
        return df.loc[mask, feature_cols].to_numpy(dtype=np.float64)

    def labels(mask: pd.Series) -> pd.Series:
        return df.loc[mask, "label"].reset_index(drop=True)

    h12     = df["folder"].isin(["healthy 1", "healthy 2"])
    h3      = df["folder"] == "healthy 3"
    fault   = df["label"] != "Normal"
    m2      = df["motor"] == "Motor 2"
    m4      = df["motor"] == "Motor 4"

    return [
        ScenarioData(
            name="Scenario A",
            label="All motors — train: all healthy 1+2 | test: all healthy 3 vs all faults",
            X_train=Xn(h12),
            X_test_normal=Xn(h3),
            X_test_fault=Xn(fault),
            fault_labels=labels(fault),
        ),
        ScenarioData(
            name="Scenario B",
            label="Motor 2 only — train: M2 healthy 1+2 | test: M2 healthy 3 vs M2 faults",
            X_train=Xn(h12 & m2),
            X_test_normal=Xn(h3 & m2),
            X_test_fault=Xn(fault & m2),
            fault_labels=labels(fault & m2),
        ),
        ScenarioData(
            name="Scenario C",
            label="Motor 4 only — train: M4 healthy 1+2 | test: M4 healthy 3 vs M4 faults",
            X_train=Xn(h12 & m4),
            X_test_normal=Xn(h3 & m4),
            X_test_fault=Xn(fault & m4),
            fault_labels=labels(fault & m4),
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(FEATURES_CSV)
    meta = {"label", "motor", "folder"}
    feature_cols = [c for c in df.columns if c not in meta]
    if len(feature_cols) != 40:
        raise ValueError(f"Expected 40 feature columns, found {len(feature_cols)}")

    scenarios = build_scenarios(df, feature_cols)
    all_bests: list[EvalResult] = []

    for sd in scenarios:
        results = sweep_contamination(sd)
        print_contamination_table(sd.name, results)
        best = pick_best(results)
        all_bests.append(best)
        print_best_detail(sd.name, best)

    scenario_names = [sd.name for sd in scenarios]
    print_comparison_table(scenario_names, all_bests)

    # Overall winner: prefer TNR-constrained, then highest F1
    winner_idx = max(
        range(len(all_bests)),
        key=lambda i: (all_bests[i].tnr >= TNR_FLOOR, all_bests[i].f1),
    )
    winner_sd   = scenarios[winner_idx]
    winner_best = all_bests[winner_idx]

    print(f"\n>>> Overall best scenario: {winner_sd.name} "
          f"(contamination={winner_best.contamination}, "
          f"F1={winner_best.f1*100:.2f}%, TNR={winner_best.tnr:.1f}%, "
          f"AUC={winner_best.auc_roc:.4f})")

    make_plots(scenario_names, all_bests)

    joblib.dump(winner_best.pipeline, MODEL_PATH)
    print(f"Model saved : {MODEL_PATH}")
    print(f"  Pipeline  : StandardScaler + IsolationForest "
          f"(contamination={winner_best.contamination})")


if __name__ == "__main__":
    main()
