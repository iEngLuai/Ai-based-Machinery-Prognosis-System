import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

bundle      = joblib.load(os.path.join(BASE_DIR, 'Layer 2 - Motor 4 - XGBoost', 'layer2_model.pkl'))
l2_model    = bundle['model']
l2_encoder  = bundle['label_encoder']
feature_cols = bundle['feature_cols']

l1_bundle   = joblib.load(os.path.join(BASE_DIR, 'Layer 1 - Motor 4', 'layer1_model.pkl'))
l1_pipeline = l1_bundle

l3_model    = load_model(os.path.join(BASE_DIR, 'Layer 3', 'Layer 3 - Load 2 Results', 'layer3_model.h5'), compile=False)
l3_scaler   = joblib.load(os.path.join(BASE_DIR, 'Layer 3', 'Layer 3 - Load 2 Results', 'layer3_scaler.pkl'))

print("All models loaded successfully.")


def run_pipeline(csv_path):
    df = pd.read_csv(csv_path)

    X_l1l2  = df[feature_cols].values                  # shape (1, 40)
    X_l3_raw = df.iloc[:, 40:520].values               # shape (1, 480)

    # ── Layer 1 — Anomaly Detection ──────────────────────────────────────────
    l1_result = l1_pipeline.predict(X_l1l2)[0]         # +1 or -1
    l1_score  = l1_pipeline.decision_function(X_l1l2)[0]

    if l1_result == 1:
        print("══════════════════════════════════════════════")
        print(" PUMP HEALTH ASSESSMENT REPORT")
        print("══════════════════════════════════════════════")
        print(f" Layer 1 — Anomaly Detection   : NORMAL")
        print(f" Anomaly Score                 : {l1_score:.4f}")
        print(f" Layer 2 — Fault Classification: NOT TRIGGERED")
        print(f" Layer 3 — RUL Prediction      : NOT TRIGGERED")
        print("──────────────────────────────────────────────")
        print(" MAINTENANCE DECISION: ✅ HEALTHY")
        print(" No anomaly detected. Continue normal operation.")
        print("══════════════════════════════════════════════")
        return

    # ── Layer 2 — Fault Classification ───────────────────────────────────────
    pred_int    = l2_model.predict(X_l1l2)
    fault_class = l2_encoder.inverse_transform(pred_int)[0]
    proba       = l2_model.predict_proba(X_l1l2)[0]
    confidence  = proba[pred_int[0]] * 100

    # ── Layer 3 — RUL Prediction ──────────────────────────────────────────────
    X_l3_scaled  = l3_scaler.transform(X_l3_raw.reshape(30, 16))
    X_l3_lstm    = X_l3_scaled.reshape(1, 30, 16)
    rul_normalized = l3_model.predict(X_l3_lstm)[0][0]
    rul_score    = float(np.clip(rul_normalized, 0, 1)) * 100

    if rul_score > 60:
        tier        = "HEALTHY"
        color_label = "🟢"
    elif rul_score >= 20:
        tier        = "MONITOR"
        color_label = "🟡"
    else:
        tier        = "REPLACE"
        color_label = "🔴"

    print("══════════════════════════════════════════════")
    print(" PUMP HEALTH ASSESSMENT REPORT")
    print("══════════════════════════════════════════════")
    print(f" Layer 1 — Anomaly Detection   : ⚠️  ANOMALY DETECTED")
    print(f" Anomaly Score                 : {l1_score:.4f}")
    print("──────────────────────────────────────────────")
    print(f" Layer 2 — Fault Classification: {fault_class}")
    print(f" Confidence                    : {confidence:.2f}%")
    print("──────────────────────────────────────────────")
    print(f" Layer 3 — RUL Score           : {rul_score:.1f} / 100")
    print(f" Maintenance Tier              : {color_label} {tier}")
    print("──────────────────────────────────────────────")
    print(" MAINTENANCE DECISION:")
    print()
    if tier == "HEALTHY":
        print(" Equipment fault identified but degradation is early.")
        print(" Schedule inspection at next planned maintenance window.")
    elif tier == "MONITOR":
        print(" Fault progressing. Prepare spare parts and schedule")
        print(" maintenance within the next maintenance cycle.")
    else:
        print(" Critical degradation detected. Immediate maintenance")
        print(" required. Do not defer.")
    print("══════════════════════════════════════════════")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_path = os.path.join(BASE_DIR, sys.argv[1])
        print(f"─── Running: {sys.argv[1]} ───")
        run_pipeline(csv_path)
    else:
        scenarios = [
            'scenario_healthy.csv',
            'scenario_anomaly_early.csv',
            'scenario_fault_critical.csv',
            'scenario_imminent_failure.csv',
        ]
        for scenario in scenarios:
            print(f"─── Running: {scenario} ───")
            run_pipeline(os.path.join(BASE_DIR, scenario))
            print()
