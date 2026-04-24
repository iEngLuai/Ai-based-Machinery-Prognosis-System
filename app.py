import os
import re
import sys
import tempfile
import subprocess
import streamlit as st

st.set_page_config(
    page_title="Pump Health Assessment",
    page_icon="⚙️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_SCRIPT = os.path.join(BASE_DIR, "pipeline.py")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Student Info")
    st.markdown("**Name:** Luai Al Ibrahim")
    st.markdown("**Student ID:** 2017 500 90")
    st.markdown("**Course:** ISE 619")
    st.markdown("**Advisor:** Dr. Ahmad Al-Hanbali")
    st.markdown("**University:** KFUPM")
    st.markdown("---")
    st.markdown("**Three-Layer AI Framework:**")
    st.markdown("- **Layer 1:** Isolation Forest — Anomaly Detection")
    st.markdown("- **Layer 2:** XGBoost + SHAP — Fault Classification")
    st.markdown("- **Layer 3:** LSTM — Remaining Useful Life Prediction")
    st.markdown("---")
    st.markdown("**Dataset 1:** NLN-EMP Centrifugal Pump Dataset *(Bruinsma et al., 2024)*")
    st.markdown("**Dataset 2:** XJTU-SY Bearing Run-to-Failure Dataset *(Wang et al., 2020)*")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚙️ Pump Health Assessment System")
st.subheader("ISE 619 — Three-Layer AI Predictive Maintenance Framework")
st.markdown(
    "This system analyzes vibration sensor data through three AI layers — "
    "**anomaly detection**, **fault classification**, and "
    "**remaining useful life prediction** — to produce a maintenance decision."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_output(text: str) -> dict:
    """
    Parse the printed output of pipeline.py into a structured dict.

    Looks for:
      - "NORMAL" or "ANOMALY"  → Layer 1 status
      - "Fault" + "Classification" colon value → Layer 2 fault class
      - "Confidence" colon percentage          → Layer 2 confidence
      - "RUL Score" colon number               → Layer 3 RUL score
      - "Tier" colon HEALTHY/MONITOR/REPLACE   → Layer 3 tier
    """
    result = {}

    for line in text.splitlines():
        stripped = line.strip()

        # Layer 1 — "NORMAL" or "ANOMALY" on lines that mention Layer 1
        if "Layer 1" in stripped or "Anomaly Detection" in stripped:
            if "NORMAL" in stripped:
                result["l1_status"] = "NORMAL"
            elif "ANOMALY" in stripped:
                result["l1_status"] = "ANOMALY"

        # Layer 2 fault class — line contains "Fault" + "Classification" + a colon value
        # pipeline prints: " Layer 2 — Fault Classification: <class>"
        elif re.search(r"Fault.*Classification\s*:", stripped) and "NOT TRIGGERED" not in stripped:
            m = re.search(r"Fault.*Classification\s*:\s*(.+)", stripped)
            if m:
                result["fault_class"] = m.group(1).strip()

        # Layer 2 confidence — line contains "Confidence" and a percentage
        # pipeline prints: " Confidence                    : 87.43%"
        elif re.search(r"Confidence\s*:", stripped) and "%" in stripped:
            m = re.search(r"([\d.]+)\s*%", stripped)
            if m:
                result["confidence"] = float(m.group(1))

        # Layer 3 RUL score — line contains "RUL Score" and a number
        # pipeline prints: " Layer 3 — RUL Score           : 45.2 / 100"
        elif re.search(r"RUL Score\s*:", stripped):
            m = re.search(r"RUL Score\s*:\s*([\d.]+)", stripped)
            if m:
                result["rul_score"] = float(m.group(1))

        # Layer 3 tier — line contains "Tier" and one of the tier labels
        # pipeline prints: " Maintenance Tier              : 🟡 MONITOR"
        elif re.search(r"Tier\s*:", stripped):
            for tier in ("HEALTHY", "MONITOR", "REPLACE"):
                if tier in stripped:
                    result["tier"] = tier
                    break

    return result


def display_results(result: dict):
    """Render the three-layer results and final decision card."""
    if not result:
        st.error("Could not parse pipeline output. Check that the CSV matches the expected format.")
        return

    st.markdown("#### Analysis Results")

    # Layer 1
    l1 = result.get("l1_status", "NORMAL")
    if l1 == "NORMAL":
        st.success("**Layer 1 — Anomaly Detection: NORMAL** — No fault detected")
    else:
        st.error("**Layer 1 — Anomaly Detection: ANOMALY DETECTED**")

    # Layer 2
    if l1 == "ANOMALY" and "fault_class" in result:
        confidence = result.get("confidence", 0.0)
        st.warning(
            f"**Layer 2 — Fault Classification:** {result['fault_class']} "
            f"(Confidence: {confidence:.2f}%)"
        )

    # Layer 3
    if "rul_score" in result and "tier" in result:
        rul_score = result["rul_score"]
        tier = result["tier"]
        msg = f"**Layer 3 — Remaining Useful Life:** {rul_score:.1f}/100 — Tier: {tier}"
        if tier == "HEALTHY":
            st.success(msg)
        elif tier == "MONITOR":
            st.warning(msg)
        else:
            st.error(msg)

    # Final decision card
    st.markdown("---")
    tier = result.get("tier")

    if l1 == "NORMAL":
        st.markdown(
            """
            <div style="background-color:#d4edda;border:2px solid #28a745;
                        border-radius:10px;padding:28px;text-align:center;">
                <h2 style="color:#155724;margin:0;">✅ FINAL DECISION: NO ACTION REQUIRED</h2>
                <p style="color:#155724;font-size:1.1em;margin-top:10px;">
                    Pump is operating normally</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif tier in ("HEALTHY", "MONITOR"):
        st.markdown(
            """
            <div style="background-color:#fff3cd;border:2px solid #ffc107;
                        border-radius:10px;padding:28px;text-align:center;">
                <h2 style="color:#856404;margin:0;">⚠️ FINAL DECISION: SCHEDULE INSPECTION</h2>
                <p style="color:#856404;font-size:1.1em;margin-top:10px;">
                    Prepare spare parts and plan maintenance within next maintenance window</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color:#f8d7da;border:2px solid #dc3545;
                        border-radius:10px;padding:28px;text-align:center;">
                <h2 style="color:#721c24;margin:0;">🔴 FINAL DECISION: IMMEDIATE MAINTENANCE REQUIRED</h2>
                <p style="color:#721c24;font-size:1.1em;margin-top:10px;">
                    Take action now</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def run_on_file(csv_path: str):
    """Run pipeline.py as a subprocess, capture its output, parse and display results."""
    with st.spinner("Running AI analysis…"):
        try:
            proc = subprocess.run(
                [sys.executable, PIPELINE_SCRIPT, csv_path],
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            st.error(f"Failed to launch pipeline subprocess: {exc}")
            return

    if proc.returncode != 0:
        st.error(
            f"Pipeline exited with code {proc.returncode}.\n\n"
            f"```\n{proc.stderr.strip()}\n```"
        )
        return

    result = parse_output(proc.stdout)
    display_results(result)


# ── Demo Scenario Section ─────────────────────────────────────────────────────
st.markdown("## Run a Demo Scenario")

scenarios = {
    "Healthy Pump": "scenario_healthy.csv",
    "Early Fault (Monitor)": "scenario_anomaly_early.csv",
    "Critical Fault": "scenario_fault_critical.csv",
    "Imminent Failure": "scenario_imminent_failure.csv",
}

col1, col2, col3, col4 = st.columns(4)
clicked_scenario = None

for col, (label, filename) in zip([col1, col2, col3, col4], scenarios.items()):
    with col:
        if st.button(label, use_container_width=True):
            clicked_scenario = filename

if clicked_scenario:
    run_on_file(os.path.join(BASE_DIR, clicked_scenario))

# ── File Upload Section ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Upload Your Own Data")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    try:
        run_on_file(tmp_path)
    finally:
        os.unlink(tmp_path)
