"""
Streamlit Dashboard for Real-Time Network Intrusion Detection

Visualizes anomaly predictions, attack types, and model performance metrics.
"""

import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Page config
st.set_page_config(page_title="NIDStream Dashboard", page_icon="🛡️", layout="wide")

# API configuration
try:
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")
except:
    API_URL = "http://localhost:8000"

# Initialize session state for prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "total_flows" not in st.session_state:
    st.session_state.total_flows = 0
if "total_attacks" not in st.session_state:
    st.session_state.total_attacks = 0

st.title("🛡️ NIDStream - Network Intrusion Detection Dashboard")
st.markdown("Real-time anomaly detection for network flows")

# Check API health
try:
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    api_healthy = health_response.status_code == 200
except:
    api_healthy = False

if not api_healthy:
    st.error("⚠️ API is not responding. Make sure the API is running at " + API_URL)
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)
        time.sleep(refresh_interval)
        st.rerun()

    threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.05)

    st.header("📁 Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("Processing batch..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            try:
                batch_response = requests.post(f"{API_URL}/predict/batch", files=files, timeout=30)
                if batch_response.status_code == 200:
                    batch_data = batch_response.json()
                    st.session_state.total_flows += batch_data["total_flows"]
                    st.session_state.total_attacks += batch_data["attacks_detected"]

                    st.success(f"✅ Processed {batch_data['total_flows']} flows")
                    st.metric("Attacks Detected", batch_data["attacks_detected"])
                    st.metric("Attack Rate", f"{batch_data['attack_percentage']:.2f}%")

                    # Store batch results in history
                    for pred in batch_data["predictions"][:50]:  # Limit to 50
                        st.session_state.prediction_history.append(
                            {
                                "timestamp": datetime.now(),
                                "is_attack": pred["prediction"] == 1,
                                "confidence": pred["confidence"],
                                "probability": pred["attack_probability"],
                            }
                        )
                else:
                    st.error(f"Batch prediction failed: {batch_response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Get model info
try:
    model_response = requests.get(f"{API_URL}/model/info", timeout=5)
    if model_response.status_code == 200:
        model_info = model_response.json()
    else:
        model_info = None
except:
    model_info = None

# Main dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Flows Analyzed", f"{st.session_state.total_flows:,}")

with col2:
    st.metric("Attacks Detected", f"{st.session_state.total_attacks:,}")

with col3:
    if st.session_state.total_flows > 0:
        detection_rate = (st.session_state.total_attacks / st.session_state.total_flows) * 100
        st.metric("Attack Rate", f"{detection_rate:.2f}%")
    else:
        st.metric("Attack Rate", "N/A")

# Model info section
if model_info:
    with st.expander("📊 Model Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", model_info.get("model_type", "Unknown"))
        with col2:
            st.metric("Strategy", model_info.get("strategy", "Unknown"))
        with col3:
            st.metric("Features", model_info.get("n_features", "Unknown"))

        if model_info.get("metrics"):
            metrics = model_info["metrics"]
            st.subheader("Model Performance")
            metric_cols = st.columns(5)
            with metric_cols[0]:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            with metric_cols[1]:
                st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            with metric_cols[2]:
                st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            with metric_cols[3]:
                st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
            with metric_cols[4]:
                st.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")

# Prediction history visualization
if st.session_state.prediction_history:
    st.subheader("📈 Recent Prediction History")

    history_df = pd.DataFrame(st.session_state.prediction_history[-100:])  # Last 100 predictions

    fig = px.scatter(
        history_df,
        x="timestamp",
        y="probability",
        color="is_attack",
        size="confidence",
        title="Prediction Timeline (Last 100 Predictions)",
        labels={"probability": "Attack Probability", "timestamp": "Time"},
        color_discrete_map={True: "red", False: "green"},
    )

    fig.add_hline(y=threshold, line_dash="dash", line_color="orange", annotation_text=f"Threshold ({threshold})")

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2 = st.columns(2)
    with col1:
        attacks = history_df["is_attack"].sum()
        st.metric("Attacks in History", f"{attacks}/{len(history_df)}")
    with col2:
        avg_confidence = history_df["confidence"].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
else:
    st.info("📥 Upload a CSV file in the sidebar to start analyzing network flows")

# Attack statistics if we have history
if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    attacks_df = history_df[history_df["is_attack"] == True]

    if len(attacks_df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚠️ Attack Confidence Distribution")
            fig_hist = px.histogram(
                attacks_df,
                x="confidence",
                nbins=20,
                title="Confidence Levels of Detected Attacks",
                labels={"confidence": "Confidence", "count": "Number of Attacks"},
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.subheader("🎯 High Confidence Attacks")
            high_conf_attacks = attacks_df[attacks_df["confidence"] >= 0.8]
            st.metric("High Confidence (>80%)", f"{len(high_conf_attacks)}/{len(attacks_df)}")

            if len(high_conf_attacks) > 0:
                st.dataframe(high_conf_attacks[["timestamp", "probability", "confidence"]].tail(10), use_container_width=True)

# Model Performance Visualization
if model_info and model_info.get("metrics"):
    st.subheader("📊 Model Performance Metrics")
    metrics = model_info["metrics"]

    metrics_df = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
            "Value": [
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1", 0),
                metrics.get("roc_auc", 0),
                metrics.get("pr_auc", 0),
            ],
        }
    )

    fig_bar = px.bar(
        metrics_df,
        x="Metric",
        y="Value",
        title="Model Evaluation Metrics",
        color="Value",
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
    )
    fig_bar.update_layout(height=400, yaxis_range=[0, 1])
    st.plotly_chart(fig_bar, use_container_width=True)

# API Health Check
with st.expander("🔧 API Health Check", expanded=False):
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            st.success("✅ API is healthy")
            st.json(health_data)
        else:
            st.error(f"❌ API returned status code: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {str(e)}")
        st.info(f"Trying to connect to: {API_URL}")

# Footer
st.divider()
st.caption("NIDStream Dashboard v1.0 | Network Intrusion Detection System")
