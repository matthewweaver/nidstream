"""
Streamlit Dashboard for Real-Time Network Intrusion Detection

Visualizes anomaly predictions, attack types, and model performance metrics.
"""

import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Page config
st.set_page_config(page_title="NIDStream Dashboard", page_icon="🛡️", layout="wide")

# API configuration - read from environment variable or Streamlit secrets
API_URL = os.environ.get("API_URL")
if not API_URL:
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
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    api_healthy = health_response.status_code == 200
except Exception as e:
    api_healthy = False
    st.error(f"⚠️ API is not responding at {API_URL}")
    st.error(f"Error: {str(e)}")
    st.info("If running in Docker Compose, the API should be at http://api:8000")
    st.stop()

if not api_healthy:
    st.error(f"⚠️ API returned unhealthy status at {API_URL}")
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

    st.header("🌊 Temporal Streaming")
    st.markdown("Simulate real-time flow streaming with temporal delays")

    # Initialize streaming state
    if "streaming_active" not in st.session_state:
        st.session_state.streaming_active = False
    if "streaming_stats" not in st.session_state:
        st.session_state.streaming_stats = {"total_sent": 0, "attacks_detected": 0, "duration": 0}
    if "streaming_data" not in st.session_state:
        st.session_state.streaming_data = None
    if "streaming_index" not in st.session_state:
        st.session_state.streaming_index = 0
    if "stream_start_time" not in st.session_state:
        st.session_state.stream_start_time = None

    # File upload for temporal data
    temporal_file = st.file_uploader(
        "Upload temporal CSV (with timestamp column)",
        type=["csv"],
        key="temporal_upload",
        help="Upload X_test_temporal.csv or similar file with timestamps",
    )

    if temporal_file is not None:
        # Load temporal data
        try:
            df_temporal = pd.read_csv(temporal_file)

            if "timestamp" not in df_temporal.columns:
                st.error("❌ File must contain a 'timestamp' column")
            else:
                df_temporal["timestamp"] = pd.to_datetime(df_temporal["timestamp"])
                df_temporal = df_temporal.sort_values("timestamp").reset_index(drop=True)
                st.session_state.streaming_data = df_temporal

                st.success(f"✅ Loaded {len(df_temporal)} flows")
                st.info(f"📅 Time range: {df_temporal['timestamp'].min()} to {df_temporal['timestamp'].max()}")

                # Calculate average delay
                time_diffs = df_temporal["timestamp"].diff().dropna()
                avg_delay = time_diffs.mean().total_seconds()
                st.info(f"⏱️ Average delay: {avg_delay:.2f} seconds")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    # Streaming controls
    stream_speed = st.select_slider(
        "Stream Speed",
        options=[0.5, 1.0, 2.0, 5.0, 10.0],
        value=2.0,
        format_func=lambda x: f"{x}x",
        disabled=st.session_state.streaming_active,
    )

    max_flows = st.number_input(
        "Max Flows (0 = unlimited)",
        min_value=0,
        max_value=10000,
        value=0,
        step=10,
        disabled=st.session_state.streaming_active,
        help="Set to 0 for continuous streaming. Otherwise, stream will stop after N flows.",
    )

    # Start/Stop streaming button
    col1, col2 = st.columns(2)

    with col1:
        start_disabled = st.session_state.streaming_active or st.session_state.streaming_data is None
        if st.button("▶️ Start Stream", disabled=start_disabled):
            st.session_state.streaming_active = True
            st.session_state.streaming_index = 0
            st.session_state.stream_start_time = time.time()
            st.session_state.streaming_stats = {"total_sent": 0, "attacks_detected": 0, "duration": 0}
            st.rerun()

    with col2:
        if st.button("⏹️ Stop Stream", disabled=not st.session_state.streaming_active):
            st.session_state.streaming_active = False
            elapsed = time.time() - st.session_state.stream_start_time if st.session_state.stream_start_time else 0
            st.session_state.streaming_stats["duration"] = elapsed
            # Store stream settings
            st.session_state.stream_speed = stream_speed
            st.session_state.max_flows = max_flows
            st.success("Stream stopped")

    # Store stream settings when starting
    if st.session_state.streaming_active and "stream_speed" not in st.session_state:
        st.session_state.stream_speed = stream_speed
        st.session_state.max_flows = max_flows

    # Display streaming stats in sidebar
    if st.session_state.streaming_stats["total_sent"] > 0:
        st.markdown("**Stream Statistics:**")
        stats = st.session_state.streaming_stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Flows Sent", stats["total_sent"])
        col2.metric("Attacks", stats["attacks_detected"])

        if stats["total_sent"] > 0:
            attack_rate = stats["attacks_detected"] / stats["total_sent"] * 100
            col3.metric("Attack Rate", f"{attack_rate:.1f}%")

        if stats["duration"] > 0:
            st.caption(f"Duration: {stats['duration']:.1f}s | Throughput: {stats['total_sent'] / stats['duration']:.1f} flows/sec")

# Get model info (before streaming execution so it renders before rerun)
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

# Show streaming status in main area
if st.session_state.streaming_active and st.session_state.streaming_data is not None:
    st.markdown("---")

    # Create a prominent streaming banner
    cols = st.columns([1, 3, 1])
    with cols[0]:
        st.markdown("## 🔴 LIVE")
    with cols[1]:
        st.markdown("## Network Flow Stream Active")
    with cols[2]:
        if st.button("⏹️ STOP", type="primary", key="stop_main"):
            st.session_state.streaming_active = False
            elapsed = time.time() - st.session_state.stream_start_time
            st.session_state.streaming_stats["duration"] = elapsed
            st.rerun()

    df = st.session_state.streaming_data
    idx = st.session_state.streaming_index
    max_flows_exec = st.session_state.get("max_flows", 0)

    if max_flows_exec > 0:
        max_idx = min(max_flows_exec, len(df))
    else:
        max_idx = len(df)

    # Progress bar (only show if limited)
    if max_flows_exec > 0:
        progress = idx / max_idx if max_idx > 0 else 0
        st.progress(progress, text=f"Processing flow {idx}/{max_idx} ({progress * 100:.1f}%)")
    else:
        st.info(f"🔄 Continuous streaming mode - Flow {idx} of {len(df)} (will loop)")

    # Show last prediction if available
    if st.session_state.prediction_history:
        last_pred = st.session_state.prediction_history[-1]

        # Prominent display of latest prediction
        if last_pred["is_attack"]:
            st.error("🚨 **ATTACK DETECTED**")
        else:
            st.success("✅ **Benign Traffic**")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            timestamp_str = (
                last_pred["timestamp"].strftime("%H:%M:%S") if hasattr(last_pred["timestamp"], "strftime") else str(last_pred["timestamp"])
            )
            st.metric("Latest Timestamp", timestamp_str)
        with col2:
            status = "ATTACK" if last_pred["is_attack"] else "BENIGN"
            st.metric("Prediction", status)
        with col3:
            st.metric("Attack Probability", f"{last_pred['probability']:.1%}")
        with col4:
            st.metric("Confidence", f"{last_pred['confidence']:.1%}")

    st.markdown("---")

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

# Prediction history visualization - Always show if we have data
# THIS RENDERS BEFORE THE STREAMING EXECUTION SO IT APPEARS IN REAL-TIME
if st.session_state.prediction_history:
    st.subheader("📈 Prediction Timeline (Live Updates)")

    # Show live feed of recent predictions if streaming
    if st.session_state.streaming_active:
        st.markdown("### 🔴 LIVE PREDICTION FEED")

        # Show last 10 predictions in a table with color coding
        recent_preds = st.session_state.prediction_history[-10:]

        for i, pred in enumerate(reversed(recent_preds)):
            timestamp_str = (
                pred["timestamp"].strftime("%H:%M:%S.%f")[:-3] if hasattr(pred["timestamp"], "strftime") else str(pred["timestamp"])
            )

            if pred["is_attack"]:
                st.markdown(
                    f"🔴 **[{timestamp_str}]** ATTACK DETECTED - "
                    f"Probability: {pred['probability']:.1%} | "
                    f"Confidence: {pred['confidence']:.1%}"
                )
            else:
                st.markdown(f"🟢 [{timestamp_str}] Benign - Probability: {pred['probability']:.1%} | Confidence: {pred['confidence']:.1%}")

        st.markdown("---")

    # Always show chart - updates in real-time during streaming
    history_df = pd.DataFrame(st.session_state.prediction_history[-100:])  # Last 100 predictions

    fig = px.scatter(
        history_df,
        x="timestamp",
        y="probability",
        color="is_attack",
        size="confidence",
        title=f"Last {len(history_df)} Predictions" + (" - LIVE" if st.session_state.streaming_active else ""),
        labels={"probability": "Attack Probability", "timestamp": "Time"},
        color_discrete_map={True: "red", False: "green"},
    )

    fig.add_hline(y=threshold, line_dash="dash", line_color="orange", annotation_text=f"Threshold ({threshold})")

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, key="prediction_timeline_main")

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
            st.plotly_chart(fig_hist, use_container_width=True, key="attack_confidence_hist_main")

        with col2:
            st.subheader("🎯 High Confidence Attacks")
            high_conf_attacks = attacks_df[attacks_df["confidence"] >= 0.8]
            st.metric("High Confidence (>80%)", f"{len(high_conf_attacks)}/{len(attacks_df)}")

            if len(high_conf_attacks) > 0:
                st.dataframe(high_conf_attacks[["timestamp", "probability", "confidence"]].tail(10), use_container_width=True)

# ============================================================
# STREAMING EXECUTION (MUST BE AT END - calls st.rerun())
# ============================================================
if st.session_state.streaming_active and st.session_state.streaming_data is not None:
    df = st.session_state.streaming_data
    idx = st.session_state.streaming_index

    # Get stream settings from session state
    stream_speed_exec = st.session_state.get("stream_speed", 2.0)
    max_flows_exec = st.session_state.get("max_flows", 0)

    # Determine max index - if max_flows is 0, loop continuously through data
    if max_flows_exec > 0:
        max_idx = min(max_flows_exec, len(df))
    else:
        # Unlimited streaming - loop through the entire dataset repeatedly
        max_idx = len(df)
        # Reset index if we've reached the end (loop back to start)
        if idx >= len(df):
            st.session_state.streaming_index = 0
            idx = 0

    if idx < max_idx or max_flows_exec == 0:
        # Get current row
        row = df.iloc[idx]

        # Calculate delay from previous flow
        delay = 0
        if idx > 0:
            prev_timestamp = df.iloc[idx - 1]["timestamp"]
            curr_timestamp = row["timestamp"]
            time_diff = (curr_timestamp - prev_timestamp).total_seconds()
            delay = max(0.1, time_diff / stream_speed_exec)  # Minimum 0.1s delay

        # Send prediction
        flow_dict = row.drop("timestamp").to_dict()

        try:
            response = requests.post(f"{API_URL}/predict", json=flow_dict, timeout=5)

            if response.status_code == 200:
                result = response.json()

                # Update stats
                st.session_state.streaming_stats["total_sent"] = idx + 1
                if result.get("prediction") == 1:
                    st.session_state.streaming_stats["attacks_detected"] += 1

                # Add to prediction history
                st.session_state.prediction_history.append(
                    {
                        "timestamp": row["timestamp"],
                        "is_attack": result.get("prediction") == 1,
                        "confidence": result.get("confidence", 0),
                        "probability": result.get("attack_probability", 0),
                    }
                )

                # Update total counters
                st.session_state.total_flows += 1
                if result.get("prediction") == 1:
                    st.session_state.total_attacks += 1

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

        # Increment index
        st.session_state.streaming_index = idx + 1

        # Wait and rerun for next flow
        time.sleep(delay)
        st.rerun()
    else:
        # Streaming complete
        st.session_state.streaming_active = False
        elapsed = time.time() - st.session_state.stream_start_time
        st.session_state.streaming_stats["duration"] = elapsed

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
