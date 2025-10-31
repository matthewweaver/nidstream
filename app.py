"""
Streamlit Dashboard for Real-Time Network Intrusion Detection

Visualizes anomaly predictions, attack types, and model performance metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime

# Page config
st.set_page_config(
    page_title="NIDStream Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# API configuration
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.title("🛡️ NIDStream - Network Intrusion Detection Dashboard")
st.markdown("Real-time anomaly detection for network flows")

# Sidebar
with st.sidebar:
    st.header("Settings")
    refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)
    threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.header("Filters")
    show_attacks_only = st.checkbox("Show Attacks Only", value=False)

# Main dashboard
col1, col2, col3 = st.columns(3)

# TODO: Connect to API and display real data
# For now, showing placeholder metrics
with col1:
    st.metric("Total Flows Analyzed", "1,234,567", "+1,234")

with col2:
    st.metric("Attacks Detected", "12,345", "+23")

with col3:
    st.metric("Detection Rate", "99.2%", "+0.1%")

# Time series plot
st.subheader("Anomaly Score Over Time")

# Placeholder data - replace with real predictions
sample_data = pd.DataFrame({
    'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='min'),
    'anomaly_score': [0.1 + 0.5 * (i % 10 == 0) for i in range(100)],
    'is_attack': [(i % 10 == 0) for i in range(100)]
})

fig = px.line(sample_data, x='timestamp', y='anomaly_score', 
              title='Anomaly Scores (Live Feed)',
              color_discrete_sequence=['#1f77b4'])
fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
              annotation_text="Threshold")

# Highlight attacks
attacks = sample_data[sample_data['is_attack']]
fig.add_scatter(x=attacks['timestamp'], y=attacks['anomaly_score'],
                mode='markers', marker=dict(color='red', size=10),
                name='Detected Attacks')

st.plotly_chart(fig, use_container_width=True)

# Attack type distribution
col1, col2 = st.columns(2)

with col1:
    st.subheader("Attack Type Distribution")
    attack_types = pd.DataFrame({
        'Attack Type': ['Botnet', 'DoS', 'DDoS', 'Infiltration', 'Brute Force'],
        'Count': [234, 456, 123, 89, 156]
    })
    fig_pie = px.pie(attack_types, values='Count', names='Attack Type',
                     title='Detected Attack Types (Last 24h)')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Model Performance")
    metrics = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Value': [0.95, 0.93, 0.94, 0.98]
    })
    fig_bar = px.bar(metrics, x='Metric', y='Value',
                     title='Model Performance Metrics',
                     color='Value', color_continuous_scale='Blues')
    fig_bar.update_yaxis(range=[0, 1])
    st.plotly_chart(fig_bar, use_container_width=True)

# Recent detections table
st.subheader("Recent Attack Detections")
recent_attacks = pd.DataFrame({
    'Timestamp': pd.date_range(start='2024-01-01 10:00', periods=10, freq='min'),
    'Source IP': [f'192.168.{i}.{j}' for i, j in zip(range(10), range(100, 110))],
    'Destination IP': [f'10.0.{i}.{j}' for i, j in zip(range(10), range(200, 210))],
    'Attack Type': ['DoS', 'Botnet', 'DDoS', 'Infiltration', 'Brute Force'] * 2,
    'Anomaly Score': [0.92, 0.87, 0.95, 0.78, 0.89, 0.91, 0.85, 0.93, 0.82, 0.88]
})

st.dataframe(recent_attacks, use_container_width=True)

# Test API connection
with st.expander("API Health Check"):
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✓ API is healthy")
            st.json(response.json())
        else:
            st.error(f"API returned status code: {response.status_code}")
    except Exception as e:
        st.error(f"Cannot connect to API: {str(e)}")
        st.info(f"Trying to connect to: {API_URL}")
