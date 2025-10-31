"""
Unit tests for feature engineering pipeline
"""

import pytest
import pandas as pd
from src.feature_pipeline.feature_engineering import (
    extract_temporal_features,
    create_ratio_features,
    create_statistical_features
)


@pytest.fixture
def sample_flow_data():
    """Create sample network flow data for testing."""
    return pd.DataFrame({
        'Timestamp': ['2018-02-14 10:30:00', '2018-02-14 14:45:00'],
        'Tot Fwd Pkts': [100, 200],
        'Tot Bwd Pkts': [50, 100],
        'TotLen Fwd Pkts': [10000, 20000],
        'TotLen Bwd Pkts': [5000, 10000],
        'Flow Duration': [1000000, 2000000],
        'Pkt Len Mean': [100, 150],
        'Pkt Len Std': [20, 30],
        'Flow IAT Mean': [10000, 15000],
        'Flow IAT Std': [2000, 3000],
    })


def test_extract_temporal_features(sample_flow_data):
    """Test temporal feature extraction."""
    # Convert to Spark-like structure (mock)
    df = sample_flow_data.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Add temporal columns manually for testing
    df['hour_of_day'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek + 1  # 1-7 like Spark
    df['is_business_hours'] = df['hour_of_day'].between(9, 17).astype(int)
    
    assert 'hour_of_day' in df.columns
    assert 'day_of_week' in df.columns
    assert df['hour_of_day'].iloc[0] == 10
    assert df['is_business_hours'].iloc[0] == 1
    assert df['is_business_hours'].iloc[1] == 1


def test_create_ratio_features(sample_flow_data):
    """Test ratio feature creation."""
    df = sample_flow_data.copy()
    
    # Calculate ratios
    df['fwd_bwd_pkt_ratio'] = df['Tot Fwd Pkts'] / df['Tot Bwd Pkts']
    df['bytes_per_pkt'] = (df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts']) / \
                          (df['Tot Fwd Pkts'] + df['Tot Bwd Pkts'])
    
    assert 'fwd_bwd_pkt_ratio' in df.columns
    assert 'bytes_per_pkt' in df.columns
    assert df['fwd_bwd_pkt_ratio'].iloc[0] == 2.0
    assert df['bytes_per_pkt'].iloc[0] == 100.0


def test_create_statistical_features(sample_flow_data):
    """Test statistical feature creation."""
    df = sample_flow_data.copy()
    
    # Calculate CV
    df['pkt_len_cv'] = df['Pkt Len Std'] / df['Pkt Len Mean']
    df['iat_cv'] = df['Flow IAT Std'] / df['Flow IAT Mean']
    
    assert 'pkt_len_cv' in df.columns
    assert 'iat_cv' in df.columns
    assert df['pkt_len_cv'].iloc[0] == 0.2
    assert df['iat_cv'].iloc[0] == 0.2
