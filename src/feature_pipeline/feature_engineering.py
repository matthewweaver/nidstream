"""
Feature Pipeline - Feature Engineering Module

Extracts temporal patterns, flow statistics, and protocol features
for anomaly detection in network flows.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, hour, dayofweek, minute,
    log1p, sqrt, when, lit
)
from pyspark.sql.types import DoubleType


def extract_temporal_features(df: DataFrame, timestamp_col: str = "Timestamp") -> DataFrame:
    """
    Extract time-based features from timestamp.
    
    Network attacks often show temporal patterns:
    - DoS attacks may peak at certain hours
    - Botnets may have periodic communication patterns
    """
    from pyspark.sql.functions import to_timestamp
    
    df = df.withColumn("ts", to_timestamp(timestamp_col))
    
    df = df.withColumn("hour_of_day", hour("ts"))
    df = df.withColumn("day_of_week", dayofweek("ts"))
    df = df.withColumn("minute_of_hour", minute("ts"))
    
    # Is it during business hours? (attacks may differ)
    df = df.withColumn(
        "is_business_hours",
        when((col("hour_of_day") >= 9) & (col("hour_of_day") <= 17), 1).otherwise(0)
    )
    
    return df


def create_ratio_features(df: DataFrame) -> DataFrame:
    """
    Create ratio-based features to capture flow asymmetry.
    
    Anomalous flows often show unusual ratios between forward/backward packets,
    bytes per packet, etc.
    """
    # Forward to backward packet ratio
    df = df.withColumn(
        "fwd_bwd_pkt_ratio",
        when(col("Tot Bwd Pkts") > 0, col("Tot Fwd Pkts") / col("Tot Bwd Pkts")).otherwise(0)
    )
    
    # Bytes per packet
    df = df.withColumn(
        "bytes_per_pkt",
        when(
            (col("Tot Fwd Pkts") + col("Tot Bwd Pkts")) > 0,
            (col("TotLen Fwd Pkts") + col("TotLen Bwd Pkts")) / 
            (col("Tot Fwd Pkts") + col("Tot Bwd Pkts"))
        ).otherwise(0)
    )
    
    # Packet rate (packets per second)
    df = df.withColumn(
        "pkt_rate",
        when(col("Flow Duration") > 0, 
             (col("Tot Fwd Pkts") + col("Tot Bwd Pkts")) / col("Flow Duration") * 1000000
        ).otherwise(0)
    )
    
    return df


def create_statistical_features(df: DataFrame) -> DataFrame:
    """
    Create statistical aggregations of flow features.
    
    These capture the distribution characteristics of packets in a flow.
    """
    # Coefficient of variation for packet lengths
    df = df.withColumn(
        "pkt_len_cv",
        when(col("Pkt Len Mean") > 0, col("Pkt Len Std") / col("Pkt Len Mean")).otherwise(0)
    )
    
    # Inter-arrival time stability
    df = df.withColumn(
        "iat_cv",
        when(col("Flow IAT Mean") > 0, col("Flow IAT Std") / col("Flow IAT Mean")).otherwise(0)
    )
    
    return df


def create_protocol_features(df: DataFrame) -> DataFrame:
    """
    Create protocol-specific features.
    
    Different attack types target different protocols (TCP, UDP, ICMP).
    """
    # Flag-based features
    df = df.withColumn(
        "syn_fin_ratio",
        when(col("FIN Flag Cnt") > 0, col("SYN Flag Cnt") / col("FIN Flag Cnt")).otherwise(0)
    )
    
    df = df.withColumn(
        "has_urgent_pkts",
        when((col("Fwd URG Flags") > 0) | (col("Bwd URG Flags") > 0), 1).otherwise(0)
    )
    
    return df


def apply_log_transform(df: DataFrame, cols_to_transform: list[str] = None) -> DataFrame:
    """
    Apply log transformation to skewed features.
    
    Many network flow features are heavily right-skewed (e.g., packet counts, durations).
    Log transformation helps normalize their distribution.
    """
    if cols_to_transform is None:
        cols_to_transform = [
            "Flow Duration",
            "Tot Fwd Pkts",
            "Tot Bwd Pkts",
            "TotLen Fwd Pkts",
            "TotLen Bwd Pkts",
            "Flow Byts/s",
            "Flow Pkts/s",
        ]
    
    for col_name in cols_to_transform:
        if col_name in df.columns:
            new_col_name = f"{col_name}_log"
            df = df.withColumn(new_col_name, log1p(col(col_name)))
    
    return df


def select_features_for_modeling(df: DataFrame) -> DataFrame:
    """
    Select relevant features for modeling and drop unnecessary columns.
    
    Removes:
    - Timestamp (already extracted temporal features)
    - Highly correlated or redundant features
    - Identifier columns
    """
    # Keep original label columns and drop Timestamp
    df = df.drop("Timestamp", "ts")
    
    # TODO: After EDA, add feature selection based on:
    # - Correlation analysis
    # - Feature importance from baseline models
    # - Domain knowledge
    
    return df


def feature_engineering_pipeline(df: DataFrame) -> DataFrame:
    """
    Run full feature engineering pipeline.
    
    Args:
        df: Preprocessed DataFrame
    
    Returns:
        DataFrame with engineered features
    """
    print("Starting feature engineering pipeline...")
    
    # 1. Extract temporal features
    df = extract_temporal_features(df)
    print("✓ Extracted temporal features")
    
    # 2. Create ratio features
    df = create_ratio_features(df)
    print("✓ Created ratio features")
    
    # 3. Create statistical features
    df = create_statistical_features(df)
    print("✓ Created statistical features")
    
    # 4. Create protocol features
    df = create_protocol_features(df)
    print("✓ Created protocol features")
    
    # 5. Apply log transformation
    df = apply_log_transform(df)
    print("✓ Applied log transformations")
    
    # 6. Select features
    df = select_features_for_modeling(df)
    print("✓ Selected features for modeling")
    
    print(f"\nFeature engineering complete! Final feature count: {len(df.columns)}")
    
    return df


if __name__ == "__main__":
    from .load import create_spark_session
    
    spark = create_spark_session("NIDStream-FeatureEngineering")
    
    # Load preprocessed data
    train_df = spark.read.parquet("data/processed/train_clean.parquet")
    val_df = spark.read.parquet("data/processed/val_clean.parquet")
    test_df = spark.read.parquet("data/processed/test_clean.parquet")
    
    # Feature engineering
    train_df = feature_engineering_pipeline(train_df)
    val_df = feature_engineering_pipeline(val_df)
    test_df = feature_engineering_pipeline(test_df)
    
    # Save feature-engineered data
    train_df.write.mode("overwrite").parquet("data/processed/train_features.parquet")
    val_df.write.mode("overwrite").parquet("data/processed/val_features.parquet")
    test_df.write.mode("overwrite").parquet("data/processed/test_features.parquet")
    
    print("\nSample features:")
    train_df.select(train_df.columns[:10]).show(5)
    
    spark.stop()
