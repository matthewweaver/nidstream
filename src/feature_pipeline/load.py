"""
Feature Pipeline - Data Loading Module

Loads CSE-CIC-IDS2018 network flow data from S3 using PySpark.
Handles large-scale CSV data distributed across S3 buckets.
"""

import os
from typing import Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType


def create_spark_session(app_name: str = "NIDStream-Load") -> SparkSession:
    """Create and configure Spark session for S3 access."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )
    return spark


def get_ids2018_schema() -> StructType:
    """
    Define schema for CSE-CIC-IDS2018 dataset.
    80+ network flow features including packet stats, protocol info, timing patterns.
    """
    # Common flow features
    schema = StructType([
        StructField("Dst Port", IntegerType(), True),
        StructField("Protocol", IntegerType(), True),
        StructField("Timestamp", StringType(), True),
        StructField("Flow Duration", DoubleType(), True),
        StructField("Tot Fwd Pkts", IntegerType(), True),
        StructField("Tot Bwd Pkts", IntegerType(), True),
        StructField("TotLen Fwd Pkts", DoubleType(), True),
        StructField("TotLen Bwd Pkts", DoubleType(), True),
        StructField("Fwd Pkt Len Max", DoubleType(), True),
        StructField("Fwd Pkt Len Min", DoubleType(), True),
        StructField("Fwd Pkt Len Mean", DoubleType(), True),
        StructField("Fwd Pkt Len Std", DoubleType(), True),
        StructField("Bwd Pkt Len Max", DoubleType(), True),
        StructField("Bwd Pkt Len Min", DoubleType(), True),
        StructField("Bwd Pkt Len Mean", DoubleType(), True),
        StructField("Bwd Pkt Len Std", DoubleType(), True),
        StructField("Flow Byts/s", DoubleType(), True),
        StructField("Flow Pkts/s", DoubleType(), True),
        StructField("Flow IAT Mean", DoubleType(), True),
        StructField("Flow IAT Std", DoubleType(), True),
        StructField("Flow IAT Max", DoubleType(), True),
        StructField("Flow IAT Min", DoubleType(), True),
        StructField("Fwd IAT Tot", DoubleType(), True),
        StructField("Fwd IAT Mean", DoubleType(), True),
        StructField("Fwd IAT Std", DoubleType(), True),
        StructField("Fwd IAT Max", DoubleType(), True),
        StructField("Fwd IAT Min", DoubleType(), True),
        StructField("Bwd IAT Tot", DoubleType(), True),
        StructField("Bwd IAT Mean", DoubleType(), True),
        StructField("Bwd IAT Std", DoubleType(), True),
        StructField("Bwd IAT Max", DoubleType(), True),
        StructField("Bwd IAT Min", DoubleType(), True),
        StructField("Fwd PSH Flags", IntegerType(), True),
        StructField("Bwd PSH Flags", IntegerType(), True),
        StructField("Fwd URG Flags", IntegerType(), True),
        StructField("Bwd URG Flags", IntegerType(), True),
        StructField("Fwd Header Len", IntegerType(), True),
        StructField("Bwd Header Len", IntegerType(), True),
        StructField("Fwd Pkts/s", DoubleType(), True),
        StructField("Bwd Pkts/s", DoubleType(), True),
        StructField("Pkt Len Min", DoubleType(), True),
        StructField("Pkt Len Max", DoubleType(), True),
        StructField("Pkt Len Mean", DoubleType(), True),
        StructField("Pkt Len Std", DoubleType(), True),
        StructField("Pkt Len Var", DoubleType(), True),
        StructField("FIN Flag Cnt", IntegerType(), True),
        StructField("SYN Flag Cnt", IntegerType(), True),
        StructField("RST Flag Cnt", IntegerType(), True),
        StructField("PSH Flag Cnt", IntegerType(), True),
        StructField("ACK Flag Cnt", IntegerType(), True),
        StructField("URG Flag Cnt", IntegerType(), True),
        StructField("CWE Flag Count", IntegerType(), True),
        StructField("ECE Flag Cnt", IntegerType(), True),
        StructField("Down/Up Ratio", DoubleType(), True),
        StructField("Pkt Size Avg", DoubleType(), True),
        StructField("Fwd Seg Size Avg", DoubleType(), True),
        StructField("Bwd Seg Size Avg", DoubleType(), True),
        StructField("Fwd Byts/b Avg", DoubleType(), True),
        StructField("Fwd Pkts/b Avg", DoubleType(), True),
        StructField("Fwd Blk Rate Avg", DoubleType(), True),
        StructField("Bwd Byts/b Avg", DoubleType(), True),
        StructField("Bwd Pkts/b Avg", DoubleType(), True),
        StructField("Bwd Blk Rate Avg", DoubleType(), True),
        StructField("Subflow Fwd Pkts", IntegerType(), True),
        StructField("Subflow Fwd Byts", DoubleType(), True),
        StructField("Subflow Bwd Pkts", IntegerType(), True),
        StructField("Subflow Bwd Byts", DoubleType(), True),
        StructField("Init Fwd Win Byts", IntegerType(), True),
        StructField("Init Bwd Win Byts", IntegerType(), True),
        StructField("Fwd Act Data Pkts", IntegerType(), True),
        StructField("Fwd Seg Size Min", IntegerType(), True),
        StructField("Active Mean", DoubleType(), True),
        StructField("Active Std", DoubleType(), True),
        StructField("Active Max", DoubleType(), True),
        StructField("Active Min", DoubleType(), True),
        StructField("Idle Mean", DoubleType(), True),
        StructField("Idle Std", DoubleType(), True),
        StructField("Idle Max", DoubleType(), True),
        StructField("Idle Min", DoubleType(), True),
        StructField("Label", StringType(), True),  # Target: Benign or Attack Type
    ])
    return schema


def load_from_s3(
    s3_path: str,
    spark: Optional[SparkSession] = None,
    sample_fraction: Optional[float] = None,
) -> DataFrame:
    """
    Load CSE-CIC-IDS2018 data from S3.
    
    Args:
        s3_path: S3 path (e.g., 's3a://bucket/path/*.csv')
        spark: Existing Spark session (creates new if None)
        sample_fraction: Fraction of data to sample (for testing)
    
    Returns:
        Spark DataFrame with network flow data
    """
    if spark is None:
        spark = create_spark_session()
    
    schema = get_ids2018_schema()
    
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .schema(schema)
        .csv(s3_path)
    )
    
    if sample_fraction:
        df = df.sample(fraction=sample_fraction, seed=42)
    
    print(f"Loaded {df.count()} records from {s3_path}")
    print(f"Schema: {len(df.columns)} columns")
    
    return df


def split_by_time(
    df: DataFrame,
    timestamp_col: str = "Timestamp",
    train_end: str = "2018-02-20",
    val_end: str = "2018-02-22",
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Split data by time to prevent data leakage.
    
    Args:
        df: Input DataFrame
        timestamp_col: Column containing timestamps
        train_end: End date for training (exclusive)
        val_end: End date for validation (exclusive)
    
    Returns:
        train_df, val_df, test_df
    """
    from pyspark.sql.functions import to_timestamp
    
    df = df.withColumn("ts", to_timestamp(timestamp_col))
    
    train_df = df.filter(f"ts < '{train_end}'")
    val_df = df.filter(f"ts >= '{train_end}' AND ts < '{val_end}'")
    test_df = df.filter(f"ts >= '{val_end}'")
    
    print(f"Train: {train_df.count()} | Val: {val_df.count()} | Test: {test_df.count()}")
    
    return train_df, val_df, test_df


def save_to_parquet(df: DataFrame, output_path: str, mode: str = "overwrite") -> None:
    """Save DataFrame to Parquet format for efficient storage."""
    df.write.mode(mode).parquet(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import sys
    
    # Example usage
    bucket = os.getenv("S3_BUCKET", "your-nids-data-bucket")
    s3_path = f"s3a://{bucket}/raw/CSE-CIC-IDS2018/*.csv"
    
    spark = create_spark_session()
    
    # Load data
    df = load_from_s3(s3_path, spark, sample_fraction=0.1 if "--sample" in sys.argv else None)
    
    # Split by time
    train_df, val_df, test_df = split_by_time(df)
    
    # Save splits
    save_to_parquet(train_df, "data/processed/train.parquet")
    save_to_parquet(val_df, "data/processed/val.parquet")
    save_to_parquet(test_df, "data/processed/test.parquet")
    
    spark.stop()
