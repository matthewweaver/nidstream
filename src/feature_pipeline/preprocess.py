"""
Feature Pipeline - Preprocessing Module

Cleans network flow data, handles missing values, removes outliers.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, isnan, count


def check_data_quality(df: DataFrame) -> None:
    """Print data quality summary."""
    print("\n=== Data Quality Summary ===")
    print(f"Total rows: {df.count()}")
    print(f"Total columns: {len(df.columns)}")
    
    # Check for nulls
    null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
    print("\nColumns with nulls:")
    for c in df.columns:
        null_count = null_counts.select(c).collect()[0][0]
        if null_count > 0:
            print(f"  {c}: {null_count}")
    
    # Check label distribution
    if "Label" in df.columns:
        print("\nLabel distribution:")
        df.groupBy("Label").count().orderBy("count", ascending=False).show(20, truncate=False)


def handle_missing_values(df: DataFrame) -> DataFrame:
    """
    Handle missing and infinite values in network flow features.
    
    Strategy:
    - Replace Inf/-Inf with None
    - Fill numeric columns with 0 (common for packet counts, flow stats)
    - Drop rows with missing labels
    """
    numeric_cols = [field.name for field in df.schema.fields 
                   if field.dataType.typeName() in ['double', 'integer', 'float']
                   and field.name != 'Label']
    
    # Replace Inf with None
    for col_name in numeric_cols:
        df = df.withColumn(
            col_name,
            when(col(col_name).isNotNull() & ~isnan(col(col_name)), col(col_name)).otherwise(None)
        )
    
    # Fill nulls with 0 for numeric columns
    df = df.fillna(0, subset=numeric_cols)
    
    # Drop rows with missing labels
    df = df.filter(col("Label").isNotNull())
    
    return df


def remove_duplicates(df: DataFrame) -> DataFrame:
    """Remove duplicate rows based on all columns."""
    before_count = df.count()
    df = df.dropDuplicates()
    after_count = df.count()
    print(f"Removed {before_count - after_count} duplicate rows")
    return df


def remove_outliers(df: DataFrame, cols_to_check: list[str] = None) -> DataFrame:
    """
    Remove outliers using IQR method on specified columns.
    
    Note: Be careful with outlier removal in anomaly detection - 
    some outliers might be actual attacks!
    This should be used cautiously or skipped entirely.
    """
    if cols_to_check is None:
        # Example: Remove outliers only on flow duration and packet counts
        cols_to_check = ["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts"]
    
    from pyspark.sql.functions import percentile_approx
    
    for col_name in cols_to_check:
        if col_name in df.columns:
            # Calculate Q1 and Q3
            quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
            q1, q3 = quantiles[0], quantiles[1]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            before_count = df.count()
            df = df.filter((col(col_name) >= lower_bound) & (col(col_name) <= upper_bound))
            after_count = df.count()
            
            print(f"Removed {before_count - after_count} outliers from {col_name}")
    
    return df


def normalize_labels(df: DataFrame) -> DataFrame:
    """
    Normalize label names and create binary label for anomaly detection.
    
    Maps all attack types to 1 (malicious) and Benign to 0.
    """
    from pyspark.sql.functions import trim, lower, when
    
    # Clean label strings
    df = df.withColumn("Label", trim(col("Label")))
    
    # Create binary label: 0 = Benign, 1 = Attack
    df = df.withColumn(
        "is_attack",
        when(lower(col("Label")) == "benign", 0).otherwise(1)
    )
    
    # Keep original label for multi-class analysis
    df = df.withColumn("attack_type", col("Label"))
    
    return df


def preprocess_pipeline(df: DataFrame, remove_outliers_flag: bool = False) -> DataFrame:
    """
    Run full preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        remove_outliers_flag: Whether to remove outliers (use cautiously)
    
    Returns:
        Preprocessed DataFrame
    """
    print("Starting preprocessing pipeline...")
    
    # 1. Data quality check
    check_data_quality(df)
    
    # 2. Handle missing values
    df = handle_missing_values(df)
    print("✓ Handled missing values")
    
    # 3. Remove duplicates
    df = remove_duplicates(df)
    print("✓ Removed duplicates")
    
    # 4. Normalize labels
    df = normalize_labels(df)
    print("✓ Normalized labels")
    
    # 5. Optionally remove outliers (use with caution!)
    if remove_outliers_flag:
        df = remove_outliers(df)
        print("✓ Removed outliers")
    
    print("\nPreprocessing complete!")
    check_data_quality(df)
    
    return df


if __name__ == "__main__":
    from .load import create_spark_session
    
    spark = create_spark_session("NIDStream-Preprocess")
    
    # Load raw data
    train_df = spark.read.parquet("data/processed/train.parquet")
    val_df = spark.read.parquet("data/processed/val.parquet")
    test_df = spark.read.parquet("data/processed/test.parquet")
    
    # Preprocess
    train_df = preprocess_pipeline(train_df, remove_outliers_flag=False)
    val_df = preprocess_pipeline(val_df, remove_outliers_flag=False)
    test_df = preprocess_pipeline(test_df, remove_outliers_flag=False)
    
    # Save preprocessed data
    train_df.write.mode("overwrite").parquet("data/processed/train_clean.parquet")
    val_df.write.mode("overwrite").parquet("data/processed/val_clean.parquet")
    test_df.write.mode("overwrite").parquet("data/processed/test_clean.parquet")
    
    spark.stop()
