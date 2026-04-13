"""
Shared utilities for model training notebooks.
Eliminates code duplication across model training notebooks.
Supports both pandas/sklearn (small datasets) and PySpark (large datasets).
"""

import time
from pathlib import Path

import joblib

# ============================================================================
# PANDAS/SKLEARN FUNCTIONS (For datasets < 5GB that fit in memory)
# ============================================================================


def load_training_data(use_smote=False):
    """
    Load training and test data from Parquet format (optimized for large datasets).

    Args:
        use_smote: If True, load SMOTE-balanced training data

    Returns:
        tuple: (X_train, X_test, y_train, y_test, project_root)
    """
    import numpy as np
    import pandas as pd

    project_root = Path().resolve()
    processed_dir = project_root / "data" / "processed" / "CSE-CIC-IDS2018"

    print(f"Loading {'balanced' if use_smote else 'original'} training data from Parquet...")

    if use_smote:
        # Load balanced data
        X_train = pd.read_parquet(processed_dir / "X_train_balanced.parquet")
        y_train = pd.read_parquet(processed_dir / "y_train_balanced.parquet")["label_binary"].values
    else:
        # Load original data
        X_train = pd.read_parquet(processed_dir / "X_train.parquet")
        y_train = pd.read_parquet(processed_dir / "y_train.parquet")["label_binary"].values

    # Test data is same for both strategies
    X_test = pd.read_parquet(processed_dir / "X_test.parquet")
    y_test = pd.read_parquet(processed_dir / "y_test.parquet")["label_binary"].values

    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Train class distribution: Benign={np.sum(y_train == 0)}, Attack={np.sum(y_train == 1)}")

    return X_train, X_test, y_train, y_test, project_root


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    """
    Train a model and evaluate on test set.

    Args:
        model: sklearn-compatible model instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_name: Name for printing

    Returns:
        tuple: (trained_model, metrics_dict)
    """
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    print("=" * 80)
    print(f"TRAINING: {model_name}")
    print("=" * 80)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"✅ Training completed in {train_time:.2f} seconds")

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "pr_auc": average_precision_score(y_test, y_pred_proba),
        "train_time": train_time,
    }

    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        if metric != "train_time":
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value:.2f}s")

    return model, metrics


def save_models(model_smote, model_weighted, metrics_smote, metrics_weighted, model_prefix, project_root):
    """
    Save trained models and metrics to disk.

    Args:
        model_smote: SMOTE-trained model
        model_weighted: Class weight-trained model
        metrics_smote: Metrics for SMOTE model
        metrics_weighted: Metrics for weighted model
        model_prefix: Prefix for filenames (e.g., 'lr', 'rf', 'xgb')
        project_root: Project root path
    """
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = models_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    smote_path = models_dir / f"{model_prefix}_smote.pkl"
    weighted_path = models_dir / f"{model_prefix}_weighted.pkl"

    joblib.dump(model_smote, smote_path)
    joblib.dump(model_weighted, weighted_path)

    print(f"✅ Saved: {smote_path}")
    print(f"✅ Saved: {weighted_path}")

    # Save metrics to metrics subfolder
    metrics = {f"{model_prefix.upper()}_SMOTE": metrics_smote, f"{model_prefix.upper()}_Weighted": metrics_weighted}
    metrics_path = metrics_dir / f"{model_prefix}_metrics.pkl"
    joblib.dump(metrics, metrics_path)

    print(f"✅ Saved metrics: {metrics_path}")


def log_to_mlflow(model, metrics, run_name, model_type, strategy, hyperparams, X_train, X_test, y_train, mlflow_logger=None):
    """
    Log model and metrics to MLflow.

    Args:
        model: Trained model
        metrics: Metrics dictionary
        run_name: Name for the MLflow run
        model_type: Type of model (e.g., 'LogisticRegression')
        strategy: Training strategy ('SMOTE' or 'Class_Weight')
        hyperparams: Dictionary of model hyperparameters
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        mlflow_logger: MLflow logging function (mlflow.sklearn, mlflow.xgboost, etc.)
    """
    import mlflow
    import numpy as np

    print(f"Logging {run_name} to MLflow...")

    with mlflow.start_run(run_name=run_name):
        # Log common parameters
        common_params = {
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "class_imbalance_ratio": float(np.sum(y_train == 0) / np.sum(y_train == 1)) if strategy != "SMOTE" else 1.0,
            "strategy": strategy,
            "model_type": model_type,
        }
        mlflow.log_params(common_params)

        # Log hyperparameters
        mlflow.log_params(hyperparams)

        # Log metrics
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "train_time_seconds": metrics["train_time"],
            }
        )

        # Log model with descriptive name
        # Convert run_name to artifact path: "LR_SMOTE" -> "lr_smote"
        artifact_path = run_name.lower().replace(" ", "_").replace("-", "_")
        if mlflow_logger:
            mlflow_logger.log_model(model, artifact_path)
        else:
            mlflow.sklearn.log_model(model, artifact_path)

        # Set tags
        model_family = run_name.split("_")[0]
        mlflow.set_tags({"model_family": model_family, "strategy": strategy})

        print(f"  ✅ Run ID: {mlflow.active_run().info.run_id}")


def print_summary(metrics_smote, metrics_weighted, model_name):
    """
    Print training summary comparing both strategies.

    Args:
        metrics_smote: Metrics for SMOTE model
        metrics_weighted: Metrics for weighted model
        model_name: Name of model (e.g., 'Logistic Regression')
    """
    print("\n" + "=" * 80)
    print(f"{model_name.upper()} TRAINING COMPLETE")
    print("=" * 80)

    print("\nSMOTE Strategy:")
    print(f"  PR-AUC: {metrics_smote['pr_auc']:.4f}")
    print(f"  F1 Score: {metrics_smote['f1']:.4f}")
    print(f"  Recall: {metrics_smote['recall']:.4f}")

    print("\nClass Weight Strategy:")
    print(f"  PR-AUC: {metrics_weighted['pr_auc']:.4f}")
    print(f"  F1 Score: {metrics_weighted['f1']:.4f}")
    print(f"  Recall: {metrics_weighted['recall']:.4f}")

    better = "SMOTE" if metrics_smote["pr_auc"] > metrics_weighted["pr_auc"] else "Class Weight"
    print(f"\n✨ Best Strategy: {better}")
    print("=" * 80)


# ============================================================================
# PYSPARK FUNCTIONS (For large datasets > 5GB that don't fit in memory)
# ============================================================================


def load_training_data_pyspark(spark, strategy="class_weight", verbose=True, data_path=None):
    """
    Load training and test data using PySpark for large datasets.

    Reads combined parquet files (features + label_binary in one file) written by
    notebooks/BCCC-CSE-CIC-IDS2018/02b_feature_engineering_pyspark.ipynb. The
    previous separate-X/Y layout is no longer supported — those files could not
    be reliably re-joined because the writes were independently repartitioned.

    Args:
        spark: SparkSession instance
        strategy: One of "class_weight", "balanced", "both".
                  - "class_weight" (default): only loads train.parquet + test.parquet.
                    train_balanced_vec is returned as None to skip the extra read.
                  - "balanced":     only loads train_balanced.parquet + test.parquet.
                    train_orig_vec is returned as None.
                  - "both":         loads everything (matches the old behaviour).
        verbose: If True, compute and display row counts and class distributions (slower).
                 Set False to skip these actions and speed up loading.
        data_path: Optional path to the processed data directory. Accepts local paths or
                   S3 URIs (e.g. "s3://nidstream/data/processed/BCCC-CSE-CIC-IDS2018").
                   Defaults to <project_root>/data/processed/CSE-CIC-IDS2018.

    Returns:
        tuple: (train_orig_vec, train_balanced_vec, test_vec, feature_cols, project_root)
            - train_orig_vec:     features-vector DataFrame for class-weight training
                                  (None when strategy="balanced")
            - train_balanced_vec: features-vector DataFrame for balanced training
                                  (None when strategy="class_weight")
            - test_vec:           features-vector DataFrame for evaluation
            - feature_cols:       list of feature column names
            - project_root:       project root path (or S3 base URI when data_path is an S3 URI)
    """
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql.functions import col, isnan, lit, when

    valid_strategies = {"class_weight", "balanced", "both"}
    if strategy not in valid_strategies:
        raise ValueError(f"strategy must be one of {valid_strategies}, got {strategy!r}")

    if data_path is not None:
        # Support S3 URIs and absolute local paths
        processed_dir = data_path.rstrip("/")
        project_root = processed_dir  # return the base path as project_root
        _join = lambda base, name: f"{base}/{name}"
    else:
        project_root = Path().resolve()
        processed_dir = project_root / "data" / "processed" / "CSE-CIC-IDS2018"
        _join = lambda base, name: str(base / name)

    load_orig = strategy in ("class_weight", "both")
    load_balanced = strategy in ("balanced", "both")

    print(f"Loading training and test data from Parquet with PySpark (strategy={strategy})...")

    # Test data is always needed for evaluation
    test_df = spark.read.parquet(_join(processed_dir, "test.parquet"))
    feature_cols = [c for c in test_df.columns if c != "label_binary"]

    train_orig = train_balanced = None
    if load_orig:
        train_orig = spark.read.parquet(_join(processed_dir, "train.parquet"))
    if load_balanced:
        train_balanced = spark.read.parquet(_join(processed_dir, "train_balanced.parquet"))

    if verbose:
        print(f"✓ Data loaded ({len(feature_cols)} features)")
        if train_orig is not None:
            print(f"  Original train: {train_orig.count():,} rows")
        if train_balanced is not None:
            print(f"  Balanced train: {train_balanced.count():,} rows")
        print(f"  Test:           {test_df.count():,} rows")
    else:
        print(f"✓ Data loaded ({len(feature_cols)} features, counts deferred for speed)")

    # Replace NaN and Infinity in feature columns before assembly.
    # Some source files contain division-by-zero artefacts (e.g. bytes_rate
    # when duration=0) that survive as Inf/NaN in the parquet. Replace with 0.0
    # so the model receives finite vectors.
    _inf = float("inf")
    feature_set = set(feature_cols)

    def _clean(df):
        if df is None:
            return None
        return df.select(
            [
                when(col(c).isNull() | isnan(col(c)) | (col(c) == _inf) | (col(c) == -_inf), lit(0.0)).otherwise(col(c)).alias(c)
                if c in feature_set
                else col(c)
                for c in df.columns
            ]
        )

    train_orig = _clean(train_orig)
    train_balanced = _clean(train_balanced)
    test_df = _clean(test_df)

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="error",
    )

    def _assemble(df):
        if df is None:
            return None
        return assembler.transform(df).select("features", col("label_binary").alias("label"))

    train_orig_vec = _assemble(train_orig)
    train_balanced_vec = _assemble(train_balanced)
    test_vec = _assemble(test_df)

    # Do NOT pre-cache here: train_and_evaluate_pyspark calls checkpoint() before
    # fitting, which serialises partitions to disk and bypasses the columnar
    # heap allocator that would OOM on a multi-GB cached DataFrame.
    print(f"✓ Assembled feature vectors (lazy — checkpointed by train_and_evaluate_pyspark)")

    if verbose:
        if train_orig_vec is not None:
            print("\nOriginal train class distribution:")
            train_orig_vec.groupBy("label").count().show()
        if train_balanced_vec is not None:
            print("\nBalanced train class distribution:")
            train_balanced_vec.groupBy("label").count().show()
        print("\nTest class distribution:")
        test_vec.groupBy("label").count().show()

    return train_orig_vec, train_balanced_vec, test_vec, feature_cols, project_root


def train_and_evaluate_pyspark(
    model_class,
    model_params,
    train_data,
    test_data,
    model_name,
    use_class_weights=False,
    n_partitions=None,
    checkpoint_dir=None,
):
    """
    Train a PySpark ML model and evaluate on test set.

    Optimised for memory-constrained environments (e.g. a laptop with a 10 GB dataset):
    # - Uses MEMORY_AND_DISK storage level (Python equivalent of JVM MEMORY_AND_DISK_SER;
    #   Python always pickles cached data so SER is implicit).
    - Coalesces to ``n_partitions`` before persisting to reduce task-scheduling overhead.
    - Checkpoints the training DataFrame when ``checkpoint_dir`` is set, truncating the
      lineage DAG that iterative solvers (L-BFGS, GBT) accumulate over many iterations.
    - Never mutates the caller's ``model_params`` dict.
    - Correctly handles the case where ``use_class_weights=True`` produces a *derived*
      DataFrame whose ``is_cached`` flag is always False even though its parent is cached.

    Recommended SparkSession settings for a laptop::

        spark.conf.set("spark.sql.shuffle.partitions", "8")   # default 200 is too many
        spark.conf.set("spark.driver.memory", "6g")
        spark.conf.set("spark.executor.memory", "6g")
        spark.conf.set("spark.memory.fraction", "0.8")

    Args:
        model_class: PySpark ML model class (e.g., LogisticRegression)
        model_params: Dictionary of model parameters (not mutated).
        train_data: Training data with 'features' and 'label' columns
        test_data: Test data with 'features' and 'label' columns
        model_name: Name for printing
        use_class_weights: If True, add class weight column to training data
        n_partitions: Coalesce train/test data to this many partitions before
            persisting.  ``None`` keeps the current partition count.  For a
            laptop with 4–8 cores, values between 8 and 32 work well.
        checkpoint_dir: If set, the (coalesced + optionally weighted) training
            DataFrame is checkpointed to this directory before fitting.  This
            truncates the lineage DAG and prevents recomputation cascades for
            iterative algorithms.  Example: ``"/tmp/spark_checkpoints"``.

    Returns:
        tuple: (trained_model, metrics_dict, predictions)
    """
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    from pyspark.sql.functions import col, when
    from pyspark.storagelevel import StorageLevel

    # Never mutate the caller's dict — surprising bugs otherwise when the same
    # params dict is reused for a second call (e.g. balanced then weighted).
    model_params = dict(model_params)

    print("=" * 80)
    print(f"TRAINING: {model_name}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Optional: reduce partition count before persisting.
    # Default Spark shuffle partitions (200) cause huge task-scheduling
    # overhead on a laptop.  Coalescing to n_partitions avoids writing
    # 200 files to disk and keeps the task queue manageable.
    # ------------------------------------------------------------------
    if n_partitions is not None:
        current = train_data.rdd.getNumPartitions()
        if current != n_partitions:
            train_data = train_data.coalesce(n_partitions) if current > n_partitions else train_data.repartition(n_partitions)
            test_data = (
                test_data.coalesce(n_partitions) if test_data.rdd.getNumPartitions() > n_partitions else test_data.repartition(n_partitions)
            )
            print(f"  Repartitioned: train {current}→{n_partitions}, test →{n_partitions} partitions")

    # ------------------------------------------------------------------
    # Class weights (computed from the *original* cached parent so we
    # avoid an extra full scan of the derived DataFrame).
    # ------------------------------------------------------------------
    if use_class_weights:
        from pyspark.sql.functions import count as spark_count

        count_row = train_data.agg(
            spark_count(when(col("label") == 0, 1)).alias("c0"),
            spark_count(when(col("label") == 1, 1)).alias("c1"),
        ).first()
        total = count_row["c0"] + count_row["c1"]
        class_weights = {
            0: total / (2 * count_row["c0"]),
            1: total / (2 * count_row["c1"]),
        }
        print(f"  Class weights: {class_weights}")

        # withColumn produces a *new* (uncached) DataFrame derived from the
        # (potentially cached) parent.  We mark it for re-persist below.
        train_data = train_data.withColumn(
            "classWeight",
            when(col("label") == 0, class_weights[0]).otherwise(class_weights[1]),
        )
        model_params["weightCol"] = "classWeight"

    # ------------------------------------------------------------------
    # Checkpoint training data to disk before fitting.
    #
    # WHY NOT persist()/cache():
    #   Every DataFrame persist() call — regardless of StorageLevel — routes
    #   through DefaultCachedBatchSerializer, which builds columnar HeapByteBuffer
    #   batches on the JVM heap.  On a 10 GB dataset this reliably causes
    #   OutOfMemoryError: Java heap space on a 16 GB laptop.
    #
    # WHY checkpoint() works:
    #   checkpoint() writes the DataFrame as serialized RDD partition files to
    #   disk (temp dir), completely bypassing the columnar serializer.  It also
    #   truncates the lineage DAG — critical for iterative solvers like L-BFGS
    #   which otherwise accumulate thousands of DAG nodes across iterations,
    #   causing slow planning and StackOverflowErrors.
    # ------------------------------------------------------------------
    ckpt_base = checkpoint_dir or "/tmp/spark_checkpoints"
    spark_ctx = train_data.sparkSession.sparkContext
    spark_ctx.setCheckpointDir(ckpt_base)
    print(f"  Checkpointing training data to {ckpt_base} …")
    train_data = train_data.checkpoint()  # eager write — blocks until all partitions written
    print(f"  Checkpoint complete ({train_data.rdd.getNumPartitions()} partitions)")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    start_time = time.time()
    model = model_class(**model_params)
    trained_model = model.fit(train_data)
    train_time = time.time() - start_time

    print(f"✓ Training completed in {train_time:.2f} seconds")
    if hasattr(trained_model, "summary"):
        print(f"  Iterations: {trained_model.summary.totalIterations}")
        if hasattr(trained_model.summary, "objectiveHistory"):
            print(f"  Objective history: {trained_model.summary.objectiveHistory[-1]:.6f} (final)")

    # ------------------------------------------------------------------
    # Evaluate — persist predictions once, run all 6 evaluators from cache
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"EVALUATING: {model_name}")
    print("=" * 80)

    # Checkpoint predictions too — avoids re-running transform() (which re-reads
    # test_data from Parquet and applies the model) for each of the 6 evaluators.
    # Using checkpoint rather than persist() for the same reason as training data:
    # bypasses the columnar heap allocator.
    eval_ckpt = (checkpoint_dir or "/tmp/spark_checkpoints") + "/predictions"
    train_data.sparkSession.sparkContext.setCheckpointDir(eval_ckpt)
    predictions = trained_model.transform(test_data).checkpoint()

    # Binary classification metrics
    binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
    pr_auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderPR"})

    # Multiclass metrics (for accuracy, precision, recall, F1)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
    recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
    f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "train_time": train_time,
    }

    print(f"Test Set Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")
    print(f"  PR AUC:    {pr_auc:.4f}")
    print(f"  Train time: {train_time:.2f}s")

    return trained_model, metrics, predictions


def save_models_pyspark(model_balanced, model_weighted, metrics_balanced, metrics_weighted, model_prefix, models_root):
    """
    Save PySpark ML models and metrics. Supports both local paths and S3 URIs.

    Either model/metrics pair may be None — in that case the corresponding
    strategy is silently skipped (matches strategy="class_weight" or "balanced"
    in load_training_data_pyspark).

    Args:
        model_balanced:  Balanced-trained model (may be None)
        model_weighted:  Class weight-trained model (may be None)
        metrics_balanced: Metrics for balanced model (may be None)
        metrics_weighted: Metrics for weighted model (may be None)
        model_prefix:    Prefix for filenames (e.g., 'lr', 'rf', 'xgb')
        models_root:     Root path — local Path/str or S3 URI (s3://bucket/models)
    """
    import pickle

    models_root = str(models_root).rstrip("/")
    is_s3 = models_root.startswith("s3://")

    print(f"✅ Saved models:")
    if model_balanced is not None:
        path = f"{models_root}/pyspark/{model_prefix}_balanced"
        model_balanced.write().overwrite().save(path)
        print(f"  {path}")
    if model_weighted is not None:
        path = f"{models_root}/pyspark/{model_prefix}_weighted"
        model_weighted.write().overwrite().save(path)
        print(f"  {path}")

    metrics = {}
    if metrics_balanced is not None:
        metrics[f"{model_prefix.upper()}_Balanced"] = metrics_balanced
    if metrics_weighted is not None:
        metrics[f"{model_prefix.upper()}_Weighted"] = metrics_weighted
    metrics_key = f"models/metrics/{model_prefix}_pyspark_metrics.pkl"

    if is_s3:
        import io

        import boto3

        bucket = models_root.split("/")[2]
        prefix = "/".join(models_root.split("/")[3:])
        metrics_s3_key = f"{prefix}/metrics/{model_prefix}_pyspark_metrics.pkl".lstrip("/")
        buf = io.BytesIO()
        pickle.dump(metrics, buf)
        buf.seek(0)
        boto3.client("s3").put_object(Bucket=bucket, Key=metrics_s3_key, Body=buf.read())
        metrics_path = f"s3://{bucket}/{metrics_s3_key}"
    else:
        metrics_dir = Path(models_root) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"{model_prefix}_pyspark_metrics.pkl"
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics, f)

    print(f"✅ Saved metrics: {metrics_path}")


def log_to_mlflow_pyspark(
    model_balanced,
    model_weighted,
    metrics_balanced,
    metrics_weighted,
    model_prefix,
    model_name,
    model_params,
    models_root,
    tracking_uri=None,
):
    """
    Log PySpark model runs to MLflow.

    Works with both a local HTTP tracking server (local runs) and direct S3
    file-store writes (EMR runs, where the tracking server is unreachable).

    Args:
        model_balanced: Balanced-trained PySpark model
        model_weighted: Class-weight-trained PySpark model
        metrics_balanced: Metrics dict from train_and_evaluate_pyspark (balanced)
        metrics_weighted: Metrics dict from train_and_evaluate_pyspark (weighted)
        model_prefix: Short model identifier, e.g. 'lr', 'rf'
        model_name: Human-readable name, e.g. 'Logistic Regression'
        model_params: Dict of hyperparameters passed to the model
        models_root: Root used when saving models (logged as a tag for traceability)
        tracking_uri: MLflow tracking URI. If None, uses MLFLOW_TRACKING_URI env var
                      or the default (./mlruns).  Pass 's3://bucket/mlflow' for EMR
                      or 'http://localhost:5000' for a local server.
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(f"nidstream_{model_prefix}")
    models_root = str(models_root).rstrip("/")

    runs = [(s, m) for s, m in (("balanced", metrics_balanced), ("class_weight", metrics_weighted)) if m is not None]
    for strategy, metrics in runs:
        with mlflow.start_run(run_name=f"{model_prefix.upper()}_{strategy}"):
            mlflow.log_params(
                {
                    **model_params,
                    "strategy": strategy,
                    "model_type": model_name,
                    "framework": "pyspark",
                }
            )
            mlflow.log_metrics(
                {
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "train_time_seconds": metrics["train_time"],
                }
            )
            mlflow.set_tag("model_path", f"{models_root}/pyspark/{model_prefix}_{strategy}")
            print(f"  {strategy}: run_id={mlflow.active_run().info.run_id}")

    print(f"MLflow experiment : nidstream_{model_prefix}")
    print(f"Tracking URI      : {mlflow.get_tracking_uri()}")


def print_summary_pyspark(metrics_balanced, metrics_weighted, model_name):
    """
    Print training summary for PySpark models. Either metrics dict may be None;
    in that case only the strategy that ran is printed (no comparison line).

    Args:
        metrics_balanced: Metrics for balanced model (may be None)
        metrics_weighted: Metrics for weighted model (may be None)
        model_name: Name of model (e.g., 'Logistic Regression')
    """
    print("\n" + "=" * 80)
    print(f"{model_name.upper()} MODEL SUMMARY")
    print("=" * 80)

    def _print(metrics, header):
        print(f"\n{header}:")
        for metric, value in metrics.items():
            if metric != "train_time":
                print(f"  {metric:12s}: {value:.4f}")
            else:
                print(f"  {metric:12s}: {value:.2f}s")

    if metrics_balanced is not None:
        _print(metrics_balanced, "Balanced Data Strategy")
    if metrics_weighted is not None:
        _print(metrics_weighted, "Class Weight Strategy")

    if metrics_balanced is not None and metrics_weighted is not None:
        better = "Balanced" if metrics_balanced["pr_auc"] > metrics_weighted["pr_auc"] else "Class Weight"
        print(f"\n✨ Best Strategy (by PR AUC): {better}")
    print("=" * 80)
