# src/drift_detection.py

from pyspark.sql import SparkSession
import yaml
import json
import os
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_baseline_stats(stats_path):
    # Load baseline stats JSON (output from baseline_stats.py)

    # stats_df = spark.read.json(stats_path)
    with open(stats_path, "r") as f:
        baseline_stats = json.load(f)

    # stats_dict = stats_df.collect()
    
    # # Convert list of Rows to dict of stats per feature
    # baseline_stats = {}
    # for row in stats_dict:
    #     metric = row['summary']  # e.g. 'mean', 'stddev', 'min', etc.
    #     for feature in row.asDict():
    #         if feature != 'summary':
    #             baseline_stats.setdefault(feature, {})[metric] = float(row[feature]) if row[feature] is not None else None

    return baseline_stats

def compute_new_stats(df, feature_cols):

    stats = {}
    # Compute descriptive stats for new data batch
    # stats_df = df.select(feature_cols).describe()
    stats_df = df.describe().select(feature_cols)

    stats = stats_df.collect()

    new_stats = {feature: {} for feature in feature_cols}
    
    for row in stats:
        metric = row["summary"]
        for feature in feature_cols:
            try:
                value = float(row[feature])
            except (ValueError, TypeError):
                value = None
            new_stats[feature][metric] = value

    return new_stats

def detect_drift(baseline_stats, new_stats, threshold=0.1):
    """
    Simple drift detection:
    Compare mean values of features.
    If absolute relative difference > threshold, flag drift.
    """
    drift_report = {}
    for feature in baseline_stats:
        base_mean = baseline_stats[feature].get('mean')
        new_mean = new_stats.get(feature, {}).get('mean')
        if base_mean is None or new_mean is None:
            continue
        if base_mean == 0:
            # Avoid division by zero
            diff = abs(new_mean - base_mean)
        else:
            diff = abs(new_mean - base_mean) / base_mean
        
        drift_report[feature] = {
            'baseline_mean': base_mean,
            'new_mean': new_mean,
            'relative_diff': diff,
            'drift': diff > threshold
        }

    return drift_report

def run_drift_detection(spark, new_data_path, baseline_stats_path, output_path, threshold=0.1):

    try:

        # Load new data to test for drift
        new_df = spark.read.parquet(new_data_path)
        
        # Load baseline stats
        baseline_stats = load_baseline_stats(baseline_stats_path)
        feature_cols = list(baseline_stats.keys())
        
        # Compute new data stats
        new_stats = compute_new_stats(new_df, feature_cols)
        
        # Detect drift
        drift_report = detect_drift(baseline_stats, new_stats, threshold)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(drift_report, f, indent=4)

        logger.info(f"Drift report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        sys.exit(1)

    return drift_report


if __name__ == "__main__":

    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    executor_mem = params["spark"]["executor_memory"]
    driver_mem = params["spark"]["driver_memory"]
    cores = params["spark"]["executor_cores"]

    # Start Spark session
    spark = SparkSession.builder \
        .appName("DriftTitanic") \
        .config("spark.executor.memory", executor_mem) \
        .config("spark.driver.memory", driver_mem) \
        .config("spark.executor.cores", cores) \
        .getOrCreate()

    try:
        # Define paths
            
        # DRIFTED_DATA_PATH = "data/processed/titanic_preprocessed_drifted"
        # THRESHOLD = 0.1
        # BASELINE_STATS_PATH = "data/baseline_stats"
        # output_path = "report/drift_report.json"

        DRIFTED_DATA_PATH = params["drift"]["drift"]
        THRESHOLD = params["drift"]["threshold"]
        BASELINE_STATS_PATH = params["drift"]["baseline"]
        DRIFT_OUT = params["drift"]["output_path"]

        # Run drift detection on the drifted dataset
        drift_report = run_drift_detection(spark, DRIFTED_DATA_PATH, BASELINE_STATS_PATH, DRIFT_OUT, THRESHOLD)
        for feature, info in drift_report.items():
            print(f"{feature}: Drift Detected = {info['drift']}, Relative Difference = {info['relative_diff']:.3f}")

    finally:
        spark.stop()
