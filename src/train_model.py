# src/train_model.py

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

from pyspark.sql import SparkSession

import yaml
import logging
import sys

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
import os, time

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(spark, input_path, model_output_path):

    start_time = time.time()
    report_dir = "report"

    try:
            
        df = spark.read.parquet(input_path)
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

        # Define feature columns
        feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "SexIndexed", "EmbarkedIndexed"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        models = {
            "lr": LogisticRegression(featuresCol="features", labelCol="Survived"),
            "rf": RandomForestClassifier(featuresCol="features", labelCol="Survived"),
            "gbt": GBTClassifier(featuresCol="features", labelCol="Survived")
        }

        param_grids = {
            "lr": ParamGridBuilder() \
                .addGrid(models["lr"].regParam, [0.01, 0.1]) \
                .addGrid(models["lr"].elasticNetParam, [0.0, 0.5]) \
                .build(),
            "rf": ParamGridBuilder() \
                .addGrid(models["rf"].numTrees, [50, 100]) \
                .addGrid(models["rf"].maxDepth, [5, 10]) \
                .build(),
            "gbt": ParamGridBuilder() \
                .addGrid(models["gbt"].maxIter, [10, 20]) \
                .addGrid(models["gbt"].maxDepth, [3, 5]) \
                .build()
        }

        evaluator = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC")

        best_model = None
        best_auc = 0.0
        best_model_name = None
        best_model_run_id = None

        # Start MLflow tracking
        mlflow.set_experiment("Titanic-Classification")

        for name, model in models.items():
            pipeline = Pipeline(stages=[assembler, model])
            param_grid = param_grids[name]

            crossval = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=5,
                parallelism=2
            )
        

            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run(run_name=f"{name}_model") as run:

                run_id = run.info.run_id
                exp_id = run.info.experiment_id

                print(f"Training model: {name}...")
                print(f"Experiment ID: {exp_id}")
                print(f"Run ID: {run_id}")

                # Train
                # model = tvs.fit(train_df)
                model = crossval.fit(train_df)
                predictions = model.transform(test_df)

                auc = evaluator.evaluate(predictions)

                # Convert predictions to Pandas
                preds_pd = predictions.select("prediction", "Survived").toPandas()
                y_true = preds_pd["Survived"]
                y_pred = preds_pd["prediction"]
                
                # Compute metrics
                cm = confusion_matrix(y_true, y_pred)
                report_cls = classification_report(y_true, y_pred, output_dict=True)

                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)

                param_map = model.bestModel.stages[-1].extractParamMap()
                best_params = {param.name: value for param, value in param_map.items()}

                for param, val in best_params.items():
                    mlflow.log_param(param, val)

                mlflow.log_param("cv_folds", 5)

                # Log Spark configs
                conf = spark.sparkContext.getConf()
                mlflow.log_param("executor_memory", conf.get("spark.executor.memory", "default"))
                mlflow.log_param("executor_cores", conf.get("spark.executor.cores", "default"))
                mlflow.log_param("driver_memory", conf.get("spark.driver.memory", "default"))

                # Log metrics
                mlflow.log_metric("AUC", auc)
                mlflow.log_metric("Accuracy", acc)
                mlflow.log_metric("Precision", prec)
                mlflow.log_metric("Recall", rec)
                mlflow.log_metric("F1_score", f1)

                mlflow.set_tag("spark.executor_memory", params["spark"]["executor_memory"])
                mlflow.set_tag("spark.driver_memory", params["spark"]["driver_memory"])
                mlflow.set_tag("spark.executor_cores", params["spark"]["executor_cores"])
                mlflow.set_tag("parallelism", params["spark"]["parallelism"])

                # Save and log confusion matrix
                cm_df = pd.DataFrame(cm)
                cm_path = os.path.join(report_dir,"confusion_matrix.csv")
                cm_df.to_csv(cm_path, index=False)
                mlflow.log_artifact(cm_path)

                # Save classification report
                report_df = pd.DataFrame(report_cls).transpose()
                report_path = os.path.join(report_dir,"classification_report.csv")
                report_df.to_csv(report_path)
                mlflow.log_artifact(report_path)


                # Log confusion matrix plot
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")
                plt.tight_layout()

                conf_matrix_path = os.path.join(report_dir,"confusion_matrix.png")
                plt.savefig(conf_matrix_path)
                mlflow.log_artifact(conf_matrix_path)
                plt.close()

                # Log the Spark model
                mlflow.spark.log_model(model.bestModel, artifact_path="model")

                print(f"{name} model AUC: {auc:.4f}")

                if auc > best_auc:
                    best_auc = auc
                    best_model = model.bestModel
                    best_model_name = name
                    best_model_run_id = run_id
                    best_metrics = {
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1_score": f1,
                        "AUC": auc
                    }

        # Save best model locally
        best_model.write().overwrite().save(model_output_path)

        # Register best model in MLflow
        result = mlflow.register_model(
            model_uri=f"runs:/{best_model_run_id}/model",
            name="TitanicSparkBestModel"
        )

        # Transition to Production
        client = MlflowClient()
        client.transition_model_version_stage(
            name=result.name,
            version=result.version,
            stage="Production",
            archive_existing_versions=True
        )

        print(f"Model {result.name} v{result.version} transitioned to 'Production'.")

        with mlflow.start_run(run_name="best_model_registration") as final_run:
            duration = time.time() - start_time
            mlflow.log_metric("training_time_seconds", duration)

        print(f"Best model: {best_model_name} with AUC: {best_auc:.4f}")
        print(f"Model registered: {result.name}, version {result.version}")
        print(f"Training time: {duration:.2f} seconds")
        print(f"Other metrics: {best_metrics}")

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    return exp_id, best_model_run_id

if __name__ == "__main__":

    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    executor_mem = params["spark"]["executor_memory"]
    driver_mem = params["spark"]["driver_memory"]
    cores = params["spark"]["executor_cores"]

    # Start Spark session
    spark = SparkSession.builder \
        .appName("TrainTitanic") \
        .config("spark.executor.memory", executor_mem) \
        .config("spark.driver.memory", driver_mem) \
        .config("spark.executor.cores", cores) \
        .getOrCreate()

    try:
        # Define paths
            
        # PROCESSED_PATH = "data/processed/titanic_preprocessed/processed_train.parquet"
        # MODEL_PATH = os.path.expanduser("~/mlops_workspace/models/logistic_model")
        MODEL_PATH = params["train"]["model_output"]
        PROCESSED_PATH = params["train"]["input_data"]

        # Train model
        exp_id, run_id = train_model(spark, PROCESSED_PATH, MODEL_PATH)

        print("Model training complete.")
        print(f"Experiment ID and Run ID to be used while running api_server.py is: {exp_id}, {run_id}")

    finally:
        spark.stop()
