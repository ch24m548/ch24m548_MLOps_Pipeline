# src/train_model.py

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

def train_model(input_path, model_output_path):
    spark = SparkSession.builder \
        .appName("TitanicModelTraining") \
        .getOrCreate()

    # Load preprocessed data
    df = spark.read.parquet(input_path)

    # Split into train/test
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Define model
    lr = LogisticRegression(featuresCol="features", labelCol="Survived")

    # Hyperparameter grid (optional tuning)
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
        .build()

    # Setup TrainValidationSplit
    tvs = TrainValidationSplit(
        estimator=lr,
        estimatorParamMaps=paramGrid,
        evaluator=BinaryClassificationEvaluator(labelCol="Survived"),
        trainRatio=0.8
    )

    # Train model
    model = tvs.fit(train_df)

    # Evaluate on test set
    predictions = model.transform(test_df)
    evaluator = BinaryClassificationEvaluator(labelCol="Survived")
    auc = evaluator.evaluate(predictions)
    print(f"✅ Test AUC: {auc:.4f}")

    # Save best model
    model.bestModel.save(model_output_path)

    spark.stop()
