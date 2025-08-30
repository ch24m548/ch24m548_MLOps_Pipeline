from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder.appName("ValidateTestData").getOrCreate()

# Path to your processed test data
TEST_PARQUET_PATH = "data/processed/titanic_preprocessed/processed_test.parquet/"


# Expected input columns for the model
expected_columns = [
    "Pclass", "Age", "SibSp", "Parch", "Fare", "SexIndexed", "EmbarkedIndexed"
]

def main():
    print("🔍 Loading test data...")
    df = spark.read.parquet(TEST_PARQUET_PATH)

    print("\n📋 Schema of the DataFrame:")
    df.printSchema()

    print("\n🔎 Preview of the data:")
    df.show(5)

    print("\n✅ Checking required columns...")
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
    else:
        print("✅ All expected columns are present.")

    print("\n⚠️ Checking for null values in expected columns...")
    df.select([col(c).isNull().alias(c) for c in expected_columns]).summary().show()

    null_counts = df.select([col(c).isNull().cast("int").alias(c) for c in expected_columns]).groupBy().sum().collect()[0].asDict()
    for col_name, null_count in null_counts.items():
        if null_count > 0:
            print(f"❌ Column '{col_name}' has {null_count} null values")
        else:
            print(f"✅ Column '{col_name}' has no nulls")

    print("\n🧪 Testing VectorAssembler compatibility...")
    try:
        assembler = VectorAssembler(
            inputCols=expected_columns,
            outputCol="features"
        )
        assembled_df = assembler.transform(df)
        print("✅ VectorAssembler successfully applied.")
        assembled_df.select("features").show(5)
    except Exception as e:
        print("❌ VectorAssembler failed:")
        print(e)

if __name__ == "__main__":
    main()
