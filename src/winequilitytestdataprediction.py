import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

def clean_dataframe(dataframe):
    return dataframe.select(*(col(column).cast("double").alias(column.strip("\"")) for column in dataframe.columns))

if __name__ == "__main__":
    spark_session = SparkSession.builder \
        .appName('WinePrediction') \
        .getOrCreate()
    spark_context = spark_session.sparkContext
    spark_context.setLogLevel('ERROR')

    if len(sys.argv) > 3:
        sys.exit(-1)
    elif len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        if not ("/" in input_path):
            input_path = "data/csv/" + input_path
        model_path = "/code/data/model/testdata.model"
        print("Test data for Input file ")
        print(input_path)
    else:
        current_dir = os.getcwd() 
        print(current_dir)
        input_path = os.path.join(current_dir, "Pa2Winepred\data\csv\testdata.csv")
        model_path = os.path.join(current_dir, "Pa2Winepred\data\model\testdata.model")

    dataframe = (spark_session.read
                .format("csv")
                .option('header', 'true')
                .option("sep", ";")
                .option("inferschema", 'true')
                .load(input_path))

    cleaned_dataframe = clean_dataframe(dataframe)
    selected_features = ['fixed acidity',
                        'volatile acidity',
                        'citric acid',
                        'chlorides',
                        'total sulfur dioxide',
                        'density',
                        'sulphates',
                        'alcohol',
                    ]
    trained_model = PipelineModel.load(model_path)
    predictions = trained_model.transform(cleaned_dataframe)
    print(predictions.show(5))
    results = predictions.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(
                                        labelCol='label', 
                                        predictionCol='prediction', 
                                        metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    print('Wine prediction model Test Accuracy = ', accuracy)
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Wine prediction model for Weighted f1 score = ', metrics.weightedFMeasure())
    sys.exit(0)
