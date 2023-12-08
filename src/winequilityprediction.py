import sys
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def clean_dataframe(dataframe):
    return dataframe.select(*(col(column).cast("double").alias(column.strip("\"")) for column in dataframe.columns))

if __name__ == "__main__":
    
    spark_session = SparkSession.builder \
        .appName('WineQualityPrediction') \
        .getOrCreate()

    spark_context = spark_session.sparkContext
    spark_context.setLogLevel('ERROR')

    if len(sys.argv) != 3:
        sys.exit(-1)

    input_path = sys.argv[1]
    valid_path = sys.argv[2]
    output_model_path = "s3://wine_quality_bucket/best_model.model"

    training_dataframe = (spark_session.read
                          .format("csv")
                          .option('header', 'true')
                          .option("sep", ";")
                          .option("inferschema", 'true')
                          .load(input_path))
    
    training_data = clean_dataframe(training_dataframe)

    validation_dataframe = (spark_session.read
                            .format("csv")
                            .option('header', 'true')
                            .option("sep", ";")
                            .option("inferschema", 'true')
                            .load(valid_path))
    
    validation_data = clean_dataframe(validation_dataframe)

    feature_columns = ['fixed acidity',
                        'volatile acidity',
                        'citric acid',
                        'residual sugar',
                        'chlorides',
                        'free sulfur dioxide',
                        'total sulfur dioxide',
                        'density',
                        'pH',
                        'sulphates',
                        'alcohol',
                        'quality',
                    ]
    feature_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    
    label_indexer = StringIndexer(inputCol="quality", outputCol="label")

    training_data.cache()
    validation_data.cache()
    
    random_forest_classifier = RandomForestClassifier(labelCol='label', featuresCol='features',numTrees=150, maxBins=8, maxDepth=15, seed=150, impurity='gini')
    
    pipeline = Pipeline(stages=[feature_assembler, label_indexer, random_forest_classifier])
    model = pipeline.fit(training_data)

    predictions = model.transform(validation_data)

    results = predictions.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(labelCol='label', 
                                                  predictionCol='prediction', 
                                                  metricName='accuracy')

    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy of wine prediction model = ', accuracy)
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted f1 score of wine prediction model = ', metrics.weightedFMeasure())
 
    cv_model = None
    param_grid = ParamGridBuilder() \
            .addGrid(random_forest_classifier.maxBins, [9, 8, 4])\
            .addGrid(random_forest_classifier.maxDepth, [25, 6 , 9])\
            .addGrid(random_forest_classifier.numTrees, [500, 50, 150])\
            .addGrid(random_forest_classifier.minInstancesPerNode, [6])\
            .addGrid(random_forest_classifier.seed, [100, 200, 5043, 1000])\
            .addGrid(random_forest_classifier.impurity, ["entropy","gini"])\
            .build()
    pipeline = Pipeline(stages=[feature_assembler, label_indexer, random_forest_classifier])
    cross_validator = CrossValidator(estimator=pipeline,
                                    estimatorParamMaps=param_grid,
                                    evaluator=evaluator,
                                    numFolds=2)

    cv_model = cross_validator.fit(training_data)
    
    best_model = cv_model.bestModel
    print(best_model)
    
    predictions = best_model.transform(validation_data)
    results = predictions.select(['prediction', 'label'])
    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy of wine prediction model = ', accuracy)
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted f1 score of wine prediction model = ', metrics.weightedFMeasure())

    best_model_path = output_model_path
    best_model.write().overwrite().save(best_model_path)
    sys.exit(0)
