import sys
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create a Spark Session
spark = SparkSession.builder.appName("MNIST_LogisticRegressionClassifier").getOrCreate()

# Load the MNIST dataset
train_data_path = sys.argv[1]
test_data_path = sys.argv[2]
output_path = sys.argv[3]

train_data = spark.read.csv(train_data_path, inferSchema=True, header=True)
test_data = spark.read.csv(test_data_path, inferSchema=True, header=True)

assembler = VectorAssembler(inputCols=train_data.columns[1:], outputCol='features')
train_data_vec = assembler.transform(train_data)
test_data_vec = assembler.transform(test_data)

# Model
lr = LogisticRegression(featuresCol='features', labelCol='label', family='multinomial', maxIter=10)
param_grid = ParamGridBuilder(). \
    addGrid(lr.regParam, [0.01, 0.1, 1.0]). \
    addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]).build()
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
cross_val = CrossValidator(estimator=lr, evaluator=evaluator, estimatorParamMaps=param_grid, numFolds=3)

# Train the logistic model
model = cross_val.fit(train_data_vec)
train_pred = model.transform(train_data_vec)

# Predict values for the test dataset
test_pred = model.transform(test_data_vec)

# Evaluate the performance
train_accuracy = evaluator.evaluate(train_pred)
test_accuracy = evaluator.evaluate(test_pred)
param_tuning_result = model.avgMetrics

print('Train accuracy:', train_accuracy)
print('Test accuracy:', test_accuracy)

for i, result in enumerate(param_tuning_result):
    print('Parameters:', param_grid[i])
    print('Accuracy:', '{:.2f}'.format(result))

# Output results
output_csv = test_pred.select(['label', 'prediction'] + train_data.columns[1:])
output_csv = output_csv.repartition(1)
output_csv.write.csv(output_path, header=True, mode='overwrite')

textual_output = spark.createDataFrame(['Train accuracy: ' + '{:.2f}'.format(train_accuracy), \
                                        'Test accuracy: ' + '{:.2f}'.format(test_accuracy)], 'string')
textual_output = textual_output.repartition(1)
textual_output.write.text(output_path + 'accuracy')

# Stop the Spark session
spark.stop()