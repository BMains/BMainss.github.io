---
title: "Logistic Regression with Apache Spark"
date: 2020-11-16
tags: [Carriege return, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Carriage returns, Data Science, Messy Data"
mathjax: "true"
---

A brief introduction to Apache Spark.It's a lightning fast tool used for big data processing,sql,streaming,machine learning and graph processing.

Spark is usually very troublesome when it comes to using small datasets but this particular version of spark is in a small free trial cluster in an AWS EC2 instance


```python
import findspark
findspark.init('/home/ubuntu/spark-3.0.1-bin-hadoop2.7')
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
spark = SparkSession.builder.appName('Mylogregexcer').getOrCreate()
df = spark.read.csv('titanic.csv',inferSchema = True,header = True)
```


```python
df.printSchema()
```

    root
     |-- PassengerId: integer (nullable = true)
     |-- Survived: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- SibSp: integer (nullable = true)
     |-- Parch: integer (nullable = true)
     |-- Ticket: string (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = true)
    
    


```python
df.columns
```




    ['PassengerId',
     'Survived',
     'Pclass',
     'Name',
     'Sex',
     'Age',
     'SibSp',
     'Parch',
     'Ticket',
     'Fare',
     'Cabin',
     'Embarked']




```python
my_cols = df.select(['Survived',
 'Pclass',
 'Sex',
 'Age',
 'SibSp',
 'Parch',
 'Ticket',
 'Fare',
 'Embarked'])
```


```python
my_final_data = my_cols.na.drop()
```


```python
from pyspark.ml.feature import (VectorAssembler,VectorIndexer,OneHotEncoder,StringIndexer)
```


```python
gender_indexer = StringIndexer(inputCol="Sex",outputCol='SexIndex')
gender_encoder = OneHotEncoder(inputCol="SexIndex",outputCol="SexVec")
```


```python
embark_indexer = StringIndexer(inputCol='Embarked',outputCol='EmbarkIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkIndex',outputCol='EmbarkVec')
```


```python
assembler = VectorAssembler (inputCols=['Pclass','SexVec','EmbarkVec','Age','SibSp','Parch','Fare'],
                            outputCol = 'features')
```


```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
```


```python
log_reg_titanic = LogisticRegression(featuresCol='features',labelCol='Survived')
```


```python
pipeline = Pipeline(stages=[gender_indexer,embark_indexer,
                           gender_encoder,embark_encoder,
                           assembler,log_reg_titanic])
```


```python
train_data , test_data = my_final_data.randomSplit([0.7,0.3])
```


```python
fit_model = pipeline.fit(train_data)
```


```python
results = fit_model.transform(test_data)
```


```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```


```python
my_eval = BinaryClassificationEvaluator(rawPredictionCol = 'prediction',labelCol='Survived')
```


```python
results.select('Survived','prediction').show()
```

    +--------+----------+
    |Survived|prediction|
    +--------+----------+
    |       0|       1.0|
    |       0|       1.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       1.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    |       0|       0.0|
    +--------+----------+
    only showing top 20 rows
    
    


```python
AUC = my_eval.evaluate(results)
```


```python
AUC
```




    0.766553480475382




```python

```
