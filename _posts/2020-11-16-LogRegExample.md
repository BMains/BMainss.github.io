---
title: "Logistic Regression with Apache Spark"
date: 2020-09-16
tags: [Logistic Regression, Machine Learning, Spark]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Logistic Regression, Machine Learning, Spark"
mathjax: "true"
---

A brief introduction to Apache Spark,It's a lightning fast tool used for big data processing,sql,streaming,machine learning and graph processing.

Spark is usually very troublesome when it comes to using small datasets but this particular version of spark is in a small free trial cluster in an AWS EC2 instance so we shouldn't worry too much about it.


## Load Data

In this section we load the data,similar to pandas when reading from a csv format

***Findspark***- This allows you to call on Spark as if it were a regular library,since spark itself isn't on the sys.path by default.Also note that findspark can be configured to edit the bashrc configuration file and set the environment variables permanently and only run the files once.

***InferSchema***-This automatically predicts the data type for each column.

***SparkSession***-SparkSession allows for creating a DataFrame,creating a Dataset, accessing the Spark SQL services.Which then allows us to import ***LogisticRegression***

```python
import findspark
findspark.init('/home/ubuntu/spark-3.0.1-bin-hadoop2.7')
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
spark = SparkSession.builder.appName('Mylogregexcer').getOrCreate()
df = spark.read.csv('titanic.csv',inferSchema = True,header = True)
```
## Data Dictionary

Data Dictionary with all the column names

| Variable    | Definition                                 | Key                                            |
|-------------|--------------------------------------------|------------------------------------------------|
| PassengerId | IdentityID                                 |                                                |
| survival    | Survival                                   | 0 = No, 1 = Yes                                |
| pclass      | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| sex         | Sex                                        |                                                |
| Name        | Name of Passenger                          |                                                |
| Age         | Age in years                               |                                                |
| sibsp       | # of siblings / spouses aboard the Titanic |                                                |
| parch       | # of parents / children aboard the Titanic |                                                |
| ticket      | Ticket number                              |                                                |
| fare        | Passenger fare                             |                                                |
| cabin       | Cabin number                               |                                                |
| embarked    | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |

***PrintSchema***-Shows the data types for each column

```python
df.printSchema()

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
    
    df.columns

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


```
## Drop missing data

In this section,only the useful columns are selected from the dataframe and drop any rows with null values.
(Note: This is a very extreme way of dealing with missing data.The best practice would be fill in the data in some sort of fashion)

```pythona
my_cols = df.select(['Survived',
 'Pclass',
 'Sex',
 'Age',
 'SibSp',
 'Parch',
 'Ticket',
 'Fare',
 'Embarked'])

 my_final_data = my_cols.na.drop()

```
## Transform Categorical data to numeric values
Next,we need to convert columns that have string into numerical values.For example,the sex column only has two values (M and F).

**stringindexer** will change it into 0 and 1,since it's  nominal data.

**OneHotEncoder** converts the string into a vector.[Here is an example](https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/) . After converting the Sex column,we do the same for the Embark column.


```python
from pyspark.ml.feature import (VectorAssembler,VectorIndexer,OneHotEncoder,StringIndexer)

gender_indexer = StringIndexer(inputCol="Sex",outputCol='SexIndex')
gender_encoder = OneHotEncoder(inputCol="SexIndex",outputCol="SexVec")

embark_indexer = StringIndexer(inputCol='Embarked',outputCol='EmbarkIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkIndex',outputCol='EmbarkVec')

```

**VectorAssembler** - converts the columns into vectors,allows our model to use categorical data

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
