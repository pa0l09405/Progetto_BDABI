#from itertools import zip_longest
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
from collections import Counter
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.feature import StringIndexer
#from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from sklearn import metrics
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.multiclass import unique_labels
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import csv
import en_core_web_sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import seaborn as sns
import spacy 
import xlrd
import pandas
import pymongo

from pymongo import MongoClient
from pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

if __name__=='__main__':
    spark = SparkSession.builder.appName("CountVectorizer").config("spark.executor.memory", '48G') \
    .config("spark.driver.memory", '48G').getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client.fake_and_real_news_dataset
    collection = db.text_cleaned
    #print(collection)
    df = DataFrame(collection.find())
    df = df.drop_duplicates()
    #print(df)
    
    #print(df.show(4))
    nlp = English()
    #text=df.rdd.map(lambda x: x.text).collect()
    #type_col=df.rdd.map(lambda x: x.type).collect()

    text = df.text.tolist()
    type_col = df.type.tolist()

    for i in range(0,len(text)):
        text[i]=text[i].lower().split(" ")
    #print(text)
    #text="When I was young Alfredo"
    #spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    df_nuova = spark.createDataFrame(zip(text, type_col), schema=['text', 'type'])
    print("Dataframe spark")
    print(df_nuova.show(20))

    #y = df.type
    #df.drop("type", axis=1)
    #df['text']=pd.Series(filtered_text)
    #print(df['text'])

    #Split train e test
    train = df_nuova.sampleBy("type", fractions={'real': 0.8, 'fake': 0.8}, seed=2)
    test = df_nuova.exceptAll(train)
    #evaluator = MulticlassClassificationEvaluator(labelCol="type", predictionCol="prediction", metricName="accuracy")
    
    print("[CountVectorizer] Determino il miglior classificatore posto che vocabSize=256 e minDocFreq=5")
    #cv = CountVectorizer(inputCol="text", outputCol='count', vocabSize = 2**8)
    cv = CountVectorizer(inputCol="text", outputCol='features', vocabSize = 2**8)
    #idf = IDF(inputCol='count', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
	
    label_stringIdx = StringIndexer(inputCol = "type", outputCol = "label")
    '''
    classifier = NaiveBayes()
    pipeline = Pipeline(stages=[cv, idf, label_stringIdx, classifier])
    pipelineFit = pipeline.fit(train)
    predictions = pipelineFit.transform(test)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[CountVectorizer - NaiveBayes] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[CountVectorizer - NaiveBayes] ROC-AUC: {0:.4f}".format(roc_auc))
        
    label=[1.0,0.0]
    predictions_pandas = predictions.toPandas()
    lbl = predictions_pandas['label'].tolist()
    prd = predictions_pandas['prediction'].tolist()
    cm = confusion_matrix(lbl, prd, labels=label)
    #print("Confusion matrix ",cm)
    tp = cm[0][0]
    #print("TP ", tp)
    fp = cm[1][0]
    #print("FP ", fp)        
    precision = tp /(tp + fp)
    print("[CountVectorizer - NaiveBayes] Precision: {0:.4f}".format(precision))
    '''
    
    classifier = LogisticRegression()
    #pipeline = Pipeline(stages=[cv, idf, label_stringIdx, classifier])
    pipeline = Pipeline(stages=[cv, label_stringIdx, classifier])
    pipelineFit = pipeline.fit(train)
    predictions = pipelineFit.transform(test)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[CountVectorizer - LogisticRegression (parametri default)] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[CountVectorizer - LogisticRegression (parametri default)] ROC-AUC: {0:.4f}".format(roc_auc))
        
    label=[1.0,0.0]
    predictions_pandas = predictions.toPandas()
    lbl = predictions_pandas['label'].tolist()
    prd = predictions_pandas['prediction'].tolist()
    cm = confusion_matrix(lbl, prd, labels=label)
    #print("Confusion matrix ",cm)
    tp = cm[0][0]
    #print("TP ", tp)
    fp = cm[1][0]
    #print("FP ", fp)        
    precision = tp /(tp + fp)
    print("[CountVectorizer - LogisticRegression (parametri default)] Precision: {0:.4f}".format(precision))
    '''
    classifier = RandomForestClassifier()
    pipeline = Pipeline(stages=[cv, idf, label_stringIdx, classifier])
    pipelineFit = pipeline.fit(train)
    predictions = pipelineFit.transform(test)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[CountVectorizer - RandomForest] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[CountVectorizer - RandomForest] ROC-AUC: {0:.4f}".format(roc_auc))
        
    label=[1.0,0.0]
    predictions_pandas = predictions.toPandas()
    lbl = predictions_pandas['label'].tolist()
    prd = predictions_pandas['prediction'].tolist()
    cm = confusion_matrix(lbl, prd, labels=label)
    #print("Confusion matrix ",cm)
    tp = cm[0][0]
    #print("TP ", tp)
    fp = cm[1][0]
    #print("FP ", fp)        
    precision = tp /(tp + fp)
    print("[CountVectorizer - RandomForest] Precision: {0:.4f}".format(precision))
    
    classifier = DecisionTreeClassifier()
    pipeline = Pipeline(stages=[cv, idf, label_stringIdx, classifier])
    pipelineFit = pipeline.fit(train)
    predictions = pipelineFit.transform(test)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[CountVectorizer - DecisionTree] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[CountVectorizer - DecisionTree] ROC-AUC: {0:.4f}".format(roc_auc))
        
    label=[1.0,0.0]
    predictions_pandas = predictions.toPandas()
    lbl = predictions_pandas['label'].tolist()
    prd = predictions_pandas['prediction'].tolist()
    cm = confusion_matrix(lbl, prd, labels=label)
    #print("Confusion matrix ",cm)
    tp = cm[0][0]
    #print("TP ", tp)
    fp = cm[1][0]
    #print("FP ", fp)        
    precision = tp /(tp + fp)
    print("[CountVectorizer - DecisionTree] Precision: {0:.4f}".format(precision))
    
    classifier = LinearSVC()
    pipeline = Pipeline(stages=[cv, idf, label_stringIdx, classifier])
    pipelineFit = pipeline.fit(train)
    predictions = pipelineFit.transform(test)
   
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[CountVectorizer - LinearSVC] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[CountVectorizer - LinearSVC] ROC-AUC: {0:.4f}".format(roc_auc))
        
    label=[1.0,0.0]
    predictions_pandas = predictions.toPandas()
    lbl = predictions_pandas['label'].tolist()
    prd = predictions_pandas['prediction'].tolist()
    cm = confusion_matrix(lbl, prd, labels=label)
    #print("Confusion matrix ",cm)
    tp = cm[0][0]
    #print("TP ", tp)
    fp = cm[1][0]
    #print("FP ", fp)        
    precision = tp /(tp + fp)
    print("[CountVectorizer - LinearSVC] Precision: {0:.4f}".format(precision))
    '''
    '''
    #print("CF")
    count = CountVectorizer(inputCol="text", outputCol='count')
    count_fit = count.fit(train)
    count_transform = count_fit.transform(train)
    #print(count_transform.show(2, truncate = False))
    #print("IDF")
    idf = IDF(inputCol='count', outputCol="features")
    idf_fit = idf.fit(count_transform)
    idf_transform = idf_fit.transform(count_transform)
    #print(idf_transform.show(2, truncate = False))
    #print("label")
    label_stringIdx = StringIndexer(inputCol = "type", outputCol = "label")
    label_fit = label_stringIdx.fit(idf_transform)
    label_transform = label_fit.transform(idf_transform)
    #print(label_transform.show(2, truncate = False))
    '''
    
    '''
    nb = NaiveBayes(featuresCol = 'features', labelCol = 'label')
    #print(nb.explainParams())
    print("ciao")
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    paramGrid = (ParamGridBuilder()
             #.addGrid(nb.smoothing, [0.0, 0.5, 1.0])
             #.addGrid(nb.thresholds, [0.0, 0.5, 1.0])
             .addGrid(idf.minDocFreq, [2, 5, 7])
             .addGrid(count.vocabSize, [2**2, 2**4, 2**6, 2**8, 2**10, 2**12])
             .build())
    cv = CrossValidator(estimator=nb, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
    cvModel = cv.fit(label_transform) 
    print("[CountVectorizer] Selezione degli iperparametri ", cvModel.avgMetrics)
    print("[CountVectorizer] Modello migliore ", cvModel.bestModel)
    best_params = cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ]
    print("[CountVectorizer] Parametri migliori", best_params)
    '''
    '''
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label')
    print("ciao")
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    paramGrid = (ParamGridBuilder()
             #.addGrid(lr.maxIter, [10, 20, 50])
             #.addGrid(lr.regParam, [0, 0.3, 0.5])
             #.addGrid(lr.elasticNetParam, [0, 0.5, 1])
             .addGrid(idf.minDocFreq, [2, 5, 7])
             .addGrid(count.vocabSize, [2**2, 2**4])
             #.addGrid(count.vocabSize, [2**4, 2**6, 2**8, 2**10])
             #.addGrid(count.vocabSize, [2**8, 2**10, 2**12])
             .build())
    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5, collectSubModels=False)
    cvModel = cv.fit(label_transform) 
    print("[CountVectorizer - LogisticRegression] Selezione degli iperparametri ")
    print("[CountVectorizer - LogisticRegression] Modello migliore ", cvModel.bestModel)
    best_params = cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ]
    print("[CountVectorizer - LogisticRegression] Parametri migliori", best_params)
    
    #... 
    1: 2,4
    2: 2,8
    3: 2,6
    4: 5,4
    '''
    
    list_acc_2 = []
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label')
    paramGrid = ParamGridBuilder().build()
        
    for i in range (1,17):
        #print("CF")
        #count = CountVectorizer(inputCol="text", outputCol='count', vocabSize=2**i)
        count = CountVectorizer(inputCol="text", outputCol='features', vocabSize = 2**i)
        count_fit = count.fit(train)
        count_transform = count_fit.transform(train)
        #print(count_transform.show(2, truncate = False))
        #print("IDF")
        '''
        idf = IDF(inputCol='count', outputCol="features")
        idf_fit = idf.fit(count_transform)
        idf_transform = idf_fit.transform(count_transform)
        '''
        #print(idf_transform.show(2, truncate = False))
        #print("label")
        label_stringIdx = StringIndexer(inputCol = "type", outputCol = "label")
        '''
		label_fit = label_stringIdx.fit(idf_transform)
        label_transform = label_fit.transform(idf_transform)
        '''
        label_fit = label_stringIdx.fit(count_transform)
        label_transform = label_fit.transform(count_transform)
        #print(label_transform.show(2, truncate = False))
        
        cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
        cvModel = cv.fit(label_transform)
        print("    [CountVectorizer - LogisticRegression] Accuracy con vocabSize  = 2^",i, cvModel.avgMetrics)
        list_acc_2.append(cvModel.avgMetrics)
            
    vocabsize = list_acc_2.index(max(list_acc_2))+1
    print("[CountVectorizer - LogisticRegression] vocabSize = ", 2**vocabsize)
    
    list_acc_2 = []
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label')
    paramGrid = ParamGridBuilder().build()
        
    for i in range (1,17):
        #print("CF")
        #count = CountVectorizer(inputCol="text", outputCol='count', vocabSize=2**i)
        count = CountVectorizer(inputCol="text", outputCol='features', vocabSize = 2**i)
        count_fit = count.fit(train)
        count_transform = count_fit.transform(train)
        #print(count_transform.show(2, truncate = False))
        #print("IDF")
        '''
        idf = IDF(inputCol='count', outputCol="features")
        idf_fit = idf.fit(count_transform)
        idf_transform = idf_fit.transform(count_transform)
        '''
        #print(idf_transform.show(2, truncate = False))
        #print("label")
        label_stringIdx = StringIndexer(inputCol = "type", outputCol = "label")
        '''
		label_fit = label_stringIdx.fit(idf_transform)
        label_transform = label_fit.transform(idf_transform)
        '''
        label_fit = label_stringIdx.fit(count_transform)
        label_transform = label_fit.transform(count_transform)
        #print(label_transform.show(2, truncate = False))
        
        cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
        cvModel = cv.fit(label_transform)
        print("    [CountVectorizer - LogisticRegression] Accuracy con minDocFreq  =",i, cvModel.avgMetrics)
        list_acc_2.append(cvModel.avgMetrics)
            
    mindocfreq = list_acc_2.index(max(list_acc_2))+1
    print("[CountVectorizer - LogisticRegression] minDocFreq = ", mindocfreq)
    
    
    #print("CF")
    count = CountVectorizer(inputCol="text", outputCol='count', vocabSize=2**vocabsize)
    count_fit = count.fit(train)
    count_transform = count_fit.transform(test)
    #print(count_transform.show(2, truncate = False))
    #print("IDF")
    idf = IDF(inputCol='count', outputCol="features")
    idf_fit = idf.fit(count_transform)
    idf_transform = idf_fit.transform(count_transform, minDocFreq=mindocfreq)
    #print(idf_transform.show(2, truncate = False))
    #print("label")
    label_stringIdx = StringIndexer(inputCol = "type", outputCol = "label")
    label_fit = label_stringIdx.fit(idf_transform)
    label_transform = label_fit.transform(idf_transform)
    #print(label_transform.show(2, truncate = False))
    
    predictions = cvModel.bestModel.transform(label_transform)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[CountVectorizer - parametri migliori del CV - LogisticRegression] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[CountVectorizer - parametri migliori del CV - LogisticRegression] ROC-AUC: {0:.4f}".format(roc_auc))
        
    label=[1.0,0.0]
    predictions_pandas = predictions.toPandas()
    lbl = predictions_pandas['label'].tolist()
    prd = predictions_pandas['prediction'].tolist()
    cm = confusion_matrix(lbl, prd, labels=label)
    #print("Confusion matrix ",cm)
    tp = cm[0][0]
    #print("TP ", tp)
    fp = cm[1][0]
    #print("FP ", fp)        
    precision = tp /(tp + fp)
    print("[CountVectorizer - parametri migliori del CV - LogisticRegression] Precision: {0:.4f}".format(precision))

    
    
    '''
    evaluator = BinaryClassificationEvaluator()
    cv = CountVectorizer(inputCol="text", outputCol='cv', vocabSize=2**16)
    idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
    label_stringIdx = StringIndexer(inputCol = "type", outputCol = "label")
    #classifier = LinearSVC(maxIter=10, regParam=0.1)
    #classifier = NaiveBayes(modelType="multinomial")
    classifier = LogisticRegression(maxIter=100)
    #classifier = RandomForestClassifier(numTrees=10)
    pipeline = Pipeline(stages=[cv, idf, label_stringIdx, classifier])

    (train_set, test_set) = df_nuova.randomSplit([0.8, 0.2])
    pipelineFit = pipeline.fit(train_set)
    predictions = pipelineFit.transform(test_set)
    pred=predictions.rdd.map(lambda x: x.prediction).collect()
    print(type(pred))
    print(pred[0])
    etichetta = test_set.rdd.map(lambda x: x.type).collect()
    print(etichetta[0])
    label_true=predictions.rdd.map(lambda x: x.label).collect()
    print(type(label_true))
    print(label_true[0])
    accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test_set.count())
    roc_auc = evaluator.evaluate(predictions)

    print ("Accuracy Score: {0:.4f}".format(accuracy))
    print ("ROC-AUC: {0:.4f}".format(roc_auc))
    '''
    
    '''
    # Make training and test sets 
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.2, random_state=random.randint(0,100))
        
    
    (trainingData, testData) = df.randomSplit([0.8, 0.2])
    # Initialize the `count_vectorizer` 
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data 
    count_train = count_vectorizer.fit_transform(trainingData['text']) 
    print(count_train)

    # Transform the test set 
    count_test = count_vectorizer.transform(testData['text'])
    print(count_test)
    
        
    # Initialize the `count_vectorizer` 
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data 
    count_train = count_vectorizer.fit_transform(X_train) 
    print(count_train)

    # Transform the test set 
    count_test = count_vectorizer.transform(X_test)
    '''

    '''
    # Initialize the `tfidf_vectorizer` 
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) 

    # Fit and transform the training data 
    tfidf_train = tfidf_vectorizer.fit_transform(X_train) 

    # Transform the test set 
    tfidf_test = tfidf_vectorizer.transform(X_test)
    '''

    '''
    # Get the feature names of `count_vectorizer` 
    #print(count_vectorizer.get_feature_names())
    #print(tfidf_vectorizer.get_feature_names())
    
    # Train a RandomForest model.
    #  Empty categoricalFeaturesInfo indicates all features are continuous.
    #  Note: Use larger numTrees in practice.
    #  Setting featureSubsetStrategy="auto" lets the algorithm choose.

    # Evaluate model on test instances and compute test error
    predictions = model.predict(count_test.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(
    print('Test Error = ' + str(testErr))
    print('Learned classification forest model:')
    print(model.toDebugString())
    
    
    #clf = MultinomialNB() 
    #clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(count_train, y_train)
    #clf.fit(tfidf_train, y_train)
    #print(clf.coef_.shape[-1])
    pred = clf.predict(count_test)
    #pred = clf.predict(tfidf_test)    
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy" ,i,  score)
    '''
    '''
    label=[1.0,0.0]
    cm = confusion_matrix(label_true, pred, labels=label)
    print(cm)

    plt.cfg()
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d"); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['real','fake']); ax.yaxis.set_ticklabels(['real','fake'])
    
        
    plt.savefig('/home/vmadmin/bigdata/confusion_count_vectorizer.pdf', dpi=300, bbox_inches="tight")
    '''