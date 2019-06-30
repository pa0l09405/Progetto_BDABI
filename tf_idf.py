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
#from spark_stratifier import StratifiedCrossValidator
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
    db = client.dataset_fake_and_real_news
    collection = db.text_as_list_of_words
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
        text[i]=text[i].split(" ")
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
    
    print("[TF-IDF] Determino il miglior classificatore posto che vocabSize=256 e minDocFreq=1")
    cv = CountVectorizer(inputCol="text", outputCol='count', vocabSize = 2**8, minDF=1.0)
    idf = IDF(inputCol='count', outputCol="features") #minDocFreq: remove sparse terms
	
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
        
    print ("[TF-IDF - NaiveBayes (parametri default)] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[TF-IDF - NaiveBayes (parametri default)] ROC-AUC: {0:.4f}".format(roc_auc))
        
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
    print("[TF-IDF - NaiveBayes (parametri default)] Precision: {0:.4f}".format(precision))
    
    classifier = LogisticRegression()
    pipeline = Pipeline(stages=[cv, idf, label_stringIdx, classifier])
    pipelineFit = pipeline.fit(train)
    predictions = pipelineFit.transform(test)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[TF-IDF - LogisticRegression (parametri default)] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[TF-IDF - LogisticRegression (parametri default)] ROC-AUC: {0:.4f}".format(roc_auc))
        
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
    print("[TF-IDF - LogisticRegression (parametri default)] Precision: {0:.4f}".format(precision))
    
    classifier = RandomForestClassifier()
    pipeline = Pipeline(stages=[cv, idf, label_stringIdx, classifier])
    pipelineFit = pipeline.fit(train)
    predictions = pipelineFit.transform(test)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[TF-IDF - RandomForest (parametri default)] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[TF-IDF - RandomForest (parametri default)] ROC-AUC: {0:.4f}".format(roc_auc))
        
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
    print("[TF-IDF - RandomForest (parametri default)] Precision: {0:.4f}".format(precision))
    
    classifier = DecisionTreeClassifier()
    pipeline = Pipeline(stages=[cv, idf, label_stringIdx, classifier])
    pipelineFit = pipeline.fit(train)
    predictions = pipelineFit.transform(test)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[TF-IDF - DecisionTree (parametri default)] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[TF-IDF - DecisionTree (parametri default)] ROC-AUC: {0:.4f}".format(roc_auc))
        
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
    print("[TF-IDF - DecisionTree (parametri default)] Precision: {0:.4f}".format(precision))
    
    
    classifier = LinearSVC()
    pipeline = Pipeline(stages=[cv, idf, label_stringIdx, classifier])
    pipelineFit = pipeline.fit(train)
    predictions = pipelineFit.transform(test)
   
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[TF-IDF - LinearSVC (parametri default)] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[TF-IDF - LinearSVC (parametri default)] ROC-AUC: {0:.4f}".format(roc_auc))
        
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
    print("[TF-IDF - LinearSVC (parametri default)] Precision: {0:.4f}".format(precision))
    '''
     
     
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    cv = CountVectorizer(inputCol="text", outputCol='count', vocabSize = 2**8, minDF=1.0)
    idf = IDF(inputCol='count', outputCol="features")
    paramGrid = ParamGridBuilder().build()
    label_stringIdx = StringIndexer(inputCol = "type", outputCol = "label")
    pipeline = Pipeline(stages=[cv, idf, label_stringIdx])
    pipelineFit = pipeline.fit(train)
    pipelineTransform = pipelineFit.transform(train)

    classifier = NaiveBayes(featuresCol = 'features', labelCol = 'label')
    xv = CrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = xv.fit(pipelineTransform)
    accuracy = cvModel.avgMetrics
    print("[TF-IDF - NaiveBayes (parametri default)] Accuracy:", accuracy)
    classifier = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label')
    xv = CrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = xv.fit(pipelineTransform)
    accuracy = cvModel.avgMetrics
    print("[TF-IDF - DecisionTree (parametri default)] Accuracy:",accuracy)
    classifier = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
    xv = CrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = xv.fit(pipelineTransform)
    accuracy = cvModel.avgMetrics
    print("[TF-IDF - RandomForest (parametri default)] Accuracy:",accuracy)
    classifier = LogisticRegression(featuresCol = 'features', labelCol = 'label')
    xv = CrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = xv.fit(pipelineTransform)
    accuracy = cvModel.avgMetrics
    print("[TF-IDF - LogisticRegression (parametri default)] Accuracy:",accuracy)
    classifier = LinearSVC(featuresCol = 'features', labelCol = 'label')
    xv = CrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = xv.fit(pipelineTransform)
    accuracy = cvModel.avgMetrics
    print("[TF-IDF - LinearSVC (parametri default)] Accuracy:",accuracy)
     
     
     
     
     
     
     
     
     
	 
    '''
    #Logistic Regressor
    
    
    list_acc_2 = []
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label')
    paramGrid = ParamGridBuilder().build()
     
    for i in range (1,16):
        #print("CF")
        count = CountVectorizer(inputCol="text", outputCol='count', vocabSize=2**i)
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
        
        cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
        cvModel = cv.fit(label_transform)
        print("    [TF-IDF - LogisticRegression] Accuracy con vocabSize  = 2^",i, cvModel.avgMetrics)
        list_acc_2.append(cvModel.avgMetrics)
            
    #vocabsize = 2**(list_acc_2.index(max(list_acc_2))+1)
    vocabsize = 2**(list_acc_2.index(max(list_acc_2))+11)
    print("[TF-IDF - LogisticRegression] vocabSize = ", vocabsize)
    
    list_acc_2 = []
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label')
    paramGrid = ParamGridBuilder().build()

    for i in range (0,3):
        #print("CF")
        count = CountVectorizer(inputCol="text", outputCol='count', vocabSize=vocabsize, minDF=0.05*i)
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
        
        cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
        cvModel = cv.fit(label_transform)
        print("    [TF-IDF - LogisticRegression] Accuracy con minDocFreq  = 0.",i, cvModel.avgMetrics)
        list_acc_2.append(cvModel.avgMetrics)
            
    mindocfreq = 0.05*(list_acc_2.index(max(list_acc_2)))
    print("[TF-IDF - LogisticRegression] minDocFreq = ", mindocfreq)

    
    #print("CF")
    count = CountVectorizer(inputCol="text", outputCol='count', vocabSize=vocabsize, minDF=mindocfreq)
    count_fit = count.fit(train)
    count_transform = count_fit.transform(train)
    count_transform_test = count_fit.transform(test)
    #print(count_transform.show(2, truncate = False))
    #print("IDF")
    idf = IDF(inputCol='count', outputCol="features")
    idf_fit = idf.fit(count_transform)
    idf_transform = idf_fit.transform(count_transform)
    idf_transform_test = idf_fit.transform(count_transform_test)
    #print(idf_transform.show(2, truncate = False))
    #print("label")
    label_stringIdx = StringIndexer(inputCol = "type", outputCol = "label")
    label_fit = label_stringIdx.fit(idf_transform)
    label_transform = label_fit.transform(idf_transform)
    label_transform_test = label_fit.transform(idf_transform_test)
    #print(label_transform.show(2, truncate = False))
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label')
    model = lr.fit(label_transform)
    predictions = model.transform(label_transform_test)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[TF-IDF - parametri migliori del CV - LogisticRegression] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[TF-IDF - parametri migliori del CV - LogisticRegression] ROC-AUC: {0:.4f}".format(roc_auc))
        
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
    print("[TF-IDF - parametri migliori del CV - LogisticRegression] Precision: {0:.4f}".format(precision))


    plt.clf()
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d"); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['real','fake']); ax.yaxis.set_ticklabels(['real','fake'])
    
        
    plt.savefig('/home/vmadmin/bigdata/confusion_tf_idf_logistic.pdf', dpi=300, bbox_inches="tight")
    '''
    
    
    












    
    
    print("\n")
    
    
    #Linear SVC
    
    list_acc_2 = []
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    svc = LinearSVC(featuresCol = 'features', labelCol = 'label')
    paramGrid = ParamGridBuilder().build()
      
    for i in range (8,16):
        #print("CF")
        count = CountVectorizer(inputCol="text", outputCol='count', vocabSize=2**i)
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
        
        cv = CrossValidator(estimator=svc, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
        cvModel = cv.fit(label_transform)
        print("    [TF-IDF - LinearSVC] Accuracy con vocabSize  = 2^",i, cvModel.avgMetrics)
        list_acc_2.append(cvModel.avgMetrics)
            
    vocabsize = 2**(list_acc_2.index(max(list_acc_2))+8)
    print("[TF-IDF - LinearSVC] vocabSize = ", vocabsize)
    
    list_acc_2 = []
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    svc = LinearSVC(featuresCol = 'features', labelCol = 'label')
    paramGrid = ParamGridBuilder().build()
        
    for i in range (0,3):
        #print("CF")
        count = CountVectorizer(inputCol="text", outputCol='count', vocabSize=vocabsize, minDF=0.05*i)
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
        
        cv = CrossValidator(estimator=svc, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
        cvModel = cv.fit(label_transform)
        print("    [TF-IDF - LinearSVC] Accuracy con minDocFreq  = 0.05 *",i, cvModel.avgMetrics)
        list_acc_2.append(cvModel.avgMetrics)
            
    mindocfreq =  0.05*(list_acc_2.index(max(list_acc_2)))
    print("[TF-IDF - LinearSVC] minDocFreq = ", mindocfreq)
    
    #print("CF")
    count = CountVectorizer(inputCol="text", outputCol='count', vocabSize=vocabsize, minDF=mindocfreq)
    count_fit = count.fit(train)
    count_transform = count_fit.transform(train)
    count_transform_test = count_fit.transform(test)
    #print(count_transform.show(2, truncate = False))
    #print("IDF")
    idf = IDF(inputCol='count', outputCol="features")
    idf_fit = idf.fit(count_transform)
    idf_transform = idf_fit.transform(count_transform)
    idf_transform_test = idf_fit.transform(count_transform_test)
    #print(idf_transform.show(2, truncate = False))
    #print("label")
    label_stringIdx = StringIndexer(inputCol = "type", outputCol = "label")
    label_fit = label_stringIdx.fit(idf_transform)
    label_transform = label_fit.transform(idf_transform)
    label_transform_test = label_fit.transform(idf_transform_test)
    #print(label_transform.show(2, truncate = False))
    svc = LinearSVC(featuresCol = 'features', labelCol = 'label')
    model = svc.fit(label_transform)
    predictions = model.transform(label_transform_test)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[TF-IDF - parametri migliori del CV - LinearSVC] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[TF-IDF - parametri migliori del CV - LinearSVC] ROC-AUC: {0:.4f}".format(roc_auc))
        
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
    print("[TF-IDF - parametri migliori del CV - LinearSVC] Precision: {0:.4f}".format(precision))


    plt.clf()
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d"); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['real','fake']); ax.yaxis.set_ticklabels(['real','fake'])
    
        
    plt.savefig('/home/vmadmin/bigdata/confusion_tf_idf_svc.pdf', dpi=300, bbox_inches="tight")