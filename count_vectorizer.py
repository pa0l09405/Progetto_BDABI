#from itertools import zip_longest
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
from collections import Counter
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
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

if __name__=='__main__':
    spark = SparkSession.builder.appName("CountVectorizer").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    #df=pd.read_csv("/home/vmadmin/bigdata/csv_cleaned_Alfredo 2.csv",nrows=10)
    #sqlContext = SQLContext(sc)
    #df = spark.read.format('com.databricks.spark.csv').options(header='true').load('/home/vmadmin/bigdata/csv_cleaned_Alfredo 2.csv')
    df = spark.read.format('com.databricks.spark.csv').options(header='true').load('/home/vmadmin/bigdata/text_and_title_cleaned.csv')
    df = df.drop_duplicates()
    
    print(df.show(4))
    nlp = English()
    text=df.rdd.map(lambda x: x.text).collect()
    type_col=df.rdd.map(lambda x: x.type).collect()

    for i in range(0,len(text)):
        text[i]=text[i].split(" ")
    #print(text)
    #text="When I was young Alfredo"
    #spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    df_nuova = spark.createDataFrame(zip(text, type_col), schema=['text', 'type'])
    print(df_nuova.show(4))

    #y = df.type
    #df.drop("type", axis=1)
    #df['text']=pd.Series(filtered_text)
    #print(df['text'])
    
        
    list_accuracy = []

    for i in range(0,1):

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
        model = RandomForest.trainClassifier(count_train, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=100, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

        # Evaluate model on test instances and compute test error
        predictions = model.predict(count_test.map(lambda x: x.features))
        labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
        testErr = labelsAndPredictions.filter(
            lambda lp: lp[0] != lp[1]).count() / float(testData.count())
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
        
        label=[1.0,0.0]
        cm = confusion_matrix(label_true, pred, labels=label)
        print(cm)

        ax= plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d"); #annot=True to annotate cells

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['real','fake']); ax.yaxis.set_ticklabels(['real','fake'])
        
        plt.savefig('/home/vmadmin/bigdata/confusione.pdf', dpi=300, bbox_inches="tight")
        
        list_accuracy.append(accuracy)

    print("Accuracy media: ",np.mean(list_accuracy))


