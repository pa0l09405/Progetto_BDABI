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
from pyspark.ml.feature import OneHotEncoderEstimator
#from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors
from sklearn import metrics
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.multiclass import unique_labels
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler

'''
def extract(row):
    return (row.pmid, )+tuple(row.scaledFeatures.toArray().tolist())
'''	       


'''
# Normalization
def normalize(df_calc, df_apply):
    
    assembler = VectorAssembler().setInputCols(df_calc.columns).setOutputCol("features")
    transformed = assembler.transform(df_calc)
    
    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

    #Compute summary statistics and generate MinMaxScalerModel
    scalerModel =  scaler.fit(transformed.select("features"))
    #scalerModel = scaler.fit(df_calc)

    # rescale each feature to range [min, max].
    #scaledData = scalerModel.transform(df_apply)
    scaledData = scalerModel.transform(transformed)
    print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
    scaledData.select("features", "scaledFeatures").show()
	
    final_data = scaledData.select("pmid","scaledFeatures").rdd.map(extract).toDF(df.columns)
	
    return final_data
'''
	
	
	



def spark_remove_low_var_features(spark_df, features, threshold, remove):
    
    '''	
    This function removes low-variance features from features columns in Spark DF
    
    INPUTS:
    @spark_df: Spark Dataframe
    @features: list of data features in spark_df to be tested for low-variance removal
    @threshold: lowest accepted variance value of each feature
    @remove: boolean variable determine if the low-variance variable should be removed or not
    
    OUTPUTS:
    @spark_df: updated Spark Dataframe 
    @low_var_features: list of low variance features 
    @low_var_values: list of low variance values
    '''
 
    
    # set list of low variance features
    low_var_features = []
    
    # set corresponded list of low-var values
    low_var_values = []
    
    # loop over data features
    for f in features:
        # compute standard deviation of column 'f'
        print(f)
        std = float(spark_df.describe(f).filter("summary = 'stddev'").select(f).collect()[0].asDict()[f])
        #std=0.01
        # compute variance
        var = std*std

        # check if column 'f' variance is less of equal to threshold
        if var <= threshold:
            
            # append low-var feature name and value to the corresponded lists
            low_var_features.append(f)
            low_var_values.append(var)
            
            print(f + ': var: ' + str(var))
            
            # drop column 'f' if @remove is True
            if remove:
                spark_df = spark_df.drop(f)
    
    # return Spark Dataframe, low variance features, and low variance values
    return spark_df, low_var_features, low_var_values











if __name__=='__main__':
    spark = SparkSession.builder.appName("FeatureStyle").getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    #sqlContext = SQLContext(spark)
    #spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    
    '''
    # Generate a Pandas DataFrame
    pdf = pd.DataFrame(np.random.rand(100, 3))
    # Create a Spark DataFrame from a Pandas DataFrame using Arrow
    df = spark.createDataFrame(pdf)
    df.show()
    '''
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client.prova
    collection = db.coll
    print(collection)
    df = pd.DataFrame(collection.find())
    print(type(df))
    df = df.drop_duplicates()
    df.drop(['_id', 'id'], axis='columns', inplace=True)
    print(df)
    
    author=df.author.tolist()
    avg_word_length_text=df.avg_word_length_text.tolist()
    comments=df.comments.tolist()
    country=df.country.tolist()
    #crawled=df.crawled.tolist()
    domain_rank=df.domain_rank.tolist()
    key=df.key.tolist()
    language=df.language.tolist()
    likes=df.likes.tolist()
    main_img_url=df.main_img_url.tolist()
    num_adj_in_text=df.num_adj_in_text.tolist()
    num_adj_in_title=df.num_adj_in_title.tolist()
    num_adv_in_text=df.num_adv_in_text.tolist()
    num_adv_in_title=df.num_adv_in_title.tolist()
    num_capital_words_in_text=df.num_capital_words_in_text.tolist()
    num_capital_words_in_title=df.num_capital_words_in_title.tolist()
    num_exclamation_mark_in_text=df.num_exclamation_mark_in_text.tolist()
    num_exclamation_mark_in_title=df.num_exclamation_mark_in_title.tolist()
    num_noun_in_text=df.num_noun_in_text.tolist()
    num_noun_in_title=df.num_noun_in_title.tolist()
    num_propn_in_text=df.num_propn_in_text.tolist()
    num_propn_in_title=df.num_propn_in_title.tolist()
    num_punct_in_text=df.num_punct_in_text.tolist()
    num_punct_in_title=df.num_punct_in_title.tolist()
    num_question_mark_in_text=df.num_question_mark_in_text.tolist()
    num_question_mark_in_title=df.num_question_mark_in_title.tolist()
    num_sentences_in_text=df.num_sentences_in_text.tolist()
    num_word_text=df.num_word_text.tolist()
    ord_in_thread=df.ord_in_thread.tolist()
    participants_count=df.participants_count.tolist()
    #published=df.published.tolist()
    replies_count=df.replies_count.tolist()
    shares=df.shares.tolist()
    #site_url=df.site_url.tolist()
    spam_score=df.spam_score.tolist()
    thread_title=df.thread_title.tolist()
    type=df.type.tolist()
    
    df_spark = spark.createDataFrame(zip(author,avg_word_length_text,comments,country,domain_rank,key,language,likes,main_img_url,num_adj_in_text,num_adj_in_title,num_adv_in_text,num_adv_in_title,num_capital_words_in_text,num_capital_words_in_title,num_exclamation_mark_in_text,num_exclamation_mark_in_title,num_noun_in_text,num_noun_in_title,num_propn_in_text,num_propn_in_title,num_punct_in_text,num_punct_in_title,num_question_mark_in_text,num_question_mark_in_title,num_sentences_in_text,num_word_text,ord_in_thread,participants_count,replies_count,shares,spam_score,thread_title,type), schema=['author','avg_word_length_text','comments','country','domain_rank','key','language','likes','main_img_url','num_adj_in_text','num_adj_in_title','num_adv_in_text','num_adv_in_title','num_capital_words_in_text','num_capital_words_in_title','num_exclamation_mark_in_text','num_exclamation_mark_in_title','num_noun_in_text','num_noun_in_title','num_propn_in_text','num_propn_in_title','num_punct_in_text','num_punct_in_title','num_question_mark_in_text','num_question_mark_in_title','num_sentences_in_text','num_word_text','ord_in_thread','participants_count','replies_count','shares','spam_score','thread_title','type'])
    #df_spark = spark.createDataFrame(zip(avg_word_length_text,comments,domain_rank,key,likes,num_adj_in_text,num_adj_in_title,num_adv_in_text,num_adv_in_title,num_capital_words_in_text,num_capital_words_in_title,num_exclamation_mark_in_text,num_exclamation_mark_in_title,num_noun_in_text,num_noun_in_title,num_propn_in_text,num_propn_in_title,num_punct_in_text,num_punct_in_title,num_question_mark_in_text,num_question_mark_in_title,num_sentences_in_text,num_word_text,ord_in_thread,participants_count,replies_count,shares,spam_score,type), schema=['avg_word_length_text','comments','domain_rank','key','likes','num_adj_in_text','num_adj_in_title','num_adv_in_text','num_adv_in_title','num_capital_words_in_text','num_capital_words_in_title','num_exclamation_mark_in_text','num_exclamation_mark_in_title','num_noun_in_text','num_noun_in_title','num_propn_in_text','num_propn_in_title','num_punct_in_text','num_punct_in_title','num_question_mark_in_text','num_question_mark_in_title','num_sentences_in_text','num_word_text','ord_in_thread','participants_count','replies_count','shares','spam_score','type'])
    
	
    list_accuracy = []
	
	
    for i in range(0,10):
   
       
	
        print("data set1")
        df_spark.groupBy("type").count().show()
        # Taking 70% of both 0's and 1's into training set
        train = df_spark.sampleBy("type", fractions={'fake': 0.8, 'real': 0.8}, seed=random.randint(0,100))
        train.show()
        print(train.describe())
        # Subtracting 'train' from original 'data' to get test set 
        test = df_spark.exceptAll(train)
        
        print("training set1")
        train.groupBy("type").count().show()
        print("test set1")
        test.groupBy("type").count().show()

        '''        
        for i in train.dtypes:
            #if i[1]=='string':
            print(i[0])
            print(train.groupBy(i[0]).count().orderBy('count', ascending=False).toPandas())
            print(pd.Series(train.select(i[0]).collect()).describe())
            print(train.select(i[0]).describe().show())
			
        '''	
			
        #data_features= set(train.columns) - set(['key','type','language','main_img_url','country'])	
		
        #y=train.select('type')
        #train=train.drop("type")
        #print("FEATURE : ",train.columns)
        #train_normalized = normalize(train,train)
		
        
		
		
        #print("Train normalized : ")
        #print(train_normalized.show())
		
        #train_df, train_low_var_features, train_low_var_values = spark_remove_low_var_features(train_normalized, data_features, 2, True)
        #train_df.drop(train_low_var_features)
        #print(train_df.columns)

        #pipeline stages
        stages = []
	
        
        #PROVA 1	
        #categoricalColumns = ['language','country','author','type']
        #numericCols = ['avg_word_length_text','comments','domain_rank','key','likes','num_adj_in_text','num_adj_in_title','num_adv_in_text','num_adv_in_title','num_capital_words_in_text','num_capital_words_in_title','num_exclamation_mark_in_text','num_exclamation_mark_in_title','num_noun_in_text','num_noun_in_title','num_propn_in_text','num_propn_in_title','num_punct_in_text','num_punct_in_title','num_question_mark_in_text','num_question_mark_in_title','num_sentences_in_text','num_word_text','ord_in_thread','participants_count','replies_count','shares','spam_score']

        #PROVA 2    
        categoricalColumns = ['language','country','author','type']
        numericCols = ['avg_word_length_text','num_adj_in_text','num_adj_in_title','num_adv_in_text','num_adv_in_title','num_capital_words_in_text','num_capital_words_in_title','num_exclamation_mark_in_text','num_exclamation_mark_in_title','num_noun_in_text','num_noun_in_title','num_propn_in_text','num_propn_in_title','num_punct_in_text','num_punct_in_title','num_question_mark_in_text','num_question_mark_in_title','num_sentences_in_text','num_word_text']


    
        
        for categoricalCol in categoricalColumns:
            stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
            encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
            stages += [stringIndexer, encoder]		
		
		
		
        label_stringIdx = StringIndexer(inputCol = 'type', outputCol = 'label')
        stages += [label_stringIdx]	
	
	
        assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
        assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
        assembler.setHandleInvalid("keep")
        stages += [assembler]

        pipeline = Pipeline(stages = stages)
        pipelineModel = pipeline.fit(df_spark)
        df_new = pipelineModel.transform(df_spark)
        selectedCols = ['label', 'features']
        df_new = df_new.select(selectedCols)
        df_new.printSchema()
		
        print("data set2")
        print(df_new.show())
		
        print("data set2 counts")
        df_new.groupBy("label").count().show()
        #split di df_new
        train = df_new.sampleBy("label", fractions={1.0: 0.8, 0.0: 0.8}, seed=random.randint(0,100))		
        print("training set 2")
        print(train.show())
        print(train.describe())
        # Subtracting 'train' from original 'data' to get test set 
        test = df_new.exceptAll(train)
        
        print("training set2")
        train.groupBy("label").count().show()
        print("test set2")
        test.groupBy("label").count().show()
		
		
        #r = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
        #lrModel = lr.fit(train)


        rf = RandomForestClassifier(numTrees=100,featuresCol = 'features', labelCol = 'label')
        rfModel = rf.fit(train)
        predictions = rfModel.transform(test)
		
        #predictions.select("predictedLabel", "label", "features").show(5)
        evaluator = BinaryClassificationEvaluator()
        accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test.count())
        roc_auc = evaluator.evaluate(predictions)

        print ("Accuracy Score: {0:.4f}".format(accuracy))
        print ("ROC-AUC: {0:.4f}".format(roc_auc))




        '''
        importance_list = pd.Series(rf_fitted.featureImportances.values)
        sorted_imp = importance_list.sort_values(ascending= False)
        kept = list((sorted_imp[sorted_imp > 0.03]).index)
        '''		

        '''
        data = [(Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),
                (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
                (Vectors.dense([6.0, 7.0, 0.0, 8.0]),),
                (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),)]
        df_chrome = spark.createDataFrame(data, ["features"])
        df_chrome.show()

        r1 = Correlation.corr(df_chrome, "features").head()
        print("Pearson correlation matrix:\n" + str(r1[0]))

        r2 = Correlation.corr(df_chrome, "features", "spearman").head()
        print("Spearman correlation matrix:\n" + str(r2[0]))
        '''
        
        '''    
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
        
        clf = MultinomialNB() 
        #clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(count_train, y_train)
        #clf.fit(tfidf_train, y_train)
        #print(clf.coef_.shape[-1])
        pred = clf.predict(count_test)
        #pred = clf.predict(tfidf_test)    
        score = metrics.accuracy_score(y_test, pred)
        print("accuracy" ,i,  score)
        
        
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
        '''
        list_accuracy.append(accuracy)

    print("Accuracy media: ",np.mean(list_accuracy))
    