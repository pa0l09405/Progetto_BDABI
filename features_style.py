from collections import Counter
from pandas import DataFrame
from pymongo import MongoClient
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorSlicer
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.stat import Correlation
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.utils.multiclass import unique_labels
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymongo
import random
import re
import seaborn as sns
import spacy 

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

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
	
'''
def spark_remove_low_var_features(spark_df, features, threshold, remove):
    
    	
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
'''

if __name__=='__main__':
    spark = SparkSession.builder.appName("Features_Style").getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    
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
    replies_count=df.replies_count.tolist()
    shares=df.shares.tolist()
    spam_score=df.spam_score.tolist()
    thread_title=df.thread_title.tolist()
    type=df.type.tolist()
    
    df_spark = spark.createDataFrame(zip(avg_word_length_text,num_adj_in_text,num_adj_in_title,num_adv_in_text,num_adv_in_title,num_capital_words_in_text,num_capital_words_in_title,num_exclamation_mark_in_text,num_exclamation_mark_in_title,num_noun_in_text,num_noun_in_title,num_propn_in_text,num_propn_in_title,num_punct_in_text,num_punct_in_title,num_question_mark_in_text,num_question_mark_in_title,num_sentences_in_text,num_word_text,type), schema=['avg_word_length_text','num_adj_in_text','num_adj_in_title','num_adv_in_text','num_adv_in_title','num_capital_words_in_text','num_capital_words_in_title','num_exclamation_mark_in_text','num_exclamation_mark_in_title','num_noun_in_text','num_noun_in_title','num_propn_in_text','num_propn_in_title','num_punct_in_text','num_punct_in_title','num_question_mark_in_text','num_question_mark_in_title','num_sentences_in_text','num_word_text','type'])
			
    '''
    #y=train.select('type')
    #train=train.drop("type")
    #print("FEATURE : ",train.columns)
    #train_normalized = normalize(train,train)
		
    #print("Train normalized : ")
    #print(train_normalized.show())
		
    #train_df, train_low_var_features, train_low_var_values = spark_remove_low_var_features(train_normalized, data_features, 2, True)
    #train_df.drop(train_low_var_features)
    #print(train_df.columns)
    '''
    
    #pipeline stages
    stages = []
	    
    #...	
    numericCols = ['avg_word_length_text','num_adj_in_text','num_adj_in_title','num_adv_in_text','num_adv_in_title','num_capital_words_in_text','num_capital_words_in_title','num_exclamation_mark_in_text','num_exclamation_mark_in_title','num_noun_in_text','num_noun_in_title','num_propn_in_text','num_propn_in_title','num_punct_in_text','num_punct_in_title','num_question_mark_in_text','num_question_mark_in_title','num_sentences_in_text','num_word_text']
    label_stringIdx = StringIndexer(inputCol = 'type', outputCol = 'label')
    stages += [label_stringIdx]	
    assemblerInputs=numericCols 
    #print("AssemblerInputs ", assemblerInputs)
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df_spark)
    df_new = pipelineModel.transform(df_spark)
    selectedCols = ['label', 'features']
    df_new = df_new.select(selectedCols)
    df_new.printSchema()
		
    print("Data set")
    print(df_new.show(truncate=False))
			
    #Split train e test
    train = df_new.sampleBy("label", fractions={1.0: 0.8, 0.0: 0.8}, seed=2)
    test = df_new.exceptAll(train)
    
    #... 
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    paramGrid = ParamGridBuilder().build()
    
    #Random Forest
    classifier = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
    cv = CrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(train) 
    print("[Random Forest] Default accuracy: ",cvModel.avgMetrics)
    
    #Decision Tree
    classifier = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label')
    cv = CrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(train) 
    print("[Decision Tree] Default accuracy: ",cvModel.avgMetrics)
    '''
    #Linear SVC
    classifier = LinearSVC(featuresCol = 'features', labelCol = 'label')
    cv = CrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(train) 
    print("Linear SVC (default accuracy) ",cvModel.avgMetrics)
    
    #Logistic Regression
    classifier = LogisticRegression(featuresCol = 'features', labelCol = 'label')
    cv = CrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(train) 
    print("Logistic Regression (default accuracy) ",cvModel.avgMetrics)
    
    #Naive Bayes
    classifier = NaiveBayes(featuresCol = 'features', labelCol = 'label')
    cv = CrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(train) 
    print("Naive Bayes (default accuracy) ",cvModel.avgMetrics)
    '''
    '''
    print("Metodo 1")
    selector = ChiSqSelector(numTopFeatures=1, featuresCol="features", outputCol="selectedFeatures", labelCol="clicked")
    result = selector.fit(train).transform(train)
    print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
    result.show()
    '''
    
    print("[Decision Tree] Feature Selection")
    random_fs = RandomForestClassifier(numTrees=100, maxDepth=5, featuresCol = 'features', labelCol = 'label')
    model = random_fs.fit(train)
    print(model.featureImportances)
    varlist = ExtractFeatureImp(model.featureImportances, train, "features")

    list_acc = []
    for i in range (1,20):
        varidx = [x for x in varlist['idx'][0:i]]
        slicer = VectorSlicer(inputCol="features", outputCol="features2", indices=varidx)
        new_train = slicer.transform(train)
        new_train = new_train.drop('rawPrediction', 'probability', 'prediction')
        dt2 = DecisionTreeClassifier(featuresCol = 'features2', labelCol = 'label')
        #print(dt2.explainParams())
        paramGrid = ParamGridBuilder().build()
        cv = CrossValidator(estimator=dt2, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
        cvModel = cv.fit(new_train) 
        print("    [Decision Tree] Accuracy con num_features = ",i, cvModel.avgMetrics)
        list_acc.append(cvModel.avgMetrics)
            
    best_idx = list_acc.index(max(list_acc))+1
    print("[Decision Tree] Il numero di features ottimo è ", best_idx)

    varidx = [x for x in varlist['idx'][0:best_idx]]
    slicer = VectorSlicer(inputCol="features", outputCol="features2", indices=varidx)
    new_train = slicer.transform(train)
    new_train = new_train.drop('rawPrediction', 'probability', 'prediction')
    dt2 = DecisionTreeClassifier(featuresCol = 'features2', labelCol = 'label')
    #print(dt2.explainParams())
    paramGrid = (ParamGridBuilder()
             .addGrid(dt2.maxDepth, [2, 5, 10, 15])
             .addGrid(dt2.impurity, ['gini', 'entropy'])
             .addGrid(dt2.minInstancesPerNode, [1, 2, 3])
             .build())
    cv = CrossValidator(estimator=dt2, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(new_train) 
    print("[Decision Tree] Selezione degli iperparametri ",cvModel.avgMetrics)
    print("[Decision Tree] Modello migliore ", cvModel.bestModel)
    print("[Decision Tree] Parametri migliori ", cvModel.bestModel.extractParamMap())
    
    varidx = [x for x in varlist['idx'][0:best_idx]]
    slicer = VectorSlicer(inputCol="features", outputCol="features2", indices=varidx)
    new_test = slicer.transform(test)
    predictions = cvModel.bestModel.transform(new_test)
        
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[Decision Tree] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[Decision Tree] ROC-AUC: {0:.4f}".format(roc_auc))
        
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
    print("[Decision Tree] Precision: {0:.4f}".format(precision))
    
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d"); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['real','fake']); ax.yaxis.set_ticklabels(['real','fake'])
        
    plt.savefig('/home/vmadmin/fake_and_real_news_project/confusione_decision_tree.pdf', dpi=300, bbox_inches="tight")
    
    
    print("\n\n\n\n")
    print("[Random Forest] Selezione degli iperparametri ")
    random_forest = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', featureSubsetStrategy='auto', minInstancesPerNode=2)
		
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    paramGrid = (ParamGridBuilder()
        .addGrid(random_forest.maxDepth, [5, 10])
        .addGrid(random_forest.numTrees, [20, 50, 100])
        .addGrid(random_forest.subsamplingRate, [0.7, 1.0])
        .build())
    cv = CrossValidator(estimator=random_forest, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
        
    # Run cross validations.  This can take about 6 minutes since it is training over 20 trees!
    cvModel = cv.fit(train)
    print("[Random Forest] Modello migliore ", cvModel.bestModel)
    print("[Random Forest] Parametri migliori ", cvModel.bestModel.extractParamMap())

    predictions = cvModel.transform(test)
        
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[Random Forest] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[Random Forest] ROC-AUC: {0:.4f}".format(roc_auc))
        
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
    print ("[Random Forest] Precision: {0:.4f}".format(precision))
    
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d"); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['real','fake']); ax.yaxis.set_ticklabels(['real','fake'])
        
    plt.savefig('/home/vmadmin/fake_and_real_news_project/confusion_random_forest.pdf', dpi=300, bbox_inches="tight")
    
    for i in range(0,1):
        print("Arrivederci")
        '''
    	random_forest = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', featureSubsetStrategy='auto', minInstancesPerNode=2)
		
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        paramGrid = (ParamGridBuilder()
             .addGrid(random_forest.maxDepth, [5, 10])
             .addGrid(random_forest.numTrees, [20, 50, 100])
             .addGrid(random_forest.subsamplingRate, [0.7, 1.0])
             .build())
        cv = CrossValidator(estimator=random_forest, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
        
        # Run cross validations.  This can take about 6 minutes since it is training over 20 trees!
        cvModel = cv.fit(train)
        print("Modello migliore ", cvModel.bestModel)
        print("Parametri migliori ", cvModel.bestModel.extractParamMap())

        predictions = cvModel.transform(test)
        
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        evaluator = BinaryClassificationEvaluator()
        roc_auc = evaluator.evaluate(predictions)        
        
        print ("Accuracy Score Best Model: {0:.4f}".format(accuracy))
        print ("ROC-AUC Best Model: {0:.4f}".format(roc_auc))
        
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
        print ("Precision Best Model: {0:.4f}".format(precision))
        '''
		
        #lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=100)
        #lrModel = lr.fit(train)
        #predictions = lrModel.transform(test)
        '''
        
        rf = RandomForestClassifier(numTrees=100, featuresCol = 'features', labelCol = 'label', maxDepth=10, minInstancesPerNode=2)
        rfModel = rf.fit(train)
        predictions = rfModel.transform(test)
        
        '''
        #svm = LinearSVC(featuresCol = 'features', labelCol = 'label', maxIter=10, regParam=0.1)
        #svmModel = svm.fit(train)
        #predictions = svmModel.transform(test)
        
        #dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label')
        #dtModel = dt.fit(train)
        #predictions = dtModel.transform(test)
        
        
        #print("                           PREDIZIONIIIIIIIIIIIIIIII ", predictions)
        
        #evaluator = BinaryClassificationEvaluator()
        #metrics = BinaryClassificationMetrics(predictions)
        #metrics = BinaryClassificationMetrics(predictionAndLabels)

        #accuracy_1 = predictions.filter(predictions.label == predictions.prediction).count() / float(test.count())
        #accuracy_2 = metrics.accuracy
        #roc_auc = evaluator.evaluate(predictions)
        #tp = predictions.filter(predictions.label == predictions.prediction).count()
        #fp = predictions.filter((predictions.label == 1.0) and (predictions.prediction == 0.0)).count()
        #precision = tp /(tp + fp)
        #roc_auc = metrics.areaUnderROC
        #precision = metrics.precision()
        
        #statistics = rfModel.summary
        #accuracy = statistics.
        #roc_auc = statistics.roc.toPandas()
        #precision = statistics.pr.toPandas()
        '''
        
        # Select (prediction, true label) and compute test error
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        evaluator = BinaryClassificationEvaluator()
        roc_auc = evaluator.evaluate(predictions)        
        
        print ("Accuracy Score a mano: {0:.4f}".format(accuracy))
        print ("ROC-AUC a mano: {0:.4f}".format(roc_auc))
        
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
        print ("Precision a mano: {0:.4f}".format(precision))
        
        
        rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
        rfModel = rf.fit(train)
        predictions = rfModel.transform(test)
        
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        evaluator = BinaryClassificationEvaluator()
        roc_auc = evaluator.evaluate(predictions)        
        
        print ("Accuracy Score default: {0:.4f}".format(accuracy))
        print ("ROC-AUC: {0:.4f}".format(roc_auc))
        
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
        print ("Precision: {0:.4f}".format(precision))
        
        
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