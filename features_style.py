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
from spark_stratified import StratifiedCrossValidator
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

if __name__=='__main__':
    spark = SparkSession.builder.appName("Features_Style").getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client.dataset_fake_and_real_news
    collection = db.features_style
    #print(collection)
    df = pd.DataFrame(collection.find())
    #print(type(df))
    print("Dataframe pandas")
    print(df)
    df = df.drop("_id", axis=1)
    df_spark = spark.createDataFrame(df)
    #print("Dataframe spark")
    #print(df_spark.show(truncate = False))
	
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
		
    print("[Data set] ")
    print(df_new.show(truncate=False))
			
    #Split train e test
    train = df_new.sampleBy("label", fractions={1.0: 0.8, 0.0: 0.8}, seed=2)
    test = df_new.exceptAll(train)
    
	#... 
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    paramGrid = ParamGridBuilder().build()
    
    #Random Forest
    classifier = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
    cv = StratifiedCrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(train) 
    print("[Random Forest] Default accuracy: ",cvModel.avgMetrics)
    
    #Decision Tree
    classifier = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label')
    cv = StratifiedCrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(train) 
    print("[Decision Tree] Default accuracy: ",cvModel.avgMetrics)
    
    #Linear SVC
    classifier = LinearSVC(featuresCol = 'features', labelCol = 'label')
    cv = StratifiedCrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(train) 
    print("[Linear SVC] Default accuracy: ",cvModel.avgMetrics)
    
    #Logistic Regression
    classifier = LogisticRegression(featuresCol = 'features', labelCol = 'label')
    cv = StratifiedCrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(train) 
    print("[Logistic Regression] Default accuracy: ",cvModel.avgMetrics)
    
    #Naive Bayes
    classifier = NaiveBayes(featuresCol = 'features', labelCol = 'label')
    cv = StratifiedCrossValidator(estimator=classifier, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(train) 
    print("[Naive Bayes] Default accuracy: ",cvModel.avgMetrics)
    
    print("\n")
    
    print("[Decision Tree con Gini Index] Feature Selection - Gini index")
    random_fs = RandomForestClassifier(numTrees=100, maxDepth=5, featuresCol = 'features', labelCol = 'label')
    model = random_fs.fit(train)
    #print(model.featureImportances)
    varlist = ExtractFeatureImp(model.featureImportances, train, "features")

    list_acc = []
    for i in range (1,20):
        varidx = [x for x in varlist['idx'][0:i]]
        slicer = VectorSlicer(inputCol="features", outputCol="features2", indices=varidx)
        new_train = slicer.transform(train)
        new_train = new_train.drop('rawPrediction', 'probability', 'prediction')
        dt2 = DecisionTreeClassifier(featuresCol = 'features2', labelCol = 'label')
        paramGrid = ParamGridBuilder().build()
        cv = StratifiedCrossValidator(estimator=dt2, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
        cvModel = cv.fit(new_train) 
        print("    [Decision Tree con Gini Index] Accuracy con num_features = ",i, cvModel.avgMetrics)
        list_acc.append(cvModel.avgMetrics)
            
    best_idx = list_acc.index(max(list_acc))+1
    print("[Decision Tree con Gini Index] Il numero di features ottimo è ", best_idx)
    
    print(ExtractFeatureImp(model.featureImportances, train, "features").head(best_idx))

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
    cv = StratifiedCrossValidator(estimator=dt2, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(new_train) 
    print("[Decision Tree con Gini Index] Selezione degli iperparametri ")
    print("[Decision Tree con Gini Index] Modello migliore ", cvModel.bestModel)
    best_params = cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ]
    print("[Decision Tree con Gini Index] Parametri migliori", best_params)
    
    
    varidx = [x for x in varlist['idx'][0:best_idx]]
    slicer = VectorSlicer(inputCol="features", outputCol="features2", indices=varidx)
    new_test = slicer.transform(test)
    predictions = cvModel.bestModel.transform(new_test)
        
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[Decision Tree con Gini Index - parametri migliori] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[Decision Tree con Gini Index - parametri migliori] ROC-AUC: {0:.4f}".format(roc_auc))
        
    label=[1.0,0.0]
    predictions_pandas = predictions.toPandas()
    lbl = predictions_pandas['label'].tolist()
    prd = predictions_pandas['prediction'].tolist()
    cm_dt_gini = confusion_matrix(lbl, prd, labels=label)
    #print("Confusion matrix ",cm)
    tp = cm_dt_gini[0][0]
    #print("TP ", tp)
    fp = cm_dt_gini[1][0]
    #print("FP ", fp)        
    precision = tp /(tp + fp)
    print("[Decision Tree con Gini Index - parametri migliori] Precision: {0:.4f}".format(precision))
    print("[Decision Tree con Gini Index - parametri migliori] Matrice di confusione: ", cm_dt_gini)
        
    ax= plt.subplot()
    sns.heatmap(cm_dt_gini, annot=True, ax = ax, cmap='Blues', fmt="d"); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['real','fake']); ax.yaxis.set_ticklabels(['real','fake'])
        
    plt.savefig('/home/vmadmin/fake_and_real_news_project/confusion_decision_tree_gini.pdf', dpi=300, bbox_inches="tight")

    print("[Decision Tree] Feature Selection - ChiSquared Selector")  
    
    #... 
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
      
    selector = ChiSqSelector(numTopFeatures=10, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    result = selector.fit(train)
    #indici=result.selectedFeatures
    #print("indici")
    #print(indici)
    result2=result.transform(train)
    colname = df_spark.toPandas().columns[result.selectedFeatures]
    #print (colname)
    list_acc_2 = []
    for i in range (1,20):
        selector = ChiSqSelector(numTopFeatures=i, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
        paramGrid = ParamGridBuilder().build()
        dt3 = DecisionTreeClassifier(featuresCol = 'selectedFeatures', labelCol = 'label')
        cv = StratifiedCrossValidator(estimator=dt3, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
        result = selector.fit(train)
        result2=result.transform(train)
        cvModel = cv.fit(result2)
        print("    [Decision Tree con ChiSq] Accuracy con numTopFeatures  = ",i, cvModel.avgMetrics)
        list_acc_2.append(cvModel.avgMetrics)
            
    best_numTopFeatures = list_acc_2.index(max(list_acc_2))+1
    print("[Decision Tree con ChiSq] Il numero di features ottimo è ", best_numTopFeatures)
    selector = ChiSqSelector(numTopFeatures=best_numTopFeatures, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    new_train = selector.fit(train)
    new_train2 = new_train.transform(train)
    print("[Decision Tree con ChiSq] Le top features sono: ", df_spark.toPandas().columns[new_train.selectedFeatures])
    
    dt3 = DecisionTreeClassifier(featuresCol = 'selectedFeatures', labelCol = 'label')
    #print(dt2.explainParams())
    paramGrid = (ParamGridBuilder()
             .addGrid(dt3.maxDepth, [2, 5, 10, 15])
             .addGrid(dt3.impurity, ['gini', 'entropy'])
             .addGrid(dt3.minInstancesPerNode, [1, 2, 3])
             .build())
    cv = StratifiedCrossValidator(estimator=dt3, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(new_train2) 
    print("[Decision Tree con ChiSq] Selezione degli iperparametri ")
    print("[Decision Tree con ChiSq] Modello migliore ", cvModel.bestModel)
    best_params = cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ]
    print("[Decision Tree con ChiSq] Parametri migliori", best_params)
    
    new_test = new_train.transform(test)
    predictions = cvModel.bestModel.transform(new_test)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[Decision Tree con ChiSq - parametri migliori] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[Decision Tree con ChiSq - parametri migliori] ROC-AUC: {0:.4f}".format(roc_auc))

        
    label=[1.0,0.0]
    predictions_pandas = predictions.toPandas()
    lbl = predictions_pandas['label'].tolist()
    prd = predictions_pandas['prediction'].tolist()
    cm_chi = confusion_matrix(lbl, prd, labels=label)
    #print("Confusion matrix ",cm)
    tp = cm_chi[0][0]
    #print("TP ", tp)
    fp = cm_chi[1][0]
    #print("FP ", fp)        
    precision = tp /(tp + fp)
    print("[Decision Tree con ChiSq - parametri migliori] Precision: {0:.4f}".format(precision))
    print("[Decision Tree con ChiSq - parametri migliori] Matrice di confusione: ", cm_chi)

    plt.clf()
    ax1= plt.subplot()
    sns.heatmap(cm_chi, annot=True, ax = ax1, cmap='Blues', fmt="d"); #annot=True to annotate cells

    # labels, title and ticks
    ax1.set_xlabel('Predicted labels');ax1.set_ylabel('True labels'); 
    ax1.set_title('Confusion Matrix'); 
    ax1.xaxis.set_ticklabels(['real','fake']); ax1.yaxis.set_ticklabels(['real','fake'])
        
    plt.savefig('/home/vmadmin/fake_and_real_news_project/confusion_decision_tree_chisq.pdf', dpi=300, bbox_inches="tight")
    
    
    print("\n\n\n\n")
    print("[Random Forest] Selezione degli iperparametri ")
    
    #... 
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    
    random_forest = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', featureSubsetStrategy='auto', minInstancesPerNode=2)
		
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    paramGrid = (ParamGridBuilder()
        .addGrid(random_forest.maxDepth, [5, 10])
        .addGrid(random_forest.numTrees, [20, 50, 100])
        .addGrid(random_forest.subsamplingRate, [0.7, 1.0])
        .build())
    cv = StratifiedCrossValidator(estimator=random_forest, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
        
    # Run cross validations.  This can take about 6 minutes since it is training over 20 trees!
    cvModel = cv.fit(train)
    print("[Random Forest] Selezione degli iperparametri ")
    print("[Random Forest] Modello migliore ", cvModel.bestModel)
    best_params = cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ]
    print("[Random Forest] Parametri migliori", best_params)

    predictions = cvModel.transform(test)
        
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = BinaryClassificationEvaluator()
    roc_auc = evaluator.evaluate(predictions)        
        
    print ("[Random Forest - parametri migliori] Accuracy Score: {0:.4f}".format(accuracy))
    print ("[Random Forest - parametri migliori] ROC-AUC: {0:.4f}".format(roc_auc))
        
    label=[1.0,0.0]
    predictions_pandas = predictions.toPandas()
    lbl = predictions_pandas['label'].tolist()
    prd = predictions_pandas['prediction'].tolist()
    cm_rand = confusion_matrix(lbl, prd, labels=label)
    #print("Confusion matrix ",cm)
    tp = cm_rand[0][0]
    #print("TP ", tp)
    fp = cm_rand[1][0]
    #print("FP ", fp)        
    precision = tp /(tp + fp)
    print ("[Random Forest - parametri migliori] Precision: {0:.4f}".format(precision))
    print ("[Random Forest - parametri migliori] Matrice di confusione: ", cm_rand)
    
    plt.clf()
    ax2= plt.subplot()
    sns.heatmap(cm_rand, annot=True, ax = ax2, cmap='Blues', fmt="d"); #annot=True to annotate cells

    # labels, title and ticks
    ax2.set_xlabel('Predicted labels');ax2.set_ylabel('True labels'); 
    ax2.set_title('Confusion Matrix'); 
    ax2.xaxis.set_ticklabels(['real','fake']); ax2.yaxis.set_ticklabels(['real','fake'])
        
    plt.savefig('/home/vmadmin/fake_and_real_news_project/confusion_random_forest.pdf', dpi=300, bbox_inches="tight")