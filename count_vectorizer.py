import spacy 
from spacy.lang.en import English
import  pandas as pd
import numpy as np
import csv
import xlrd
from itertools import zip_longest
import re
import en_core_web_sm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import random

df=pd.read_csv("./csv_cleaned_Alfredo 2.csv")
df=df.drop_duplicates()

print(df)


nlp = English()


text=df.text.tolist()

#text="When I was young Alfredo"
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

#Implementation of stop words:
filtered_text=[]
#text[0]="run runner runs ciao"
for i in range(0, len(text)):
	if(i%1000==0):
		print(i)
	filtered_sent=""

	#  "nlp" Object is used to create documents with linguistic annotations.
	#text[i]= re.sub('\d*', '', text[i])
	text[i]= re.sub('<', '', text[i])
	text[i]= re.sub('>', '', text[i])
	text[i]= re.sub('\|', '', text[i])
	text[i]= re.sub('\_', '', text[i])
	text[i]= re.sub('\-', '', text[i])
	text[i]= re.sub('\d*', '', text[i])
	doc = nlp(text[i])

	'''
	for word in lem:
    print(word.text,word.lemma_)
	'''
	# filtering stop words
	for word in doc:
		if word.is_stop==False and word.is_punct==False:
			filtered_sent+=" "+word.lemma_
	filtered_text.append(filtered_sent)
	#print("Filtered Sentence:",filtered_sent)


y = df.type
df.drop("type", axis=1)
df['text']=pd.Series(filtered_text)
#print(df['text'])
list_accuracy = []

for i in range(0,11):
	# Make training and test sets 
	X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.25, random_state=random.randint(0,100))

	# Initialize the `count_vectorizer` 
	count_vectorizer = CountVectorizer(stop_words='english')

	# Fit and transform the training data 
	count_train = count_vectorizer.fit_transform(X_train) 
	#print(count_train)

	# Transform the test set 
	count_test = count_vectorizer.transform(X_test)

	# Get the feature names of `count_vectorizer` 
	#print(count_vectorizer.get_feature_names())

	clf = MultinomialNB() 
	clf.fit(count_train, y_train)
	#print(clf.coef_.shape[-1])
	pred = clf.predict(count_test)
	score = metrics.accuracy_score(y_test, pred)
	print("accuracy" ,i,  score)
	#cm = metrics.confusion_matrix(y_test, pred, labels=['fake', 'real'])
	#plot_confusion_matrix(cm, classes=['fake', 'real'])

	list_accuracy.append(score)

print("Accuracy media: ",np.mean(list_accuracy))

