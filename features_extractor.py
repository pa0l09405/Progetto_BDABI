from collections import Counter
from itertools import zip_longest
from spacy.lang.en import English
from pymongo import MongoClient
from pandas import DataFrame
import csv
import en_core_web_sm
import numpy as np
import re
import pandas as pd
import pymongo
import spacy 
import xlrd

'''
client = MongoClient('mongodb://localhost:27017/')
db = client.fake_and_real_news_db
collection = db.original_data
#print(collection)
df = DataFrame(collection.find())
'''
#Lettura da csv
df=pd.read_csv('/home/vmadmin/fake_and_real_news_project/fake_and_real_news_dataset.csv')
df=df.drop_duplicates()
#print(df)


#												[Parte 1] Calcolo num_sentences_in_text
# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()
# Create the pipeline 'sentencizer' component
sbd = nlp.create_pipe('sentencizer')
# Add the component to the pipeline
nlp.add_pipe(sbd)

num_sentences=[]
text=df.text.tolist()
title=df.title.tolist()
size=len(title)

print("[Parte 1] Calcolo num_sentences_in_text")

for i in range(0,size):
	if(i%1000==0):
		print(i)

	#"nlp" Object is used to create documents with linguistic annotations.
	doc = nlp(text[i])
		
	# create list of sentence tokens
	sents_list = []
	for sent in doc.sents:
		#print(sent.text)
		sents_list.append(sent.text)
	frasi=np.array(sents_list)
	num_sentences.append(frasi.shape[0])
#print(num_sentences) 
#print(len(num_sentences))
df['num_sentences_in_text']=pd.Series(num_sentences);
print(df)


#												[Parte 2] Calcolo ? e ! in text e title
num_exclamation_mark_in_text = []
num_exclamation_mark_in_title = []
num_question_mark_in_text = []
num_question_mark_in_title = []

print("[Parte 2] Calcolo ? e ! in text e title")
for i in range(0,size):
	if(i%1000==0):
		print(i)
		
	num_exclamation_mark_in_text.append(text[i].count('!'))
	num_exclamation_mark_in_title.append(title[i].count('!'))
	num_question_mark_in_text.append(text[i].count('?'))
	num_question_mark_in_title.append(title[i].count('?'))
#print(num_exclamation_mark_in_text)
#print(num_question_mark_in_text)

df['num_exclamation_mark_in_text']=pd.Series(num_exclamation_mark_in_text);
df['num_exclamation_mark_in_title']=pd.Series(num_exclamation_mark_in_title);
df['num_question_mark_in_text']=pd.Series(num_question_mark_in_text);
df['num_question_mark_in_title']=pd.Series(num_question_mark_in_title);


#												[Parte 3] Calcolo num_capital_words in text e title
num_capital_words_in_text = []
num_capital_words_in_title = []

print("[Parte 3] Calcolo num_capital_words in text e title")
for i in range(0,size):
	if(i%1000==0):
		print(i)
		
	num_capital_words_in_text.append(len(re.findall(r'[A-Z]',text[i])))
	num_capital_words_in_title.append(len(re.findall(r'[A-Z]',title[i])))
#print(num_capital_words)

df['num_capital_words_in_text']=pd.Series(num_capital_words_in_text);
df['num_capital_words_in_title']=pd.Series(num_capital_words_in_title);


#												[Parte 4] Calcolo part_of_speech in text
nlp = en_core_web_sm.load()

list_word_pos = []
list_adv = []
list_noun = []
list_propn = []
list_adj = []
list_punct = []

print("[Parte 4] Calcolo part_of_speech in text")

for i in range(0,size):
	#if(i%1000==0):
	print(i)
		
	list_word_pos = []
	docs=nlp(text[i])
	for word in docs:
		list_word_pos.append(word.pos_)
	list_adv.append(list_word_pos.count("ADV")) 
	list_noun.append(list_word_pos.count("NOUN"))
	list_propn.append(list_word_pos.count("PROPN"))
	list_punct.append(list_word_pos.count("PUNCT"))
	list_adj.append(list_word_pos.count("ADJ"))
	
df['num_adj_in_text']=pd.Series(list_adj);
df['num_adv_in_text']=pd.Series(list_adv);
df['num_noun_in_text']=pd.Series(list_noun);
df['num_propn_in_text']=pd.Series(list_propn);
df['num_punct_in_text']=pd.Series(list_punct);

list_word_pos = []
list_adv = []
list_noun = []
list_propn = []
list_adj = []
list_punct = []

print("[Parte 4] Calcolo part_of_speech in title")

for i in range(0,size):
	if(i%1000==0):
		print(i)
		
	list_word_pos = []
	docs=nlp(title[i])
	for word in docs:
		list_word_pos.append(word.pos_)
	list_adv.append(list_word_pos.count("ADV")) 
	list_noun.append(list_word_pos.count("NOUN"))
	list_propn.append(list_word_pos.count("PROPN"))
	list_punct.append(list_word_pos.count("PUNCT"))
	list_adj.append(list_word_pos.count("ADJ"))
	
df['num_adj_in_title']=pd.Series(list_adj);
df['num_adv_in_title']=pd.Series(list_adv);
df['num_noun_in_title']=pd.Series(list_noun);
df['num_propn_in_title']=pd.Series(list_propn);
df['num_punct_in_title']=pd.Series(list_punct);
	
	
#											[Parte 5] Calcolo num_words e avg_word_lenght

list_num_word=[]
list_avg_word_lenght=[]

print("[Parte 5] Calcolo num_words e avg_word_lenght")

for i in range(0,len(text)):
	if(i%1000==0):
		print(i)
	
	token_list = []

	my_doc = nlp(text[i])
	
	total=0
	for token in my_doc: 
		if not token.is_punct | token.is_space:
			total += len(token.text)
			appoggio = re.sub('[^a-zA-Z]+', '', str(token.text))
			token_list.append(appoggio)
	list_num_word.append(len(token_list))
	list_avg_word_lenght.append(float(total) / float(len(token_list)))
	#print(token_list)
#print(list_num_word)
#print(list_avg_word_lenght)

df['num_word_text']=pd.Series(list_num_word);
df['avg_word_length_text']=pd.Series(list_avg_word_lenght);


#Rimuovo colonne del text e del title
df.drop(['text', 'title'], axis='columns', inplace=True)
#print(df)

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["prova"]
mycol = mydb["coll"]
data = df.to_dict(orient='records')  # Here's our added param..
mycol.insert_many(data)