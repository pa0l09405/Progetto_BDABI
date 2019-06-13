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

# Load English tokenizer, tagger, parser, NER and word vectors
#nlp = English()

#for multilanguage
nlp=spacy.load("xx_ent_wiki_sm")

df=pd.read_csv("./csv_cleaned_Alfredo 2.csv")
df=df.drop_duplicates()

print(df)


# Create the pipeline 'sentencizer' component
sbd = nlp.create_pipe('sentencizer')

# Add the component to the pipeline
nlp.add_pipe(sbd)

num_sentences=[]

#per csv
text=df.text.tolist()
title=df.title.tolist()
#print(text)



size=len(title)
#print("8704",str(title[6647]))
#print("8706",str(title[13930]))

'''
for i in range(0,size):
#  "nlp" Object is used to create documents with linguistic annotations.
	doc = nlp(text[i])
		
	# create list of sentence tokens
	sents_list = []
	for sent in doc.sents:
		#print(sent.text)
		sents_list.append(sent.text)
	frasi=np.array(sents_list)
	num_sentences.append(frasi.shape[0])
print(num_sentences) 
print(len(num_sentences))
df['num_sentences_in_text']=pd.Series(num_sentences);
'''
#print(df)

'''
# print on csv with new col
with open('./num_sentences_in_text.csv', 'w', encoding="ISO-8859-1", newline='') as csv_num_sentences:
		writer = csv.writer(csv_num_sentences)
		rows = zip_longest(*[num_sentences], fillvalue = '')
		writer.writerows(rows)
csv_num_sentences.close()
'''

'''
num_exclamation_mark_in_text = []
num_exclamation_mark_in_title = []
num_question_mark_in_text = []
num_question_mark_in_title = []

for i in range(0,size):
	num_exclamation_mark_in_text.append(text[i].count('!'))
	num_exclamation_mark_in_title.append(title[i].count('!'))
	num_question_mark_in_text.append(text[i].count('?'))
	num_question_mark_in_title.append(title[i].count('?'))
#print(num_exclamation_mark_in_text)
#print(num_question_mark_in_text)

with open('./num_marks.csv', 'w', encoding="ISO-8859-1", newline='') as csv_num_marks:
		writer = csv.writer(csv_num_marks)
		rows = zip_longest(*[num_exclamation_mark_in_text,num_exclamation_mark_in_title,num_question_mark_in_text,num_question_mark_in_title], fillvalue = '')
		writer.writerows(rows)
csv_num_marks.close()
'''
'''
num_capital_words_in_text = []
num_capital_words_in_title = []

for i in range(0,size):
	num_capital_words_in_text.append(len(re.findall(r'[A-Z]',text[i])))
	num_capital_words_in_title.append(len(re.findall(r'[A-Z]',title[i])))
#print(num_capital_words)

with open('./num_capital_words.csv', 'w', encoding="ISO-8859-1", newline='') as csv_num_capital_words:
		writer = csv.writer(csv_num_capital_words)
		rows = zip_longest(*[num_capital_words_in_text,num_capital_words_in_title], fillvalue = '')
		writer.writerows(rows)
csv_num_capital_words.close()
'''

nlp = en_core_web_sm.load()
list_word_pos = []

list_adv = []
list_noun = []
list_propn = []
list_I_we_me_us = []
list_adj = []
list_no_not = []
list_punct = []

for i in range(0,2):
	list_word_pos = []
	docs=nlp(text[i])
	for word in docs:
		list_word_pos.append(word.pos_)
	list_adv.append(list_word_pos.count("ADV")) 
	list_noun.append(list_word_pos.count("NOUN"))
	list_propn.append(list_word_pos.count("PROPN"))
	list_punct.append(list_word_pos.count("PUNCT"))
	list_adj.append(list_word_pos.count("ADJ"))
	
	count_I_we_us=text[i].count(" I ")+text[i].count(" we ")+text[i].count(" WE ")+text[i].count(" We ")+text[i].count(" us ")+text[i].count(" US ")+text[i].count(" Us ")+text[i].count(" me ")+text[i].count(" ME ")+text[i].count(" Me ")
	list_I_we_me_us.append(count_I_we_us)
	
	count_no_not=text[i].count(" no ")+text[i].count(" NO ")+text[i].count(" No ")+text[i].count(" not ")+text[i].count(" NOT ")+text[i].count(" Not ")
	list_no_not.append(count_no_not)
	
print("List adv = ",list_adv)
print("List adj = ",list_adj)
print("List noun = ",list_noun)
print("List I_we_me_us = ",list_I_we_me_us)
print("List no_not = ",list_no_not)
print("List propn = ",list_propn)
print("List punct = ",list_punct)


		

