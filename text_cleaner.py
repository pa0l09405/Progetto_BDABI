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
nlp = English()
df=pd.read_csv("./text_cleaned.csv")
df=df.drop_duplicates()

print(df)

# Create the pipeline 'sentencizer' component
sbd = nlp.create_pipe('sentencizer')

# Add the component to the pipeline
nlp.add_pipe(sbd)

num_sentences=[]

#per csv
text=df.text.tolist()
type=df.type.tolist()
size=len(text)
#print("8704",str(title[6647]))
#print("8706",str(title[13930]))

list_text = []
for i in range(0,size):

	cleaned_text = ""

	if(i%1000==0):
	    print(i)
	my_doc = nlp(text[i])
	
	for token in my_doc: 
	
		if token.is_punct==False and token.is_space==False and token.is_stop==False:
			appoggio = str(token.lemma_)
			appoggio = re.sub('[^a-zA-Z]+', '', appoggio)
			cleaned_text +=" "+appoggio

	#print(cleaned_text)
	list_text.append(cleaned_text)

print(list_text)

df = pd.DataFrame(list(zip(list_text, type)), columns =['text', 'type'])

df.to_csv('./text_and_title_cleaned.csv')
'''
with open('./word.csv', 'w', encoding="ISO-8859-1", newline='') as csv_word:
		writer = csv.writer(csv_word)
		rows = zip_longest(*[list_lunghezze,list_avg_lunghezze], fillvalue = '')
		writer.writerows(rows)
csv_word.close()
'''





		

