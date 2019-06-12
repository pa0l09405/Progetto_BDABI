import spacy 
# Word tokenization
from spacy.lang.en import English
import  pandas as pd
import numpy as np
import csv
import xlrd
from itertools import zip_longest
import re

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()
#nlp=spacy.load("xx_ent_wiki_sm")

df=pd.read_csv("./csv_cleaned_Alfredo 2.csv")
df=df.drop_duplicates()
#ctesto=df.provider
#print(ctesto)

print(df)

'''
workbook = xlrd.open_workbook("./all_data_filtered.xlsx","rb", encoding_override='UTF-8')
sheet = workbook.sheet_by_index(0)
rows = []
j=15
for i in range(sheet.nrows):
    columns = []
    #for j in range(sheet.ncols):
    columns.append(sheet.cell(i, j).value)
    rows.append(columns)
#print (rows)
print(type(rows))

'''
'''
for i in range (sheet.nrows):
	text=str(rows[i])
#text = """When learning data science, you shouldn't get discouraged!
#Challenges and setbacks aren't failures, they're just part of the journey. You've got this!"""

#  "nlp" Object is used to create documents with linguistic annotations.
	my_doc = nlp(text)

# Create list of word tokens
	token_list = []
	for token in my_doc:
		token_list.append(token.text)
	#print(token_list)
'''
# Load English tokenizer, tagger, parser, NER and word vectors
#nlp = English()

# Create the pipeline 'sentencizer' component
sbd = nlp.create_pipe('sentencizer')

# Add the component to the pipeline
nlp.add_pipe(sbd)
#for i in range (sheet.nrows):

num_sentences=[]

#per csv
text=df.text.tolist()
title=df.title.tolist()
#print(text)

'''
# per excel
for i in range (1,sheet.nrows):
	text=str(rows[i])
	delete_list=re.findall("<U+.*?>",text)#2019
	delete_list_distinct=list(set(delete_list))
	print(delete_list_distinct)
'''
'''
	for el in delete_list_distinct:
		stringa_da_testare=el[3]+el[4]
		if stringa_da_testare=='00':
			stringa_da_convertire=el[5]+el[6]
			carattere_decimale=int(stringa_da_convertire,16)
			print(carattere_decimale)
			carattere_ascii=str(chr(carattere_decimale))
			print(carattere_ascii)
			text=text.replace(el,carattere_ascii)
	#print(delete_list_distinct)
	#for word in delete_list_distinct:
		#if word in text:
			#text=text.replace(word," ")
	#print(text)
#print(text)

'''
size=len(title)
#print("8704",str(title[6647]))
#print("8706",str(title[13930]))
for i in range(0,size-1):
	print(i)
	#print(str(text[i]))
#  "nlp" Object is used to create documents with linguistic annotations.
	#text=str(text[i])
	doc = nlp(title[i])
		
		# create list of sentence tokens
	sents_list = []
	for sent in doc.sents:
		#print(sent.text)
		sents_list.append(sent.text)
	frasi=np.array(sents_list)
	#print(frasi.shape)
	num_sentences.append(frasi.shape[0])
print(num_sentences)

'''
with open('./num_sentences.csv', 'w', encoding="ISO-8859-1", newline='') as csv_num_sentences:
		writer = csv.writer(csv_num_sentences)
		writer.writerow(['num_sentences'])
		rows = zip_longest(*[num_sentences], fillvalue = '')
		writer.writerows(rows)
csv_num_sentences.close()
'''