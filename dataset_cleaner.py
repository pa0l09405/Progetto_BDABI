import csv
import re
import pandas as pd
import numpy as np

#cols=np.arange(2,22,1)
#df=pd.read_csv("./prova.csv", index_col=0, usecols=cols, quotechar='~')
#print(language_list_distinct)
#print(type(df))

df=pd.read_csv("./csv_definitivo.csv")
print(df)

print("inizio la modifica dei linguaggi")

df.language=df.language.replace(np.nan,"english")
df.language=df.language.replace("ignore","english")
language_list_distinct=list(set(df.language))
print(language_list_distinct)
list_language_to_delete=["chinese","arabic","polish","turkish","finnish","russian","norwegian","greek"]
for elem in list_language_to_delete:
	df=df[df.language != elem]
#df['site_url'] = df.site_url.str.replace(r'(^.*ball.*$)', 'ball sport')

print("flne della modifica dei linguaggi")


print("inizio la modifica dei siti")

clanguage=df.site_url.tolist()
#cauthor=df.author.tolist()
#print(clanguage[3214])

	
#df['site_url'][df.site_url.str.contains('wsj.com')] = 'wsj.com'
print("inizio wsj")
df['site_url'] = df.site_url.str.replace('.*wsj.*', 'wsj.com')
print("fine wsj")
df['site_url'] = df.site_url.str.replace('.*nytimes.*', 'nytimes.com')
print("fine nyt")
df['site_url'] = df.site_url.str.replace('.*politico.*', 'politico.com')
print("fine politico")
df['site_url'] = df.site_url.str.replace('.*americannews.*', 'americannews.com')
print("fine americannews")


print("fine della modifica dei siti")


#language_list_distinct=list(set(df.site_url))
#print(language_list_distinct)
clanguage=df.site_url.tolist()


print("inizio la modifica dei testi")


new_text=df.text.tolist()

for i in range(0,len(new_text)-1):
	delete_list = re.findall("<U+.*?>",new_text[i])
	#print(delete_list)
	for el in delete_list:
		stringa_da_testare=el[3]+el[4]
		if stringa_da_testare=='00':
			stringa_da_convertire=el[5]+el[6]
			carattere_decimale=int(stringa_da_convertire,16)
			#print(carattere_decimale)
			carattere_ascii=str(chr(carattere_decimale))
			#print(carattere_ascii)
			new_text[i]=new_text[i].replace(el,carattere_ascii)
			#print(new_text[i])
	#print(elem)

	
	
for i in range(0,len(new_text)-1):
	new_text[i]=new_text[i].replace("<U+2018>","'")
	new_text[i]=new_text[i].replace("<U+2019>","'")
	new_text[i]=new_text[i].replace("<U+2022>",".")
	new_text[i]=new_text[i].replace("<U+2026>","...")
	new_text[i]=new_text[i].replace("<U+201C>","<")
	new_text[i]=new_text[i].replace("<U+201D>",">")
	new_text[i]=new_text[i].replace("\n"," ")
	delete_list=re.findall("<U+.*?>",new_text[i])
	for el in delete_list:
		new_text[i]=new_text[i].replace(el, " ")
	#print(new_text[i])	
	
df['text']=new_text	

print("fine della modifica dei testi")

print("inizio la modifica dei titoli")

new_text=df.title.tolist()


for i in range(0,len(new_text)-1):
	delete_list = re.findall("<U+.*?>",new_text[i])
	#print(delete_list)
	for el in delete_list:
		stringa_da_testare=el[3]+el[4]
		if stringa_da_testare=='00':
			stringa_da_convertire=el[5]+el[6]
			carattere_decimale=int(stringa_da_convertire,16)
			#print(carattere_decimale)
			carattere_ascii=str(chr(carattere_decimale))
			#print(carattere_ascii)
			new_text[i]=new_text[i].replace(el,carattere_ascii)
			#print(new_text[i])
	#print(elem)

	
for i in range(0,len(new_text)-1):
	new_text[i]=new_text[i].replace("<U+2018>","'")
	new_text[i]=new_text[i].replace("<U+2019>","'")
	new_text[i]=new_text[i].replace("<U+2022>",".")
	new_text[i]=new_text[i].replace("<U+2026>","...")
	new_text[i]=new_text[i].replace("<U+201C>","<")
	new_text[i]=new_text[i].replace("<U+201D>",">")
	delete_list=re.findall("<U+.*?>",new_text[i])
	for el in delete_list:
		new_text[i]=new_text[i].replace(el, " ")
	#print(new_text[i])	
	

df['title']=new_text	
#print(df)


print("fine della modifica dei titoli")


#df.to_csv('csv_cleaned_Alfredo.csv', quotechar='~')
df.to_csv('csv_cleaned_Alfredo.csv')

