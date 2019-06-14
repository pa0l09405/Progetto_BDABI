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

#for multilanguage
#nlp=spacy.load("xx_ent_wiki_sm")

df=pd.read_csv("./csv_cleaned_Alfredo 2.csv")
df=df.drop_duplicates()

print(df)

df2=pd.read_csv("./new_features.csv")
print(df2)

df_merged = pd.concat([df, df2], axis=1)
print(df_merged)

df_merged.to_csv(".\merged.csv")