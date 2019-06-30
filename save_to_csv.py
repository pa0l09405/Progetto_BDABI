from pymongo import MongoClient
from pandas import DataFrame
import csv
import numpy as np
import pandas as pd
import pymongo

client = MongoClient('mongodb://localhost:27017/')
db = client.prova
collection = db.coll
df = DataFrame(collection.find())
print(client.prova.coll.find().count())
print(df)
df.to_csv('data_with_new_features.csv')