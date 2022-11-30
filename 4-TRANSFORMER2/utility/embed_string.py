import pandas as pd
import numpy as np
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras
df = pd.read_csv('DB-Output.csv')

moves_number=[]
for i in range(len(df['Moves'])):
    moves_number.append(hash(df['Moves'][i]))

def countOccurrence(a):
  k = {}
  for j in a:
    if j in k:
      k[j] +=1
    else:
      k[j] =1
  return k



