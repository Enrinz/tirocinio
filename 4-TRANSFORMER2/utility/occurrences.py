import pandas as pd
import numpy as np
import csv

df = pd.read_csv('DB-Output.csv')

print(df['Moves'].value_counts())
#print(df['Initial Solution'].value_counts()) 