import pandas as pd
import numpy as np
import csv

df = pd.read_csv('DB-Output.csv')


#create csv file with list of moves and label 1 if OF_DIFF>0, 0 otherwise
def label(num):
    if num>0:
        return 1
    else: return 0

header=["moves","label"]

with open('input_1.csv', 'w', encoding='UTF8', newline='') as f:
    writer=csv.writer(f)
    writer.writerow(header)

    for i in df.index:
        row=df['Moves'][i] #row è di tipo lista
        labels=label(df['OF_Diff'][i])
        writer.writerow([row,labels])


