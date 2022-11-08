import pandas as pd
import numpy as np
import csv

df = pd.read_csv('input_1_shuffled.csv')
header=["moves","label"]
positive=0
negatives=0
list_negative=[]
list_positive=[]
with open('input_1_shuffled_balanced.csv', 'w', encoding='UTF8', newline='') as f:
    writer=csv.writer(f)
    writer.writerow(header)
    for i in df.index:
        row=df['moves'][i] #row è di tipo lista
        label=df['label'][i]
        if(label==1): 
            list_positive=[row,label]
            writer.writerow(list_positive)
            positive=positive+1
        if (label==0 and negatives<4947):
            list_negative=[row,label]
            writer.writerow(list_negative)
            negatives=negatives+1
