import pandas as pd
import numpy as np
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras
df = pd.read_csv('DB-Output.csv')


def count_1(dataframe):
    count_1=0
    for i in range(len(dataframe)):
        if dataframe['OF_Diff'][i]>0:
            count_1+=1
    return(count_1)

algorithms=df['Moves'].value_counts()
algorithms_df=algorithms.to_frame()

pos=[]
for i in range(len(algorithms_df)):
    grouped = df.groupby(df.Moves)
    df_new = grouped.get_group(algorithms_df.index[i])
    shuffled_df=df_new.sample(frac=1).reset_index(drop=True)
    count=count_1(shuffled_df)
    for i in range(len(shuffled_df)):
        if(shuffled_df['OF_Diff'][i]>0):
            shuffled_df.loc[[i]].to_csv('balance_dataset\\balanced.csv', mode='a', index=False, header=False)
    for i in range(len(shuffled_df)):
        if(count>0 and shuffled_df['OF_Diff'][i]<=0):
            shuffled_df.loc[[i]].to_csv('balance_dataset\\balanced.csv', mode='a', index=False, header=False)
            count-=1


def count_1_0(dataframe):
    count_1=0
    count_0=0
    for i in range(len(dataframe)):
        if dataframe['OF_Diff'][i]>0:
            count_1+=1
        else: count_0+=1
    print(count_1,count_0)

df_bal = pd.read_csv('balance_dataset\\balanced.csv')
count_1_0(df_bal)