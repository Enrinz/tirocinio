import pandas as pd
import numpy as np
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras
df = pd.read_csv('DB-Output.csv')


algorithms=df['Moves'].value_counts()
algorithms_df=algorithms.to_frame()
print(algorithms_df.index[0])
path='separete_alg\input_splitted\\'
for i in range(len(algorithms_df)):
    grouped = df.groupby(df.Moves)
    df_new = grouped.get_group(algorithms_df.index[i])
    df_new.to_csv(path+str(i)+'.csv', index=False)