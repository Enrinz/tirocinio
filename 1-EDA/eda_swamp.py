import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('DB-Output.csv')


#PRINT INFO ABOUT DATASET

#print(df.head())
#print(df.info())
#print(df.describe())
#print(df.duplicated().sum())
#print(len(df['Initial Solution'].unique()))
#print(df['Moves'].unique())
#sns.countplot(df['Moves']).unique()
#print(df.isnull().sum())
#print(df.dtypes)



#Filter data
print(df[df['OFIS']>=100].head())


#Correlation 
print(df.corr())
#Correlation plot
sns.heatmap(df.corr())
plt.show()