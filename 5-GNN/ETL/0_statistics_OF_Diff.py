import pandas as pd
import numpy as np

df = pd.read_csv('ETL\DB-Output.csv')

# Calculate the mean of the column
mean = df['OF_Diff'].mean()

# Calculate the standard deviation of the column
std = df['OF_Diff'].std()

# Calculate the minimum value of the column
min_value = df['OF_Diff'].min()

# Calculate the maximum value of the column
max_value = df['OF_Diff'].max()

# Calculate the median of the column
median = df['OF_Diff'].median()

# Calculate the quantiles of the column
quantiles = df['OF_Diff'].quantile([0.25, 0.5, 0.75])

#print(f'mean: {mean}\nstandard deviation: {std}\nminimum value: {min_value}\nmaximum value: {max_value}\nmedian: {median}\nquantiles: {quantiles}')

zeros=[]
positives=[]
negatives=[]
for index,element in enumerate(df['OF_Diff']):
    if element>0:
        positives.append(element)
    if element<0:
        negatives.append(element)
    if element==0:
        zeros.append(element)


pos=np.array(positives,dtype=np.float64)
neg=np.array(negatives,dtype=np.float64)
mean_pos = np.mean(pos)
mean_neg = np.mean(neg)
std_pos=np.std(pos)
std_neg=np.std(neg)
median_pos=np.median(pos)
median_neg=np.median(neg)
print("\n","media positivi: ",mean_pos,"\n","media negativi: ",mean_neg,"\n","deviazione standard: positivi: ",std_pos,"\n","deviazione standard negativi: ",std_neg,"\n","max positivo: ",np.max(pos),"\n","min positivo: ",np.min(pos),"\n","max negativo: ",np.max(neg),"\n","min negativo: ",np.min(neg),"\n")

much_improved=[]
little_improved=[]
neutral=zeros
little_worse=[]
much_worse=[]

for index,element in enumerate(df['OF_Diff']):
    if element>median_pos:
        much_improved.append(element)
    if element<=median_pos and element>0:
        little_improved.append(element)
    if element<median_neg:
        little_worse.append(element)
    if element>=median_neg and element<0:
        much_worse.append(element)
print(len(much_improved),len(little_improved),len(neutral),len(little_worse),len(much_worse))
    