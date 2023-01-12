import pandas as pd
import numpy as np
import os


directory = 'ETL\splitted_datasets'
for filename in os.listdir(directory):
    label=[]
    df = pd.read_csv(directory+"\\"+filename)
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
    #print("\n","media positivi: ",mean_pos,"\n","media negativi: ",mean_neg,"\n","deviazione standard: positivi: ",std_pos,"\n","deviazione standard negativi: ",std_neg,"\n","max positivo: ",np.max(pos),"\n","min positivo: ",np.min(pos),"\n","max negativo: ",np.max(neg),"\n","min negativo: ",np.min(neg),"\n")

    much_improved=[]
    little_improved=[]
    neutral=zeros
    little_worse=[]
    much_worse=[]

    for index,element in enumerate(df['OF_Diff']):
        if element>median_pos:
            much_improved.append(element)
            label.append(4)
        if element<=median_pos and element>0:
            little_improved.append(element)
            label.append(3)
        if element==0:label.append(2)
        if element<median_neg:
            little_worse.append(element)
            label.append(1)
        if element>=median_neg and element<0:
            much_worse.append(element)
            label.append(0)
        
    print(len(much_improved),len(little_improved),len(neutral),len(little_worse),len(much_worse))
    name=filename.split(".")
    dir_labels="5_labels_median\\"
    with open(dir_labels+name[0]+".txt", "a") as f:
        for i in label:
            f.write(str(i)+"\n")

