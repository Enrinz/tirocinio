import pandas as pd
import os


directory = 'ETL\splitted_datasets'

# Loop through the files in the directory
for filename in os.listdir(directory):
    label=[]
    df = pd.read_csv(directory+"\\"+filename)
    for i in range(len(df)):
        if df['OF_Diff'][i]>0:label.append(1)
        else: label.append(0)
    name=filename.split(".")
    dir_labels="labels\\"
    with open(dir_labels+name[0]+".txt", "a") as f:
        for i in label:
            f.write(str(i)+"\n")
    








