import pandas as pd

df = pd.read_csv('ETL\splitted_datasets\DemandBasedDestroyCustomer.csv')


label=[]
for i in range(len(df)):
    if df['OF_Diff'][i]>0:label.append(1)
    else: label.append(0)

with open("DemandBasedDestroyCustomer_labels.txt", "a") as f:
    for i in label:
        f.write(str(i)+"\n")
    
