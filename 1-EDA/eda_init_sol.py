import pandas as pd
from numpy import var
import statistics
df = pd.read_csv('DB-Output.csv')
def Average(lst):
    return sum(lst) / len(lst)
def metrics(dataframe):
    num_elem=[]
    for i in dataframe.index:
        init_sol=df['Initial Solution'][i]
        num=init_sol.count(",")+1
        num_elem.append(num)
    print("Media:",Average(num_elem),"Deviazione Standard:",statistics.pstdev(num_elem),"Varianza:",var(num_elem),"Min:",min(num_elem),"Max:",max(num_elem))
metrics(df)