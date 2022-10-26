import pandas as pd
from numpy import var
import statistics
df = pd.read_csv('DB-Output.csv')


#Informazioni del dataset

print(df.info())
print("Numero di istanze:",len(df.index))
print(df.describe())
print(df.dtypes)

#count di istanze che migliorano e non
def count_improved(dataframe):
    improved=0 
    not_improved=0
    pos_OF_Diff=0
    for i in dataframe.index:
        if ((dataframe['OFFS'][i]-dataframe['OFIS'][i])>0):
            improved=improved+1
        else: 
            not_improved=not_improved+1
        if(dataframe['OF_Diff'][i]>0):
            pos_OF_Diff=pos_OF_Diff+1
    print("migliorate:",improved)
    print("non migliorate:",not_improved)
    print("OF_DIFF positiva:",pos_OF_Diff)

def count_null_moves(dataframe):
    count_single_null=0
    count_double_null=0
    count_no_null=0
    count_middle_null=0
    for i in dataframe.index:
        moves=df['Moves'][i]
        if (moves[2:6]=="null"):
            count_single_null=count_single_null+1
        if ((moves[2:6]=="null") and (moves[10:14]=="null")):
            count_double_null=count_double_null+1
        if ("null" in moves[15:]) :
            count_middle_null=count_middle_null+1
        if (moves.find("null")==-1):
            count_no_null=count_no_null+1
    print("Single null beginning:",count_single_null,"\a","Double null beginning:",count_double_null,"\a","Middle null:",count_middle_null,"\a","No null element:",count_no_null)


def Average(lst):
    return sum(lst) / len(lst)
def metrics(dataframe):
    num_elem=[]
    for i in dataframe.index:
        moves=df['Moves'][i]
        num=moves.count(",")+1
        num_elem.append(num)
    print("Media:",Average(num_elem),"Deviazione Standard:",statistics.pstdev(num_elem),"Varianza:",var(num_elem),"Min:",min(num_elem),"Max:",max(num_elem))

count_improved(df)
count_null_moves(df)
metrics(df)