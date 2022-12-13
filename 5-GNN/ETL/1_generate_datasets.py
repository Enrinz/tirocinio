import pandas as pd
import csv
df = pd.read_csv('ETL\DB-Output.csv')


def filter_df(df, string):
  mask = df.apply(lambda x: x.str.contains(string, case=False))
  return df[mask]

moves=[]
for i in range(len(df)):
    string_input = df['Moves'][i]
    string_list = string_input[1:-1].split(", ")
    for i in range(len(string_list)):
        string_list_clean=string_list[i].replace("'","")
        moves.append(string_list_clean)

def unique(lst):
  unique_set = set()
  for elem in lst:
    unique_set.add(elem)
  return list(unique_set)

moves_unique=unique(moves)

root='ETL\\splitted_datasets\\'
for i in range(len(moves_unique)):
    filtered_df = df[df['Moves'].str.contains(moves_unique[i])]
    path=root+moves_unique[i]+".csv"
    filtered_df.to_csv(path, index=False)
    