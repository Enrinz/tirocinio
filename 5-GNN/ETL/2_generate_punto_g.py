import pandas as pd
import ast
import itertools
import os

directory = 'ETL\splitted_datasets'

def string_to_list(input_string: str) -> list:
  # Use the ast (abstract syntax tree) module to parse the input string
  parsed = ast.literal_eval(input_string)
  return parsed

for filename in os.listdir(directory):
  df=pd.read_csv(directory+"\\"+filename)
  for x in range(len(df)):
    node_list = string_to_list(df['Initial Solution'][x])
    merged_node_list=list(itertools.chain.from_iterable(node_list))
    unique_merged_node_list =list(dict.fromkeys(merged_node_list))

    ids=[]
    for i in range (len(unique_merged_node_list)):
      ids.append(i+1)

    dict={}
    for i in range(len(unique_merged_node_list)):
      dict[unique_merged_node_list[i]]=ids[i]

    name=filename.split(".")
    dir_labels="punto_g\\"
    with open(dir_labels+name[0]+".g", "a") as f:
      f.write("XP"+"\n")
      for i in range(len(dict)):
        f.write("v "+str(list(dict.values())[i])+" "+str(list(dict.keys())[i])+"\n")
      for j in range(len(merged_node_list)-1):
        string1 = merged_node_list[j]
        string2 = merged_node_list[j + 1]
        if(string1!=string2 and string1=='D0'):
          f.write("e "+"1"+" "+str(dict.get(string2))+" "+"D0"+"__"+string2+"\n")
        elif(string1!=string2 and string2=='D0'):
          f.write("e "+str(dict.get(string1))+" "+"1"+" "+str(string1)+"__"+"D0"+"\n")
        elif(string1!=string2):
          f.write("e "+str(dict.get(string1))+" "+str(dict.get(string2))+" "+string1+"__"+string2+"\n")
        else:continue
      f.write("\n")
