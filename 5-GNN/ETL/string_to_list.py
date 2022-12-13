import ast
import pandas as pd

df = pd.read_csv('ETL\DB-Output.csv')

def string_to_list(input_string: str) -> list:
  # Use the ast (abstract syntax tree) module to parse the input string
  parsed = ast.literal_eval(input_string)
  return parsed


input_string = "[['a', 'b', 'c'],['d', 'e', 'f']]"
node_list = string_to_list(df['Initial Solution'][0])


#lista delle labels
label=[]
for i in range(len(df)):
    if df['OF_Diff'][i]>0:label.append(1)
    else: label.append(0)


# define the input string
string_input = df['Moves'][0]

# remove the square brackets and split the string on the commas to get a list of strings
string_list = string_input[1:-1].split(", ")

# print the resulting list of strings
print(string_list[0].replace("'",""))





