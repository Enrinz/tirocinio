import random
import sys
from tkinter.messagebox import YES
import pandas as pd

df = pd.read_csv('input\input_1.csv', delimiter=',')
ds = df.sample(frac=1)
print(ds)
ds.to_csv('input_1_shuffled.csv',index=False)