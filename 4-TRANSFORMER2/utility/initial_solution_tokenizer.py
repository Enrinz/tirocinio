import pandas as pd
import numpy as np
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras


df = pd.read_csv('DB-Output.csv')

vocab_size = 1000
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
maxlen=250

moves_number=[]
for i in range(len(df['Moves'])):
    moves_number.append(hash(df['Moves'][i]))

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(df['Initial Solution'])
init_sol_seq = tokenizer.texts_to_sequences(df['Initial Solution'])
init_sol = keras.preprocessing.sequence.pad_sequences(init_sol_seq, maxlen=maxlen)
init_sol_list=init_sol.tolist()


for i in range(len(df)):
    init_sol_list[i].append(moves_number[i])

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(init_sol_list)
init_sol_seq = tokenizer.texts_to_sequences(init_sol_list)
init_sol = keras.preprocessing.sequence.pad_sequences(init_sol_seq, maxlen=maxlen)
init_sol_list=init_sol.tolist()



print(init_sol_list)





