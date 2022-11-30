import pandas as pd
import numpy as np
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras
df = pd.read_csv('DB-Output.csv')

vocab_size = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
maxlen=4

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(df['Moves'])
moves_sequences = tokenizer.texts_to_sequences(df['Moves'])
moves = keras.preprocessing.sequence.pad_sequences(moves_sequences, maxlen=maxlen)


