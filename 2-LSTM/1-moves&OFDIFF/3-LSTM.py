#
import csv
from email.errors import MultipartConversionError
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
import keras

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        rand_int = tf.random.uniform((), 0, 2, dtype=tf.int32)
        print(rand_int)
        

vocab_size = 18
embedding_dim = 64
max_length = 4
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

moves_string=[]
label=[]
moves=[]
with open("input\input_1.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        moves_string.append(row[0])
        label.append(row[1])

for i in range (len(moves_string)):
    moves.append(moves_string[i].replace("[","").replace("]","").replace("\'","").split(","))

#counter delle occorrenze per calcolare la dimensione del vocabolario
counter=Counter(x for xs in moves for x in set(xs))

#Start training 
train_size = int(len(moves) * training_portion)
train_moves = moves[0: train_size]
train_label = label[0: train_size]
validation_moves = moves[train_size:]
validation_label = label[train_size:]


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_moves)
word_index = tokenizer.word_index
dict(list(word_index.items())[0:10])

#trasforma le parole in token numerici
train_sequences = tokenizer.texts_to_sequences(train_moves)


# limita la grandezza dei token a 200, nel nostro caso è 4,aggiunge 0 se mancano elementi, in questo caso maxlength è settato a 4
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#validation
validation_sequences = tokenizer.texts_to_sequences(validation_moves)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#label tokenizer
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(label)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_label))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_label))

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_moves(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


'''print(len(train_padded),len(training_label_seq))
print(train_padded[16924],training_label_seq[16924])'''
print(len(train_padded))
model = Sequential()
model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=(4,1)))
model.add(LSTM(49, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
history=model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_padded,training_label_seq,epochs=10)
