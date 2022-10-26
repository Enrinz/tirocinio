#https://gist.github.com/urigoren/b7cd138903fe86ec027e715d493451b4

from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import pandas as pd
import numpy as np
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa


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
with open("input\input_1_balanced.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        moves_string.append(row[0])
        label.append(row[1])

for i in range (len(moves_string)):
    moves.append(moves_string[i].replace("[","").replace("]","").replace("\'","").split(","))

counter=Counter(x for xs in moves for x in set(xs))
train_size = int(len(moves) * training_portion)
train_moves = moves[0: train_size]
train_label = label[0: train_size]

validation_moves = moves[train_size:]
validation_label = label[train_size:]
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_moves)
word_index = tokenizer.word_index
dict(list(word_index.items())[0:10])
train_sequences = tokenizer.texts_to_sequences(train_moves)


train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#validation
validation_sequences = tokenizer.texts_to_sequences(validation_moves)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#label tokenizer
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(label)

#label of training e validation
training_label_seq = np.array(train_label).astype(np.integer)
validation_label_seq = np.array(validation_label).astype(np.integer)

#hyperparam
#https://keras.io/api/layers/core_layers/embedding/
#https://keras.io/api/layers/recurrent_layers/lstm/

def create_model(input_length):
    print ('Creating model...')
    model = Sequential()
    model.add(Embedding(input_dim = 50, output_dim = 50, input_length = input_length))
    model.add(LSTM(units=2, activation='relu' ,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM( units=2,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    print ('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives(),tf.keras.metrics.FalsePositives()])

    return model

X_train=train_padded
y_train=training_label_seq
X_test= validation_padded
y_test=validation_label_seq
print(len(X_train),len(y_train),len(X_test),len(y_test))
model = create_model(len(X_train[0]))
#print(len(X_train[0]))

print ('Fitting model...')
hist = model.fit(X_train, y_train, batch_size=16, epochs=20, verbose = 2)

#validation_split = 0.1,
'''score, acc = model.evaluate(X_test, y_test, batch_size=1)
print('Test score:', score)
print('Test accuracy:', acc)
'''
