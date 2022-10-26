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

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


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
with open("input\input_1_shuffled.csv", 'r') as csvfile:
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
print(len(train_label))
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

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_label))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_label))
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_moves(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


#hyperparam
#https://keras.io/api/layers/core_layers/embedding/
#https://keras.io/api/layers/recurrent_layers/lstm/

def create_model(input_length):
    print ('Creating model...')
    model = Sequential()
    model.add(Embedding(input_dim = 18, output_dim = 18, input_length = input_length))
    model.add(LSTM(units=2, activation='sigmoid' ,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM( units=2,activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    print ('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    return model
''',tf.keras.metrics.FalseNegatives(),tf.keras.metrics.Precision(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives()'''
X_train=train_padded
y_train=training_label_seq
X_test= validation_padded
y_test=validation_label_seq

#print("X_train:\n",X_train,"\n","Y_train:",y_train,"\n","X_test",X_test,"\n","Y_test",y_test)

model = create_model(len(X_train[0]))

print ('Fitting model...')
hist = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split = 0.1, verbose = 1)


'''score, acc = model.evaluate(X_test, y_test, batch_size=1)
print('Test score:', score)
print('Test accuracy:', acc)
'''
yhat = model.predict(X_train)
print(len(yhat),y_train)
