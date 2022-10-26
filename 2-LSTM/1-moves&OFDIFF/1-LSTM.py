#https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35
#https://github.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/blob/master/BBC%20News_LSTM.ipynb
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

#counter delle occorrenze per calcolare la dimensione del vocabolario
counter=Counter(x for xs in moves for x in set(xs))
#print(counter)

#Start training 
train_size = int(len(moves) * training_portion)
train_moves = moves[0: train_size]
train_label = label[0: train_size]
validation_moves = moves[train_size:]
validation_label = label[train_size:]
'''
print(train_size)
print(len(train_moves))
print(len(train_label))
print(len(validation_moves))
print(len(validation_label))
'''

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_moves)
word_index = tokenizer.word_index
dict(list(word_index.items())[0:10])

#trasforma le parole in token numerici
train_sequences = tokenizer.texts_to_sequences(train_moves)
print(train_sequences[0])


# limita la grandezza dei token a 200, nel nostro caso è 4,aggiunge 0 se mancano elementi, in questo caso maxlength è settato a 4
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
'''print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))


print(train_padded[10])
'''

#validation
validation_sequences = tokenizer.texts_to_sequences(validation_moves)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
'''
print(len(validation_sequences))
print(validation_padded.shape)'''


#label tokenizer
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(label)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_label))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_label))
'''print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)'''
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_moves(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
'''print(decode_moves(train_padded[10]))
print('---')
print(train_moves[10])

'''

model = tf.keras.Sequential([
    # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    # use ReLU in place of tanh function since they are very good alternatives of each other.
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 6 units and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    tf.keras.layers.Dense(3, activation='softmax')
])
model.summary()
print(set(label))


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 10

history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)
'''def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")'''
