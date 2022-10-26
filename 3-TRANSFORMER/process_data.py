import csv
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
moves_string=[]
label=[]
moves=[]
with open("input_1.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        moves_string.append(row[0])
        label.append(row[1])
for i in range (len(moves_string)):
    moves.append(moves_string[i].replace("[","").replace("]","").replace("\'","").split(","))


vocab_size = 20
embedding_dim = 64
max_length = 4
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8
maxlen=4


#TRAINING moves and labels
train_size = int(len(moves) * training_portion)
train_moves = moves[0: train_size]
train_label = label[0: train_size]

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_moves)
train_sequences = tokenizer.texts_to_sequences(train_moves)
x_train = keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=maxlen)
y_train = np.array(train_label).astype(np.integer)

#VALIDATION moves and labels
validation_moves = moves[train_size:]
validation_label = label[train_size:]

validation_sequences = tokenizer.texts_to_sequences(validation_moves)
x_val= keras.preprocessing.sequence.pad_sequences(validation_sequences, maxlen=maxlen)
y_val=np.array(validation_label).astype(np.integer)

print(x_train[0],y_train[0],"***",x_val[0],y_val[0])