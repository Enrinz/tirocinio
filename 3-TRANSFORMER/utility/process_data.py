import csv
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
moves_string=[]
label=[]
moves=[]
with open("inputs/input_1_balanced_shuffled.csv", 'r') as csvfile:
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
############################################################################################
count_train_1=0
count_train_0=0
for i in train_label:
    if i=="1" or i==1:
        count_train_1=count_train_1+1
    elif i=="0" or i==0:
         count_train_0=count_train_0+1
print("train 0: ",count_train_0,"train 1:",count_train_1)

count_val_1=0
count_val_0=0
for i in validation_label:
    if i=="1" or i==1:
        count_val_1=count_val_1+1
    elif i=="0" or i==0:
         count_val_0=count_val_0+1
print("val 0: ",count_val_0,"val 1: ",count_val_1)
