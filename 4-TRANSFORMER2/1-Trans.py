
#https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_with_transformer.py
import csv
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from re import X
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K

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


#Implement a Transformer block as a layer

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)



# Implement embedding layer
#Two seperate embedding layers, one for tokens, one for token index (positions).



class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


df = pd.read_csv('DB-Output.csv')
#Process the data before training
label=[]
for i in range(len(df)):
    if df['OF_Diff'][i]>0:label.append(1)
    else: label.append(0)


vocab_size = 200
embedding_dim = 32
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.8
maxlen=250
#lista con tutte le moves
moves=[]
for i in range(len(df['Moves'])):
    moves.append(df['Moves'][i])

#lista con tutte le soluzioni iniziali
solution=[]
for i in range(len(df['Initial Solution'])):
    solution.append(df['Initial Solution'][i])

#coppia soluzione iniziale-algoritmo moves
sol_moves=[]
for i in range(len(df)):
    sol_moves.insert(i,solution[i]+moves[i])



tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sol_moves)
init_sol_seq = tokenizer.texts_to_sequences(sol_moves)
init_sol = keras.preprocessing.sequence.pad_sequences(init_sol_seq, maxlen=maxlen)
init_sol_list=init_sol.tolist()

print(len(init_sol_list))

#TRAINING moves and labels
train_size = int(len(init_sol_list) * training_portion)
train_moves = init_sol_list[0: train_size]
train_label = label[0: train_size]

#print(train_moves,type(train_moves),train_moves[0],type(train_moves[0]))


x_train = np.array([np.array(xi) for xi in train_moves])
y_train = np.array(train_label).astype(np.integer)

#print(x_train,type(x_train),x_train[0],type(x_train[0]))


#VALIDATION moves and labels
validation_moves = init_sol_list[train_size:]
validation_label = label[train_size:]

#tokenizer.fit_on_texts(validation_moves)
x_val=np.array([np.array(xi) for xi in validation_moves])
y_val=np.array(validation_label).astype(np.integer)



embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(100, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer="Adam", loss="sparse_categorical_crossentropy",metrics=["accuracy",precision_m,recall_m,f1_m])
history = model.fit(
    x_train, y_train, batch_size=32, epochs=30, validation_data=(x_val, y_val)
)

