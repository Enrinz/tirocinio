
#https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_with_transformer.py
import csv
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from re import X
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



#Process the data before training

moves_string=[]
label=[]
moves=[]
with open("input_1_balanced_shuffled.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        moves_string.append(row[0])
        label.append(row[1])
for i in range (len(moves_string)):
    moves.append(moves_string[i].replace("[","").replace("]","").replace("\'","").split(","))


vocab_size = 30
embedding_dim = 8
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



embed_dim = 32  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="softmax")(x)
print(outputs)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer="adam", loss="binary_crossentropy",metrics=["accuracy",tf.keras.metrics.TruePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

history = model.fit(
    x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val), verbose=2
)
