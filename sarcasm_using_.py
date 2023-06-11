import os
import pandas as pd
import numpy as np
import nltk
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
data = pd.read_json("E:\ANZ JO\Mini Project\Sarcasm Analyser\Sarcasm_Headlines_Dataset.json", lines = True)
data.drop('article_link', axis = 1, inplace = True)
data.head()
data.drop_duplicates(subset=['headline'], inplace = True)
data.describe(include = 'all')
import re
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

def cleantext(text):
    text = re.sub(r'\d+', '', text)
    text = "".join([char for char in text if char not in string.punctuation])
    return text

data['headline']=data['headline'].apply(cleantext)
data['sentence_len'] = data['headline'].str.len()
max_features = 10000
maxlen = 300
embed_size = 200

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=max_features,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' ', char_level=False)
tokenizer.fit_on_texts(data['headline'])
from keras.utils import pad_sequences
X = tokenizer.texts_to_sequences(data['headline'])
X = pad_sequences(X, maxlen = maxlen)
y = np.asarray(data['is_sarcastic'])
vocab_size=len(tokenizer.word_index)
glove_file = 'E:\ANZ JO\Mini Project\Sarcasm Analyser\glove.6B.200d.txt'
embeddings = {}

for line in open(glove_file , encoding='utf-8',
                 errors='ignore'):
    word = line.split(" ")[0]
    embed = line.split(" ")[1:]
    embed = np.asarray(embed, dtype = 'float32')
    embeddings[word] = embed
embedding_matrix = np.zeros((vocab_size, 200))

for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential

input_layer = Input(shape=(maxlen,),dtype=keras.int64)
embed = Embedding(embedding_matrix.shape[0],output_dim=200,weights=[embedding_matrix],input_length=maxlen, trainable=True)(input_layer)
lstm=Bidirectional(LSTM(128))(embed)
drop=Dropout(0.3)(lstm)
dense =Dense(100,activation='relu')(drop)
out=Dense(2,activation='softmax')(dense)



