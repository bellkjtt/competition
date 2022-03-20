import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Embedding, Dense, LSTM,Flatten,Bidirectional,Dropout, Conv1D, GlobalMaxPooling1D, Input,Concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint




from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

import itertools

from konlpy.tag import Okt
from konlpy.tag import Mecab
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from pykospacing import Spacing

import gensim
import urllib.request
from gensim.models.word2vec import Word2Vec


spacing = Spacing()


train_data = pd.read_csv('data6.csv' ,encoding='CP949')

ans1 = train_data['digit_1'].tolist()
ans2 = train_data['digit_2'].tolist()
ans3 = train_data['digit_3'].tolist()

a1 = train_data['1'].tolist()
a2 = train_data['2'].tolist()
a3 = train_data['3'].tolist()
a4 = train_data['4'].tolist()
a5 = train_data['5'].tolist()
a6 = train_data['6'].tolist()
a7 = train_data['7'].tolist()
a8 = train_data['8'].tolist()
a9 = train_data['9'].tolist()
a10 = train_data['10'].tolist()
a11 = train_data['11'].tolist()
a12 = train_data['12'].tolist()
a13 = train_data['13'].tolist()
a14 = train_data['14'].tolist()
a15 = train_data['15'].tolist()
a16 = train_data['16'].tolist()
a17 = train_data['17'].tolist()

b1 = np.array(a1).T
b2 = np.array(a2).T
b3 = np.array(a3).T
b4 = np.array(a4).T
b5 = np.array(a5).T
b6 = np.array(a6).T
b7 = np.array(a7).T
b8 = np.array(a8).T
b9 = np.array(a9).T
b10 = np.array(a10).T
b11 = np.array(a11).T
b12 = np.array(a12).T
b13 = np.array(a13).T
b14 = np.array(a14).T
b15 = np.array(a15).T
b16 = np.array(a16).T
b17 = np.array(a17).T

c1 =np.vstack([b1,b2])
c1 =np.vstack([c1,b3])
c1 =np.vstack([c1,b4])
c1 =np.vstack([c1,b5])
c1 =np.vstack([c1,b6])
c1 =np.vstack([c1,b7])
c1 =np.vstack([c1,b8])
c1 =np.vstack([c1,b9])
c1 =np.vstack([c1,b10])
c1 =np.vstack([c1,b11])
c1 =np.vstack([c1,b12])
c1 =np.vstack([c1,b13])
c1 =np.vstack([c1,b14])
c1 =np.vstack([c1,b15])
c1 =np.vstack([c1,b16])
c1 =np.vstack([c1,b17])

d = c1.T

e = d.tolist()

idx_encode = preprocessing.LabelEncoder()
idx_encode.fit(ans1)

label_train = idx_encode.transform(ans1) # 주어진 고유한 정수로 변환

label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(e)

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

vocab_size = total_cnt - rare_cnt + 1
print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(e)

sequences = tokenizer.texts_to_sequences(e)













max_len = 10

intent_train = pad_sequences(sequences, maxlen = max_len)
label_train = to_categorical(np.asarray(label_train))


indices = np.arange(intent_train.shape[0])
np.random.shuffle(indices)
print('랜덤 시퀀스 :',indices)

n_of_val = int(0.1 * intent_train.shape[0])
print('검증 데이터의 개수 :',n_of_val)


X_train = intent_train[:-n_of_val]
y_train = label_train[:-n_of_val]
X_val = intent_train[-n_of_val:]
y_val = label_train[-n_of_val:]

print('훈련 데이터의 크기(shape):', X_train.shape)
print('검증 데이터의 크기(shape):', X_val.shape)
print('훈련 데이터 레이블의 크기(shape):', y_train.shape)
print('검증 데이터 레이블의 크기(shape):', y_val.shape)





word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

print('모델의 크기(shape) :',word2vec_model.vectors.shape) # 모델의 크기 확인


embedding_dim = 300

embedding_matrix = np.zeros((vocab_size,embedding_dim))
print('임베딩 행렬의 크기(shape) :',np.shape(embedding_matrix))

def get_vector(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None

for word, index in tokenizer.word_index.items():
    # 단어와 맵핑되는 사전 훈련된 임베딩 벡터값
    vector_value = get_vector(word)
    if vector_value is not None:
        embedding_matrix[index] = vector_value



hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(128, return_sequences=True,
               input_shape=(8, 16)))
model.add(LSTM(128, return_sequences=True)) 
model.add(LSTM(hidden_units))
model.add(Dense(19, activation='softmax'))


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('predict_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['acc'])

history = model.fit(X_train, y_train,
          batch_size=17,
          epochs=1,
          validation_data=(X_val, y_val))

model.save('predict_model.h5')
loaded_model = load_model('predict_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_train, y_train)[1]))

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['acc'])
plt.plot(epochs, history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


y_predicted = model.predict(X_train)
y_predicted = y_predicted.argmax(axis=-1) # 예측을 정수 시퀀스로 변환

print('정확도(Accuracy) : ', sum(y_predicted == y_test) / len(y_test))

