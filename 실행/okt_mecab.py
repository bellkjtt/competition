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

import sys



ar = pd.read_csv('data4.csv' ,encoding='CP949')
arr = pd.read_csv('data5.csv' ,encoding='CP949')

arr1 = ar['label1'].tolist()
arr2 = ar['label2'].tolist()
arr3 = ar['label3'].tolist()
arr4 = arr['label'].tolist()


ans1 = arr['digit_1'].tolist()
ans2 = arr['digit_2'].tolist()
ans3 = arr['digit_3'].tolist()

    
okt = Okt()
mecab = Mecab(dicpath=r"C:/mecab/mecab-ko-dic")

okt1 =[]
okt2 =[]

lenth1 = len(ar)
lenth2 = len(arr)

for i in range(lenth1):
    a = okt.nouns(arr1[i])
    b = okt.nouns(arr2[i])
    c = okt.nouns(arr3[i])
    d = a+b+c
    okt1.append(list(set(d)))

for i in range(lenth2):
    a = okt.nouns(arr4[i])
    okt2.append(list(set(a)))


print('----------------------')


mecab1 =[]
mecab2 =[]
          
for i in range(lenth1):
    a = mecab.nouns(arr1[i])
    b = mecab.nouns(arr2[i])
    c = mecab.nouns(arr3[i])
    d = a+b+c
    mecab1.append(list(set(d)))

for i in range(lenth2):
    a = mecab.nouns(arr4[i])
    mecab2.append(list(set(a)))



mix1 =[]
mix2 =[]

for i in range(lenth1):
    mix1.append(list(set(okt1[i]+mecab1[i]))) #okt1[i]

for i in range(lenth2):
    mix2.append(list(set(okt2[i]+mecab2[i]))) #okt1[i]


dataframe = pd.DataFrame(mix1)
dataframe.to_csv('test_1.csv',header=False, index=False ,mode="w",encoding='euc-kr')

dataframe = pd.DataFrame(mix2)
dataframe.to_csv('test_2.csv',header=False, index=False ,mode="w",encoding='euc-kr')


