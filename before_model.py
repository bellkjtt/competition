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


spacing = Spacing()


train_data = pd.read_csv('data1.csv' ,encoding='CP949')
train_data2 = pd.read_csv('data3.csv' ,encoding='CP949')

arr1 = train_data['label1'].tolist()
arr2 = train_data['label2'].tolist()
arr3 = train_data['label3'].tolist()
arr4= train_data2['label'].tolist()

ans1 = train_data['digit_1'].tolist()
ans2 = train_data['digit_2'].tolist()
ans3 = train_data['digit_3'].tolist()


train_data.drop_duplicates(subset=['label3'], inplace=True)
print(train_data.isnull().values.any())
print(train_data.isnull().sum())
train_data.loc[train_data.label1.isnull()]
train_data.loc[train_data.label2.isnull()]
train_data.loc[train_data.label3.isnull()]
train_data = train_data.dropna(how = 'any')


train_data2.drop_duplicates(subset=['label'], inplace=True)
print(train_data2.isnull().values.any())
print(train_data2.isnull().sum())
train_data2.loc[train_data2.label.isnull()]
train_data2 = train_data2.dropna(how = 'any')


train_data['label1'] = train_data['label1'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

train_data['label2'] = train_data['label2'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

train_data['label3'] = train_data['label3'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

train_data2['label'] = train_data2['label'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

dataframe = pd.DataFrame(train_data)
dataframe.to_csv('test_1.csv',header=False, index=False ,mode="w",encoding='euc-kr')

dataframe = pd.DataFrame(train_data2)
dataframe.to_csv('test_2.csv',header=False, index=False ,mode="w",encoding='euc-kr')
