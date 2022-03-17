import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

import numpy as np
import itertools

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

list1=[]
list2=[]

f = open("a1.txt", 'r')
lines = f.readlines()

for line in lines:
    line = line.strip()  # 줄 끝의 줄 바꿈 문자를 제거한다.
    list1.append(line)


for i in range(len(list1)):
    word1 = list1[i].split('|')
    list2.append(word1)

arr=np.array(list2)


okt = Okt()


#tokenized_doc = okt.pos(doc)
#tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

#print('품사 태깅 10개만 출력 :',tokenized_doc[:10])
#print('명사 추출 :',tokenized_nouns)

#k = sum(1 for line in df)
#df_0 =pd.read_csv('b.csv')
                                  
#df_0 =df_0.fillna(0)

#i = df_0[df_0['수시로'] == '서브'].index #원하는 행 삭제 가능
#i2 = df_0[df_0['수시로'] == '0'].index
#i3 = df_0[df_0['ex2'] == 'ex2'].index


#df_0 = df_0.drop(i)
#df_0 = df_0.drop(i2)
#df_0 = df_0.drop(i3)
#df_1 = df_0.drop(['수시로'], axis=1) #열 중 맨 위의 이름으로만 삭제 가능
#low = df_1.drop(['ex3','ex4','ex5','5','6','7','8','9'], axis=1)
#middle = df_1.drop(['ex2','ex4','ex5','4','7','8','9'], axis=1)
#high = df_1.drop(['ex2','ex3','ex5','4','5','6','8','9'], axis=1)
#high2 = df_1.drop(['ex2','ex3','ex4','4','5','6','7'], axis=1)
#ll = df_1.drop(['ex2','ex4','ex5','4','7','8','9'], axis=1)
#mm = df_1.drop(['ex2','ex3','ex5','4','5','6','8','9'], axis=1)
#hh = df_1.drop(['ex2','ex3','ex4','4','5','6','7'], axis=1)
