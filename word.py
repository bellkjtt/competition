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

import sys

list1=[]
list2=[]

f = open("a1.txt", 'r')
lines = f.readlines()

for line in lines:
    line = line.strip()  # 줄 끝의 줄 바꿈 문자를 제거한다.
    list1.append(line)

characters =".,!?'()"


for i in range(len(list1)):
    word1 = list1[i].replace(characters[0]," ")
    word1 = word1.replace(characters[1]," ")
    word1 = word1.replace(characters[2]," ")
    word1 = word1.replace(characters[3]," ")
    word1 = word1.replace(characters[4]," ")
    word1 = word1.replace(characters[5]," ")
    word1 = word1.replace(characters[6]," ")
    word1 = word1.split('|')
    
    list2.append(word1)

arr=np.array(list2)
arr=np.delete(arr,(0),axis=0)
arr=np.delete(arr,(0),axis=1)

ans1= []
ans2= []
ans3= []
arr2 =[]

for i in range(len(arr)):
    ans1.append(arr[i][0])
    ans2.append(arr[i][1])
    ans3.append(arr[i][2])
    arr2.append(arr[i][3:6])
    

okt = Okt()

no1=[]
no2=[]
no3=[]

arr3=[]


for i in range(len(arr2)):
      arr3.append(arr2[i][0]+" "+arr2[i][1]+" "+arr2[i][2])

s = open('no1.txt','w')


for i in range(20):
    a = okt.nouns(arr2[i][0])
    
    b = okt.nouns(arr2[i][1])
    c = okt.nouns(arr2[i][2])
    print(okt.nouns(arr2[i][0]),okt.nouns(arr2[i][1]),okt.nouns(arr2[i][2]))
    print(okt.nouns(arr3[i]))

	    
s.close()

#for i in range(len(arr2)):
    #no1.append(okt.nouns(arr2[i][0]))
    #no2.append(okt.nouns(arr2[i][1]))
    #no3.append(okt.nouns(arr2[i][2]))


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
