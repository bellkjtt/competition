import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

import itertools

from konlpy.tag import Okt
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from pykospacing import Spacing

import sys


spacing = Spacing()

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
mecab = Mecab(dicpath=r"C:/mecab/mecab-ko-dic")

no1=[]
no2=[]
no3=[]

arr3=[]


for i in range(len(arr2)):
      arr3.append(arr2[i][0]+" "+arr2[i][1]+" "+arr2[i][2])

okt1 =[]

for i in range(20):
    a = okt.nouns(arr2[i][0])  
    b = okt.nouns(arr2[i][1])
    c = okt.nouns(arr2[i][2])
    d = okt.nouns(arr2[i][0])+okt.nouns(arr2[i][1])+okt.nouns(arr2[i][2])
    e = okt.nouns(arr3[i])
    okt1.append(list(set(d+e)))

print('----------------------')


mecab1 =[]
#with open('no1.txt','w',encoding='UTF-8') as s:
          
for i in range(20):
    a = mecab.nouns(arr2[i][0])       
    b = mecab.nouns(arr2[i][1])
    c = mecab.nouns(arr2[i][2])
    d = mecab.nouns(arr2[i][0])+okt.nouns(arr2[i][1])+okt.nouns(arr2[i][2])
    e = mecab.nouns(arr3[i])
    mecab1.append(list(set(d+e)))

  
#s.close()


mix1 =[]

for i in range(20):
    mix1.append(list(set(okt1[i]+mecab1[i])))

print(mix1)






