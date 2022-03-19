import numpy as np
import pandas as pd
from pandas import DataFrame

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

ans4=[]
arr3=[]


for i in range(len(arr2)):
      arr3.append(arr2[i][0]+" "+arr2[i][1]+" "+arr2[i][2])

arr4=[]

for i in range(len(arr3)):
    arr4.append([ans1[i],ans2[i],ans3[i],arr3[i]])

dataframe = pd.DataFrame(arr4)
dataframe.to_csv('test.csv',header=False, index=False ,mode="w",encoding='euc-kr')


