# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:17:55 2021

@author: SREE VARSHAN
"""
import pandas as pd
import numpy as np
import  nltk #natural language tool kit
import re #regular expression -removing the special characters
from  nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras import layers

from sklearn.feature_extraction.text import CountVectorizer

data=pd.read_csv(r'C:\Users\SREE VARSHAN\Desktop\AI and ML course\Project\kindle_reviews update.csv',header=0)

nltk.download("stopwords")
print(list(data.columns))

df=[]
for i in range(0,1000):
    review=data['reviewText'][i]
    #a)remove un neccessary .,
    review=re.sub('[^a-zA-Z]',' ',review)
    #b) lower case the text
    review=review.lower()
    #c)split the text
    review=review.split()
    #4.stemming
    #5. remove stop words
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #6.join the splitted data
    review=' '.join(review)
    df.append(review)
    
#print(df)
    

cv=CountVectorizer(max_features=2000)

x=cv.fit_transform(df).toarray()
print("Outputting Devi")
print(x)

y=data.iloc[0:1000:,3:4].values
print(y)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model=Sequential()
model.add(layers.Dense(units=1000, activation="relu")) # input layer
model.add(layers.Dense(units=2000, activation="relu")) # 1st hidden layer
model.add(layers.Dense(units= 1, activation="softmax")) # output layer

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

model.fit(x_train,y_train,batch_size=128,epochs=1)

model.save('kindle.h5')

