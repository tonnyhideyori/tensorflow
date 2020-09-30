# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 07:56:24 2020

@author: edwayne
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r'D:\programming\python\pyTrain\tensorflow\FINAL_TF2_FILES\TF_2_Notebooks_and_Data\DATA\kc_house_data.csv')

"""feature engineering, dropping irrelevant features"""
df=df.drop('id',axis=1)
#converting date to date object
df['date']=pd.to_datetime(df['date'])
df['year']=df['date'].apply(lambda date:date.year)
df['month']=df['date'].apply(lambda date:date.month)

#now we trying to see if price of the house relates to date either year or month
sns.boxplot(x='month',y='price',data=df)
df.groupby('month').mean()['price'].plot()
df.groupby('year').mean()['price'].plot()
#we drop date feature
df=df.drop('date',axis=1)
df=df.drop('zipcode',axis=1)

X=df.drop('price',axis=1).values
y=df['price'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#model definition and compliation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=400)
losses=pd.DataFrame(model.history.history)
losses.plot()
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score
predicts=model.predict(X_test)
mean_squared_error(y_test,predicts)
mean_absolute_error(y_test,predicts)
#now we look at dataframe to see the correlation between the mean price and the mean absolute error 
df['price'].describe()
explained_variance_score(y_test,predicts)
plt.scatter(y_test,predicts)
#predict unseen house 
new_house=df.drop('price',axis=1).iloc[0]
new_house=scaler.transform(new_house.values.reshape(-1,19))
model.predict(new_house)