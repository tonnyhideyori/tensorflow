import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv(r"D:\programming\python\pyTrain\tensorflow\FINAL_TF2_FILES\TF_2_Notebooks_and_Data\DATA\fake_reg.csv")
#sns.pairplot(df)
from sklearn.model_selection import train_test_split
X=df.iloc[:,1:3].values
y=df['price'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
"""model=Sequential([Dense(4,activation='relu'),Dense(2,activation='relu'),Dense(1)])"""
model=Sequential()
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mse')
model.fit(x=X_train,y=y_train,epochs=50)
#loss_df=pd.DataFrame(model.history.history)
model.evaluate(X_test,y_test,verbose=0)
model.evaluate(X_train,y_train,verbose=0)
test_pred=model.predict(X_test)
test_pred=pd.Series(test_pred.reshape(300,))
pred_df=pd.DataFrame(y_test,columns=['test_truevalue'])
pred_df=pd.concat([pred_df,test_pred],axis=1)