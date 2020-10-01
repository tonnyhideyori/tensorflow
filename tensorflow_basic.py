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
model=Sequential()
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mse')
model.fit(x=X_train,y=y_train,epochs=250)
#loss_df=pd.DataFrame(model.history.history)
model.evaluate(X_test,y_test,verbose=0)
model.evaluate(X_train,y_train,verbose=0)
test_pred=model.predict(X_test)
test_pred=pd.Series(test_pred.reshape(300,))
pred_df=pd.DataFrame(y_test,columns=['test_truevalue'])
pred_df=pd.concat([pred_df,test_pred],axis=1)
pred_df.columns=['true Y','pred']
sns.scatterplot(x='true Y',y='pred',data=pred_df)
from sklearn.metrics import mean_absolute_error,mean_squared_error

mean_absolute_error(pred_df['true Y'],pred_df['pred'])
mean_squared_error(pred_df['true Y'],pred_df['pred'])
new_gem=[[998,1000]]
new_gem=scaler.transform(new_gem)
model.predict(new_gem)
from tensorflow.keras.models import load_model
model.save('my_gem.h5')
later=load_model('my_gem.h5')