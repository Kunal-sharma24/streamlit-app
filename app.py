import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model # type: ignore

st.title('Stock Price Predcition')
user_input=st.text_input('Enter Stock Ticker' , 'SBIN.BO')
df = yf.download(user_input, start='2010-01-01', end=pd.to_datetime('today').strftime('%Y-%m-%d'))

st.subheader('Data from 2010-01-01 to TODAY')
st.write(df.describe())

st.subheader('Closing Time vs TIme chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.xlabel("date")
plt.ylabel("price")
plt.grid(True)
st.pyplot(fig)

ma100=df.Close.rolling(100).mean()
st.subheader('Closing time vs Time chart with 100MA')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.plot(ma100 , 'r')
plt.xlabel("date")
plt.ylabel("price")
plt.grid(True)
st.pyplot(fig)

ma200=df.Close.rolling(200).mean()
st.subheader('Closing time vs Time chart with 100MA and 200MA')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.plot(ma100 , 'r')
plt.plot(ma200 , 'g')
plt.xlabel("date")
plt.ylabel("price")
plt.grid(True)
st.pyplot(fig)
# yeha tak sahi ha

training_data=pd.DataFrame(df['Close'][:int(len(df)*0.70)])
testing_data=pd.DataFrame(df['Close'][int(len(df)*0.70):])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(training_data)

xtrain=[]
ytrain=[]
for i in range(100,data_training_array.shape[0]):
      xtrain.append(data_training_array[i-100:i])
      ytrain.append(data_training_array[i,0])
xtrain,ytrain=np.array(xtrain),np.array(ytrain)  

model=load_model('stock.keras')

last_100_days = training_data.tail(100)
past_100_days=pd.DataFrame(testing_data[-100:0])
test_df=pd.DataFrame(testing_data)
final_df=pd.concat([test_df,past_100_days],ignore_index=True)
input_data=scaler.fit_transform(final_df)

xtest=[]
ytest=[]
for i in range(100,input_data.shape[0]):
    xtest.append(input_data[i-100:i])
    ytest.append(input_data[i,0])
xtest,ytest=np.array(xtest),np.array(ytest)

ypred=model.predict(xtest)

scaler=scaler.scale_

scale_factor=1/scaler[0]
ypred=ypred*scale_factor
ytest=ytest*scale_factor

st.subheader('Predictions vs Original: ')
fig2 = plt.figure(figsize=(12,6))
plt.plot(ytest,'b',label='original price')
plt.plot(ypred,'r',label='predicted price')
plt.xlabel("date")
plt.ylabel("price")
plt.legend()
plt.grid(True)
st.pyplot(fig2)