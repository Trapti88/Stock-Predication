import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

df = pd.read_csv('C:/Users/DELL/Documents/codi/AXISBANK.csv')
#print(df)

df = df.set_index(pd.DatetimeIndex(df['Date'].values))
#print(df)

def SMA(data,period=30,column="Close"):
    return data[column].rolling(windows=period).median()

df['SMA20']= SMA(df,20) 
df["SMA50"]= SMA(df,50)   

df["Signal"] = np.where(df["SMA20"] > df["SMA50"], 1 ,0)
df["Position"] = df["Signal"].diff()

df["Buy"] = np.where(df["Position"] ==1 , df["Close"] , np.NaN)
df["Sell"] = np.where(df["Position"] == -1, df["CLose"], np.NAN)

plt.figure(figsize=(16,8))
plt.title('Close Price w/ Buy and Sell Signals', fontsize=18)
plt.plot(df['Close'], alpha =0.5, label = 'Close')
plt.plot(df['SMA20'], alpha =0.5, label = 'SMA20')
plt.plot(df['SMA50'], alpha =0.5, label = 'SMA50')
plt.scatter(df.index, df['Buy'], alpha= 1, label="Buy Signal", marker="^", Color='green')
plt.scatter(df.index, df['Sell'], alpha= 1, label="Selll Signal", marker="v", Color='red')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close', fontsize=18)
plt.show()