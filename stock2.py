 # Project =  This  program predicts if the stock price of a company will increase or decrease based on the top news headlines.

"""  Author - Trapti Meshram
     Date - 06/12/2022
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
plt.style.use('bmh')
df = pd.read_csv('C:/Users/DELL/Documents/codi/AXISBANK.csv')
#print(df)
df.info()

df = df["Close"]
#print(df)

future_days = 15
df["Predaction"] = df[["Close"]].shift(-future_days)
#df.tail()

X = np.array(df.drop(['Predaction'],1))[:-future_days]
#print(X)

y = np.array(df['Predaction'])[:-future_days]
#print(y)

# Divided data in train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeRegressor().fit(X_train, y_train)

Lr = LinearRegression().fit(X_train, y_train)
 
x_future = df.drop(["Predction"],1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
#print(x_future)

tree_prediction = tree.predict(x_future)
#print(tree_prediction)

Lr_prediction = Lr.predict(x_future)
#print(Lr_prdiction)

# Start predication 1. Tree Predication

prediction = tree_prediction

vaild = df[X.shape[0]:]
vaild["Predications"] = prediction

plt.figure(figsize=(16,8))
plt.tile("Model")
plt.plot(df["Colse"], alpha =0.5, label = 'Pred')
plt.plot(vaild[['Close',"Predications"]])
plt.xlabel('Days')
plt.ylabel('Close')
plt.legend(['Orig','Val','Pred'])
plt.show()

#  2.LinerRegression
prediction = Lr_prediction

vaild = df[X.shape[0]:]
vaild["Predications"] = prediction

plt.figure(figsize=(16,8))
plt.tile("Model")
plt.plot(df["Colse"], alpha =0.5, label = 'Pred')
plt.plot(vaild[['Close',"Predications"]])
plt.xlabel('Days')
plt.ylabel('Close')
plt.legend(['Orig','Val','Pred'])
plt.show()






