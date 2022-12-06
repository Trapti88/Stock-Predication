import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

sc = pd.read_csv('C:/Users/DELL/Documents/codi/AXISBANK.csv')
#print(sc)

sc = sc.set_index(pd.DatetimeIndex(sc['Date'].values))
print(sc)

lr = sc.copy()
lr["Numbers"] = list(range(0, len(lr)))

X= np.array(lr[["Numbers"]])

y = lr['Close'].values

liner_model =LinearRegression().fit(X,y)
print("Intercept:", liner_model.intercept_)
print("Slope:", liner_model.coef_)

y_pred = liner_model.coef_ * X + liner_model.intercept_
lr["Pred"] = y_pred 

plt.figure(figsize=(16,8))
plt.title('Close Price History', fontsize=18)
plt.plot(lr['Pred'], alpha =0.5, label = 'Pred')
plt.plot(lr['Close'], alpha =0.5, label = 'Close')
plt.xlabel('Close', fontsize =18)
plt.ylabel('Pred', fontsize=18)
plt.show()

r2_score(lr["Close"], lr["Pred"])

liner_model.coef_ * len(lr)+1 + liner_model.intercept_
