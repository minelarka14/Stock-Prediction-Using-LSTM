import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import requests
import datetime
import yfinance as yf
from tkinter import *
import ssl

root = Tk()
root.title("LSTM Scout Open Theme Badge")
e = Entry(root)
e.grid(row=0, column=0)


def mainfc():
  print("Running")
  ssl._create_default_https_context = ssl._create_unverified_context
  plt.style.use('dark_background')
  yesterday = datetime.date.today() - datetime.timedelta(days=1)
  stime = yesterday.strftime("%d/%m/%Y")
  utime = datetime.datetime.strptime(stime, "%d/%m/%Y").timestamp()
  symbol = e.get()
  tst = pd.read_csv(
    "https://query1.finance.yahoo.com/v7/finance/download/" + str(symbol) + "?period1=1546300800&period2=" + str(
      round(utime)) + "&interval=1d&events=history", date_parser=True)
  data = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/' + str(
    symbol) + '?period1=1277251200&period2=1592438400&interval=1d&events=history', date_parser=True)
  data_training = data[data['Date'] < '2019-01-01'].copy()  # 2010 to 2019  946684800
  training_data = data_training.drop(['Date', 'Adj Close'], axis=1)
  scaler = MinMaxScaler()
  training_data = scaler.fit_transform(training_data)
  print(training_data.shape)
  X_train = []
  y_train = []
  for i in range(60, training_data.shape[0]):
    X_train.append(training_data[i - 60:i])
    y_train.append(training_data[i, 0])
  X_train, y_train = np.array(X_train), np.array(y_train)
# Build Model 4 layers
  regressior = Sequential()
  regressior.add(LSTM(units=60, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 5)))
  regressior.add(Dropout(0.2))
  regressior.add(LSTM(units=60, activation='relu', return_sequences=True))
  regressior.add(Dropout(0.3))
  regressior.add(LSTM(units=80, activation='relu', return_sequences=True))
  regressior.add(Dropout(0.4))
  regressior.add(LSTM(units=120, activation='relu'))
  regressior.add(Dropout(0.5))
  regressior.add(Dense(units=1))
  regressior.compile(optimizer='adam', loss='mean_squared_error')

  regressior.fit(X_train, y_train, epochs=1, batch_size=64)

  p6d = tst.tail(60)
  df = p6d.append(tst, ignore_index=True)
  df = df.drop(['Date', 'Adj Close'], axis=1)
  inputs = scaler.transform(df)
  x_test = []
  y_test = []
  for i in range(60, inputs.shape[0]):
    x_test.append(inputs[i - 60:i])
    y_test.append(inputs[i, 0])
  x_test, y_test = np.array(x_test), np.array(y_test)
  y_pred = regressior.predict(x_test)
  cnt = y_pred.size - 1
  rp = yf.Ticker(symbol).history(period="1d")['Open'][0]
  scale = 1 / 5.08196485e-03
  y_pred = y_pred * scale
  y_test = y_test * scale
  ypstr = int(y_pred[cnt])
  if int(ypstr) >= rp:
    bs = "Undervalued"
  else:
    bs = "Overvalued"
  title = "Current Price: " + str(rp) + "\n Predicted Price: " + str(ypstr) + "\n Indication: " + str(bs)

# Visualisation (Delete if not needed)
  plt.figure(figsize=(14, 6))
  plt.plot(y_test, color='red', label='Real ' + str(symbol) + ' Price')
  plt.plot(y_pred, color='cyan', label='Predicted ' + str(symbol) + ' Price')
  plt.title(title)
  plt.legend()
  plt.show()


btn = Button(root, text="Run LSTM Prediction", command=mainfc).grid(row=1, column=0)
lb = Label(root, text="Made by Muhammad Osaid and Ethan Chew").grid(row=2, column=0)

root.mainloop()
