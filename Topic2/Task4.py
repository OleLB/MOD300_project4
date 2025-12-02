"""Task 4: Time Series Prediction with LSTM"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Source:
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

g = pd.read_csv("Topic2/ebola_cases_guinea.dat", delim_whitespace=True)
l = pd.read_csv("Topic2/ebola_cases_liberia.dat", delim_whitespace=True)
sl = pd.read_csv("Topic2/ebola_cases_sierra_leone.dat", delim_whitespace=True)

data_g = g["NumOutbreaks"].values.astype("float32").reshape(-1, 1)
data_l = l["NumOutbreaks"].values.astype("float32").reshape(-1, 1)
data_sl = sl["NumOutbreaks"].values.astype("float32").reshape(-1, 1)

scaler_g = MinMaxScaler(feature_range=(0, 1))
data_g_scaled = scaler_g.fit_transform(data_g)

scaler_l = MinMaxScaler(feature_range=(0, 1))
data_l_scaled = scaler_l.fit_transform(data_l)

scaler_sl = MinMaxScaler(feature_range=(0, 1))
data_sl_scaled = scaler_sl.fit_transform(data_sl)

train_size_g = int(len(data_g_scaled) * 0.67) # 67% for training
train_g = data_g_scaled[:train_size_g]
test_g = data_g_scaled[train_size_g:]

train_size_l = int(len(data_l_scaled) * 0.67)
train_l = data_l_scaled[:train_size_l]
test_l = data_l_scaled[train_size_l:]

train_size_sl = int(len(data_sl_scaled) * 0.67)
train_sl = data_sl_scaled[:train_size_sl]
test_sl = data_sl_scaled[train_size_sl:]

def create_dataset(dataset, look_back):
    """    
    Create sequences of data for LSTM input

    :param dataset: The dataset to create sequences from
    :param look_back: The number of previous time steps to use as input variables
    """
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back):
        data_x.append(dataset[i:i+look_back, 0])
        data_y.append(dataset[i+look_back, 0])
    return np.array(data_x), np.array(data_y)

LOOK_BACK = 3

# LSTM for each country
# ------------------- Guinea ---------------------------

trainX_g, trainY_g = create_dataset(train_g, LOOK_BACK)
testX_g, testY_g = create_dataset(test_g, LOOK_BACK)

trainX_g = trainX_g.reshape((trainX_g.shape[0], LOOK_BACK, 1))
testX_g = testX_g.reshape((testX_g.shape[0], LOOK_BACK, 1))

model_g = Sequential()
model_g.add(LSTM(50, input_shape=(LOOK_BACK, 1)))
model_g.add(Dense(1))

model_g.compile(loss="mean_squared_error", optimizer="adam")

history_g = model_g.fit(
    trainX_g, trainY_g,
    epochs=50,
    batch_size=1,
    verbose=1
)

trainPredict_g = model_g.predict(trainX_g)
testPredict_g = model_g.predict(testX_g)

# reshape y for inverse scaling
trainY_g_2d = trainY_g.reshape(-1, 1)
testY_g_2d = testY_g.reshape(-1, 1)

trainPredict_g_inv = scaler_g.inverse_transform(trainPredict_g)
testPredict_g_inv = scaler_g.inverse_transform(testPredict_g)
trainY_g_inv = scaler_g.inverse_transform(trainY_g_2d)
testY_g_inv = scaler_g.inverse_transform(testY_g_2d)

# build arrays for plotting
trainPlot_g = np.empty_like(data_g)
trainPlot_g[:] = np.nan
trainPlot_g[LOOK_BACK:LOOK_BACK + len(trainPredict_g_inv), 0] = trainPredict_g_inv[:, 0]

testPlot_g = np.empty_like(data_g)
testPlot_g[:] = np.nan
test_start_g = train_size_g + LOOK_BACK
testPlot_g[test_start_g:test_start_g + len(testPredict_g_inv), 0] = testPredict_g_inv[:, 0]

# ------------------- Liberia ---------------------------

trainX_l, trainY_l = create_dataset(train_l, LOOK_BACK)
testX_l, testY_l = create_dataset(test_l, LOOK_BACK)

trainX_l = trainX_l.reshape((trainX_l.shape[0], LOOK_BACK, 1))
testX_l = testX_l.reshape((testX_l.shape[0], LOOK_BACK, 1))

model_l = Sequential()
model_l.add(LSTM(50, input_shape=(LOOK_BACK, 1)))
model_l.add(Dense(1))

model_l.compile(loss="mean_squared_error", optimizer="adam")

history_l = model_l.fit(
    trainX_l, trainY_l,
    epochs=50,
    batch_size=1,
    verbose=1
)

trainPredict_l = model_l.predict(trainX_l)
testPredict_l = model_l.predict(testX_l)

# reshape y
trainY_l_2d = trainY_l.reshape(-1, 1)
testY_l_2d = testY_l.reshape(-1, 1)

# inverse transform
trainPredict_l_inv = scaler_l.inverse_transform(trainPredict_l)
testPredict_l_inv = scaler_l.inverse_transform(testPredict_l)
trainY_l_inv = scaler_l.inverse_transform(trainY_l_2d)
testY_l_inv = scaler_l.inverse_transform(testY_l_2d)

trainPlot_l = np.empty_like(data_l)
trainPlot_l[:] = np.nan
trainPlot_l[LOOK_BACK:LOOK_BACK + len(trainPredict_l_inv), 0] = trainPredict_l_inv[:, 0]

testPlot_l = np.empty_like(data_l)
testPlot_l[:] = np.nan
test_start_l = train_size_l + LOOK_BACK
testPlot_l[test_start_l:test_start_l + len(testPredict_l_inv), 0] = testPredict_l_inv[:, 0]

# ------------------- Sierra Leone ---------------------------

trainX_sl, trainY_sl = create_dataset(train_sl, LOOK_BACK)
testX_sl, testY_sl = create_dataset(test_sl, LOOK_BACK)

trainX_sl = trainX_sl.reshape((trainX_sl.shape[0], LOOK_BACK, 1))
testX_sl = testX_sl.reshape((testX_sl.shape[0], LOOK_BACK, 1))

model_sl = Sequential()
model_sl.add(LSTM(50, input_shape=(LOOK_BACK, 1)))
model_sl.add(Dense(1))

model_sl.compile(loss="mean_squared_error", optimizer="adam")

history_sl = model_sl.fit(
    trainX_sl, trainY_sl,
    epochs=50,
    batch_size=1,
    verbose=1
)

trainPredict_sl = model_sl.predict(trainX_sl)
testPredict_sl = model_sl.predict(testX_sl)

# reshape y
trainY_sl_2d = trainY_sl.reshape(-1, 1)
testY_sl_2d = testY_sl.reshape(-1, 1)

# inverse transform
trainPredict_sl_inv = scaler_sl.inverse_transform(trainPredict_sl)
testPredict_sl_inv = scaler_sl.inverse_transform(testPredict_sl)
trainY_sl_inv = scaler_sl.inverse_transform(trainY_sl_2d)
testY_sl_inv = scaler_sl.inverse_transform(testY_sl_2d)

trainPlot_sl = np.empty_like(data_sl)
trainPlot_sl[:] = np.nan
trainPlot_sl[LOOK_BACK:LOOK_BACK + len(trainPredict_sl_inv), 0] = trainPredict_sl_inv[:, 0]

testPlot_sl = np.empty_like(data_sl)
testPlot_sl[:] = np.nan
test_start_sl = train_size_sl + LOOK_BACK
testPlot_sl[test_start_sl:test_start_sl + len(testPredict_sl_inv), 0] = testPredict_sl_inv[:, 0]

# ---------------------------- Plotting ---------------------------

plt.figure(figsize=(12, 12))

# Guinea
plt.subplot(3, 1, 1)
plt.plot(data_g[:, 0], label="Actual")
plt.plot(trainPlot_g[:, 0], label="Train prediction")
plt.plot(testPlot_g[:, 0], label="Test prediction")
plt.title("Guinea")
plt.xlabel("Time index")
plt.ylabel("NumOutbreaks")
plt.legend()

# Liberia
plt.subplot(3, 1, 2)
plt.plot(data_l[:, 0], label="Actual")
plt.plot(trainPlot_l[:, 0], label="Train prediction")
plt.plot(testPlot_l[:, 0], label="Test prediction")
plt.title("Liberia")
plt.xlabel("Time index")
plt.ylabel("NumOutbreaks")
plt.legend()

# Sierra Leone
plt.subplot(3, 1, 3)
plt.plot(data_sl[:, 0], label="Actual")
plt.plot(trainPlot_sl[:, 0], label="Train prediction")
plt.plot(testPlot_sl[:, 0], label="Test prediction")
plt.title("Sierra Leone")
plt.xlabel("Time index")
plt.ylabel("NumOutbreaks")
plt.legend()

plt.tight_layout()
plt.show()
