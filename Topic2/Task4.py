"""Task 4: Time Series Prediction with LSTM on Ebola Cases Data"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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

def make_sequences(dataset, look_back):
    """    
    Create sequences of data for LSTM input

    :param dataset: The dataset to create sequences from
    :param look_back: The number of previous time steps to use as input variables
    """
    var_x, var_y = [], []
    for i in range(len(dataset) - look_back):
        var_x.append(dataset[i:i+look_back, 0])
        var_y.append(dataset[i+look_back, 0])
    return np.array(var_x), np.array(var_y)

LOOK_BACK = 3

# LSTM for each country
# ------------------- Guinea ---------------------------

trainX_g, trainY_g = make_sequences(train_g, LOOK_BACK)
testX_g, testY_g = make_sequences(test_g, LOOK_BACK)

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

trainPred_g = model_g.predict(trainX_g)
testPred_g = model_g.predict(testX_g)

# reshape y for inverse scaling
trainY_g_2d = trainY_g.reshape(-1, 1)
testY_g_2d = testY_g.reshape(-1, 1)

trainPred_g_inv = scaler_g.inverse_transform(trainPred_g)
testPred_g_inv = scaler_g.inverse_transform(testPred_g)
trainY_g_inv = scaler_g.inverse_transform(trainY_g_2d)
testY_g_inv = scaler_g.inverse_transform(testY_g_2d)

# build arrays for plotting
data_g_orig = data_g
trainPlot_g = np.empty_like(data_g_orig)
trainPlot_g[:] = np.nan
trainPlot_g[LOOK_BACK:LOOK_BACK + len(trainPred_g_inv), 0] = trainPred_g_inv[:, 0]

testPlot_g = np.empty_like(data_g_orig)
testPlot_g[:] = np.nan
test_start_g = train_size_g + LOOK_BACK
testPlot_g[test_start_g:test_start_g + len(testPred_g_inv), 0] = testPred_g_inv[:, 0]

# ------------------- Liberia ---------------------------

trainX_l, trainY_l = make_sequences(train_l, LOOK_BACK)
testX_l, testY_l = make_sequences(test_l, LOOK_BACK)

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

trainPred_l = model_l.predict(trainX_l)
testPred_l = model_l.predict(testX_l)

# reshape y
trainY_l_2d = trainY_l.reshape(-1, 1)
testY_l_2d = testY_l.reshape(-1, 1)

# inverse transform
trainPred_l_inv = scaler_l.inverse_transform(trainPred_l)
testPred_l_inv = scaler_l.inverse_transform(testPred_l)
trainY_l_inv = scaler_l.inverse_transform(trainY_l_2d)
testY_l_inv = scaler_l.inverse_transform(testY_l_2d)

data_l_orig = data_l
trainPlot_l = np.empty_like(data_l_orig)
trainPlot_l[:] = np.nan
trainPlot_l[LOOK_BACK:LOOK_BACK + len(trainPred_l_inv), 0] = trainPred_l_inv[:, 0]

testPlot_l = np.empty_like(data_l_orig)
testPlot_l[:] = np.nan
test_start_l = train_size_l + LOOK_BACK
testPlot_l[test_start_l:test_start_l + len(testPred_l_inv), 0] = testPred_l_inv[:, 0]

# ------------------- Sierra Leone ---------------------------

trainX_sl, trainY_sl = make_sequences(train_sl, LOOK_BACK)
testX_sl, testY_sl = make_sequences(test_sl, LOOK_BACK)

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

trainPred_sl = model_sl.predict(trainX_sl)
testPred_sl = model_sl.predict(testX_sl)

# reshape y
trainY_sl_2d = trainY_sl.reshape(-1, 1)
testY_sl_2d = testY_sl.reshape(-1, 1)

# inverse transform
trainPred_sl_inv = scaler_sl.inverse_transform(trainPred_sl)
testPred_sl_inv = scaler_sl.inverse_transform(testPred_sl)
trainY_sl_inv = scaler_sl.inverse_transform(trainY_sl_2d)
testY_sl_inv = scaler_sl.inverse_transform(testY_sl_2d)

data_sl_orig = data_sl
trainPlot_sl = np.empty_like(data_sl_orig)
trainPlot_sl[:] = np.nan
trainPlot_sl[LOOK_BACK:LOOK_BACK + len(trainPred_sl_inv), 0] = trainPred_sl_inv[:, 0]

testPlot_sl = np.empty_like(data_sl_orig)
testPlot_sl[:] = np.nan
test_start_sl = train_size_sl + LOOK_BACK
testPlot_sl[test_start_sl:test_start_sl + len(testPred_sl_inv), 0] = testPred_sl_inv[:, 0]

# ---------------------------- Plotting ---------------------------

plt.figure(figsize=(12, 12))

# Guinea
plt.subplot(3, 1, 1)
plt.plot(data_g_orig[:, 0], label="Actual")
plt.plot(trainPlot_g[:, 0], label="Train prediction")
plt.plot(testPlot_g[:, 0], label="Test prediction")
plt.title("Guinea")
plt.xlabel("Time index")
plt.ylabel("NumOutbreaks")
plt.legend()

# Liberia
plt.subplot(3, 1, 2)
plt.plot(data_l_orig[:, 0], label="Actual")
plt.plot(trainPlot_l[:, 0], label="Train prediction")
plt.plot(testPlot_l[:, 0], label="Test prediction")
plt.title("Liberia")
plt.xlabel("Time index")
plt.ylabel("NumOutbreaks")
plt.legend()

# Sierra Leone
plt.subplot(3, 1, 3)
plt.plot(data_sl_orig[:, 0], label="Actual")
plt.plot(trainPlot_sl[:, 0], label="Train prediction")
plt.plot(testPlot_sl[:, 0], label="Test prediction")
plt.title("Sierra Leone")
plt.xlabel("Time index")
plt.ylabel("NumOutbreaks")
plt.legend()

plt.tight_layout()
plt.show()
