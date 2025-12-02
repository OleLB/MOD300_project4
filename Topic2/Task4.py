"""Task 4: Time Series Prediction with LSTM on Ebola Cases Data"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Source:
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

g = pd.read_csv("Topic2/ebola_cases_guinea.dat", delim_whitespace=True)
l = pd.read_csv("Topic2/ebola_cases_liberia.dat", delim_whitespace=True)
sl = pd.read_csv("Topic2/ebola_cases_sierra_leone.dat", delim_whitespace=True)

data_g = g["NumOutbreaks"].values.astype("float32").reshape(-1, 1)
data_l = l["NumOutbreaks"].values.astype("float32").reshape(-1, 1)
data_sl = sl["NumOutbreaks"].values.astype("float32").reshape(-1, 1)

scaler_g = MinMaxScaler(feature_range=(0, 1))
data_g_scaled = scaler_g.fit_transform(data_g)

train_size = int(len(data_g_scaled) * 0.7) # 70% for training
train_g = data_g_scaled[:train_size]
test_g = data_g_scaled[train_size:]

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

trainX_g, trainY_g = make_sequences(train_g, LOOK_BACK)
testX_g, testY_g = make_sequences(test_g, LOOK_BACK)

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

# Plotting
data_g_orig = data_g

trainPlot_g = np.empty_like(data_g_orig)
trainPlot_g[:] = np.nan
trainPlot_g[LOOK_BACK:LOOK_BACK+len(trainPred_g_inv), 0] = trainPred_g_inv[:, 0]
testPlot_g = np.empty_like(data_g_orig)
testPlot_g[:] = np.nan
test_start = train_size + LOOK_BACK
testPlot_g[test_start:test_start+len(testPred_g_inv), 0] = testPred_g_inv[:, 0]
plt.figure(figsize=(10, 5))
plt.plot(data_g_orig[:, 0], label="Actual")
plt.plot(trainPlot_g[:, 0], label="Train prediction")
plt.plot(testPlot_g[:, 0], label="Test prediction")
plt.title("Ebola cases – Guinea (LSTM)")
plt.xlabel("Time index")
plt.ylabel("NumOutbreaks")
plt.legend()
plt.show()
