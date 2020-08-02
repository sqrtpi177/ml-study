import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


def min_max_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)


seq_length = 7
data_dim = 5
output_dim = 1
learning_rate = 0.01
iterations = 500

xy = np.loadtxt('data-02.csv', delimiter=',')
xy = xy[::-1]

train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]

train_set = min_max_scaler(train_set)
test_set = min_max_scaler(test_set)


def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        x = time_series[i:i + seq_length, :]
        y = time_series[i + seq_length, [-1]]
        print(x, "->", y)
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)


trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

print(trainX.shape)
print(trainY.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=1, input_shape=(seq_length, data_dim)))
model.add(tf.keras.layers.Dense(units=output_dim, activation='tanh'))
model.summary()

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=learning_rate))

with tf.device('/GPU:0'):
    start = time.time()
    model.fit(trainX, trainY, epochs=iterations)
    end = time.time()
    print("Time: ", end - start)

test_predict = model.predict(testX)

plt.plot(testY)
plt.plot(test_predict)
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.show()
