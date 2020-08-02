import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-01.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=3))
model.add(tf.keras.layers.Activation('linear'))

model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))
model.summary()
history = model.fit(x_data, y_data, epochs=100)

y_predict = model.predict(np.array([[72., 93., 90.]]))
print(y_predict)
