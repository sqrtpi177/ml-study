import numpy as np
import tensorflow as tf

x_train = [1, 2, 3]
y_train = [1, 2, 3]

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(units=1, input_dim=1))

sgd = tf.keras.optimizers.SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=200)

y_predict = model.predict(np.array([5]))
print(y_predict)
