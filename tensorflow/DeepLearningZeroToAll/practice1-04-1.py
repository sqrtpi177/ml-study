import tensorflow as tf
import numpy as np

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=3))
model.add(tf.keras.layers.Activation('linear'))

model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))
model.summary()
history = model.fit(x_data, y_data, epochs=100)

y_predict = model.predict(np.array([[72., 93., 90.]]))
print(y_predict)
