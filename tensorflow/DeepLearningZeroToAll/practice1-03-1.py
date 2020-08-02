import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_train = [1, 2, 3]
y_train = [1, 2, 3]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=1))

sgd = tf.keras.optimizers.SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd)

model.summary()

history = model.fit(x_train, y_train, epochs=100)

y_predict = model.predict(np.array([5, 4]))
print(y_predict)

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
