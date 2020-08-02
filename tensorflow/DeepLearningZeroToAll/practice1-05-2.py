import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-03.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=x_data.shape[1], activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
model.summary()

history = model.fit(x_data, y_data, epochs=500)

print("Accuracy: {0}".format(history.history['accuracy'][-1]))

y_predict = model.predict([[0.176471, 0.155779, 0, 0, 0, 0.052161, -0.952178, -0.733333]])
print("Prediction: {0}".format(y_predict))

evaluate = model.evaluate(x_data, y_data)
print("loss: {0}, accuracy:{1}".format(evaluate[0], evaluate[1]))
