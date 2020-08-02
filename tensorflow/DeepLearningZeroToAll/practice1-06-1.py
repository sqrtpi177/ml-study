import tensorflow as tf
import numpy as np

x_raw = [[1, 2, 1, 1],
         [2, 1, 3, 2],
         [3, 1, 3, 4],
         [4, 1, 5, 5],
         [1, 7, 5, 5],
         [1, 2, 5, 6],
         [1, 6, 6, 6],
         [1, 7, 7, 7]]
y_raw = [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

x_data = np.array(x_raw, dtype=np.float32)
y_data = np.array(y_raw, dtype=np.float32)

nb_classes = 3

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(input_dim=4, units=nb_classes, use_bias=True))

model.add(tf.keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
model.summary()

history = model.fit(x_data, y_data, epochs=2000)

print("--------------")
a = model.predict(np.array([[1, 11, 7, 9]]))
print(a, tf.keras.backend.eval(tf.argmax(a, axis=1)))

print("--------------")
b = model.predict(np.array([[1, 3, 4, 3]]))
print(b, tf.keras.backend.eval(tf.argmax(b, axis=1)))

print("--------------")
c = model.predict([[1, 1, 0, 1]])
c_onehot = model.predict_classes(np.array([[1, 1, 0, 1]]))
print(c, c_onehot)

print("--------------")
all = model.predict(np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]))
all_onehot = model.predict_classes(np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]))
print(all, all_onehot)
