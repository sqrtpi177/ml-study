import numpy as np
import tensorflow as tf

xy = np.loadtxt('data-04.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7

y_one_hot = tf.keras.utils.to_categorical(y_data, nb_classes)
print("one_hot:", y_one_hot)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=16, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
model.summary()

history = model.fit(x_data, y_one_hot, epochs=1000)

test_data = np.array([[0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]])
print(model.predict(test_data), model.predict_classes(test_data))

pred = model.predict_classes(x_data)
for p, y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
