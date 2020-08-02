import datetime
import tensorflow as tf
import numpy as np
import time

start = time.time()

with tf.device('/GPU:0'):
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=2, input_dim=2, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
    model.summary()

    log_dir = ".\\logs\\test\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x_data, y_data, epochs=5000, callbacks=[tensorboard_callback])

    predictions = model.predict(x_data)
    print("Prediction: \n", predictions)

    score = model.evaluate(x_data, y_data)
    print("Accuracy: \n", score[1])

finish = time.time()
print("Time:", finish - start)
