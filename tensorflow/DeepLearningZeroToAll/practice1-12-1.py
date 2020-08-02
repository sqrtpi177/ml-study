import numpy as np
import tensorflow as tf
import time

with tf.device('/GPU:0'):
    start = time.time()

    idx2char = ['h', 'i', 'e', 'l', 'o']
    # x_data = [[0, 1, 0, 2, 3, 3]]
    y_data = [[1, 0, 2, 3, 3, 4]]

    num_classes = 5
    input_dim = 5
    sequence_length = 6
    learning_rate = 0.1

    x_one_hot = np.array([[[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 0]]],
                         dtype=np.float32)

    y_one_hot = tf.keras.utils.to_categorical(y_data, num_classes=num_classes)
    print(x_one_hot.shape)
    print(y_one_hot)

    model = tf.keras.Sequential()

    cell = tf.keras.layers.LSTMCell(units=num_classes, input_shape=(sequence_length, input_dim))
    model.add(tf.keras.layers.RNN(cell=cell, return_sequences=True))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=num_classes, activation='softmax')))
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    model.fit(x_one_hot, y_one_hot, epochs=50)

    predictions = model.predict(x_one_hot)
    for i, prediction in enumerate(predictions):
        print(prediction)
        result_str = [idx2char[c] for c in np.argmax(prediction, axis=1)]
        print("\tPrediction str: ", ''.join(result_str))

    end = time.time()
    print("Time: ", end - start)
