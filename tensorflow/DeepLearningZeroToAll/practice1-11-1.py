import numpy as np
import tensorflow as tf
import random
import time

with tf.device('/GPU:0'):
    start = time.time()

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    print(x_train[0], y_train[0])

    learning_rate = 0.001
    training_epochs = 12
    batch_size = 256

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=10, kernel_initializer='glorot_normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

    y_predicted = model.predict(x_test)
    for x in range(0, 10):
        random_index = random.randint(0, x_test.shape[0]-1)
        print("Index: ", random_index,
              "Actual y: ", np.argmax(y_test[random_index]),
              "Predicted y: ", np.argmax(y_predicted[random_index]))

    evaluation = model.evaluate(x_test, y_test)
    print("Loss: ", evaluation[0])
    print("Accuracy: ", evaluation[1])

    end = time.time()
    print("Time: ", end - start)
