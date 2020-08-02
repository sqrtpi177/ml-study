import numpy as np
import random
import datetime
import tensorflow as tf

tf.debugging.set_log_device_placement(True)

with tf.device('/GPU:0'):
    random.seed(777)
    learning_rate = 0.001
    batch_size = 100
    training_epochs = 15
    nb_classes = 10
    drop_rate = 0.3

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)

    y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_dim=784, units=512, kernel_initializer='glorot_normal', activation='relu'))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(units=512, kernel_initializer='glorot_normal', activation='relu'))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(units=512, kernel_initializer='glorot_normal', activation='relu'))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(units=512, kernel_initializer='glorot_normal', activation='relu'))
    model.add(tf.keras.layers.Dropout(drop_rate))
    model.add(tf.keras.layers.Dense(units=nb_classes, kernel_initializer='glorot_normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    model.summary()

    log_dir = ".\\logs\\test\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x_train, y_train, epochs=training_epochs, callbacks=[tensorboard_callback])

    y_predicted = model.predict(x_test)
    for x in range(0, 10):
        random_index = random.randint(0, x_test.shape[0] - 1)
        print("Index: ", random_index,
              "actual y: ", np.argmax(y_test[random_index]),
              "predicted_y: ", np.argmax(y_predicted[random_index]))

    evaluation = model.evaluate(x_test, y_test)
    print("Loss: ", evaluation[0])
    print("Accuracy: ", evaluation[1])
