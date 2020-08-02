import tensorflow as tf
import numpy as np

sample = " if you want you"
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}

dic_size = len(char2idx)
hidden_size = len(char2idx)
num_classes = len(char2idx)
batch_size = 1
sequence_length = len(sample) - 1
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

x_one_hot_eager = tf.one_hot(x_data, num_classes)
y_one_hot_eager = tf.one_hot(y_data, num_classes)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=num_classes, input_shape=(sequence_length, x_one_hot_eager.shape[2]),
                               return_sequences=True))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=num_classes, activation='softmax')))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
model.fit(x_one_hot_eager, y_one_hot_eager, epochs=50)

predictions = model.predict(x_one_hot_eager)

for i, prediction in enumerate(x_one_hot_eager):
    result_str = [idx2char[c] for c in np.argmax(prediction, axis=1)]
    print("\tPrediction str: ", ''.join(result_str))
