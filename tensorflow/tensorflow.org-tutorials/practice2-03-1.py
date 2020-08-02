import tensorflow as tf
import numpy as np
import os

"""
path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
"""

text = open('text/Charlie and the Chocolate Factory.txt', 'rb').read().decode(encoding='utf-8')
print(text)

print('텍스트의 길이: {}자'.format(len(text)))
print(text[:250])

vocab = sorted(set(text))
print('고유 문자수 {}개'.format(len(vocab)))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

print('{')
for char, _ in zip(char2idx, range(20)):
    print('    {:4s}: {:3d}'.format(repr(char), char2idx[char]))
print('    ...\n}')

print('{} ---- 문자들이 다음의 정수로 매핑되었습니다 ----> {}'.format(repr(text[:13]), text_as_int[:13]))

seq_length = 100
examples_per_epoch = len(text)//seq_length

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print('입력 데이터: ', repr(''.join(idx2char[input_example.numpy()])))
    print('타깃 데이터: ', repr(''.join(idx2char[target_example.numpy()])))
    for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
        print('{:4d}단계:'.format(i))
        print("  입력: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
        print("  예상 출력: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE
)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, '# (배치 크기, 시퀀스 길이, 어휘 사전 크기)')

    sample_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sample_indices = tf.squeeze(sample_indices, axis=-1).numpy()

    print(sample_indices)

    print('입력: \n', repr(''.join(idx2char[input_example_batch[0]])))
    print()
    print('예측된 다음 문자: \n', repr(''.join(idx2char[sample_indices])))

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    print('예측 배열 크기(shape): ', example_batch_predictions.shape, '# (배치 크기, 시퀀스 길이, 어휘 사전 크기)')
    print('스칼라 손실:         ', example_batch_loss.numpy().mean())

model.summary()

model.compile(optimizer='Adam', loss=loss)

checkpoint_dir = '.\\training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt1_{epoch}')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

EPOCHS_1 = 100
history = model.fit(dataset, epochs=EPOCHS_1, callbacks=[checkpoint_callback])

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

"""
EPOCHS_2 = 100
history = model.fit(dataset, epochs=EPOCHS_2, callbacks=[checkpoint_callback])
"""

# load model
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


def generate_text(model, start_string):
    num_generate = 1000

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"The "))
