import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("허브 버전: ", hub.__version__)
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")

train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
(train_data, validation_data), test_data = tfds.load(name="imdb_reviews",
                                                     split=(train_validation_split, tfds.Split.TEST),
                                                     as_supervised=True)
