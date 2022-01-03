'''
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
'''
import tensorflow as tf
from tensorflow import keras


# AlexNet Model
def AlexNet():
    # Sequential 모델 선언
    model = keras.Sequential()

    # 첫 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, padding='SAME', activation=tf.nn.relu,
                                  input_shape=(32, 32, 3)))
    # Max Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME'))

    # 두 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='SAME', activation=tf.nn.relu))
    # Max Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME'))

    # 세 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu))
    # 네 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu))
    # 다섯 번째 Convolutional Layer
    model.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=3, padding='SAME', activation=tf.nn.relu))
    # Max Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME'))

    # Connecting it to a Fully Connected layer
    model.add(keras.layers.Flatten())

    # 첫 번째 Fully Connected Layer
    model.add(keras.layers.Dense(4096, input_shape=(32 * 32 * 3,), activation=tf.nn.relu))
    # Add Dropout to prevent overfitting
    model.add(keras.layers.Dropout(0.4))

    # 두 번째 Fully Connected Layer
    model.add(keras.layers.Dense(4096, activation=tf.nn.relu))
    # Add Dropout
    model.add(keras.layers.Dropout(0.4))

    # 세 번째 Fully Connected Layer
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

    return model

