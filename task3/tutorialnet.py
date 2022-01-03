"""
Slight modification of network that was given by Professor Yipeng Hu
https://github.com/YipengHu/COMP0090/blob/main/tutorials/img_cls/network_pt.py
"""

import tensorflow as tf
from tensorflow import keras


def TutorialNet():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.nn.relu, padding='SAME',
                                  input_shape=(32, 32, 3)))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.nn.relu, padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.nn.relu, padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(120, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(84, activation=tf.nn.relu))
    model.add(keras.layers.Dense(10))
    return model

