import tensorflow as tf
from tensorflow import keras


# LeNet Model
def LeNet():
    model = keras.Sequential()
    # Conv 1 Layer
    model.add(keras.layers.Conv2D(filters=6, kernel_size=5, strides=1, activation=tf.nn.relu, input_shape=(32, 32, 3)))

    # Sub Sampling Layer (Max Pooling)
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Conv 1 Layer
    model.add(keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, activation=tf.nn.relu, input_shape=(16, 16, 3)))

    # Sub Sampling Layer (Max Pooling)
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Fully Connected (FC) Layer와 연결하기 위한 Flatten
    model.add(keras.layers.Flatten())

    # FC1 Layer
    model.add(keras.layers.Dense(120, activation=tf.nn.relu))
    # FC2 Layer
    model.add(keras.layers.Dense(84, activation=tf.nn.relu))

    # Output Softmax
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

    return model

