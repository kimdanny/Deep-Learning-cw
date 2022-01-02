import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import load_model
from PIL import Image
import os
from cutout import Cutout
from densenet3 import DenseNet3
from utils import get_concat_h_multi_resize, get_concat_v_multi_resize

# Global parameters
EPOCHS = 10
# set model saving path
curr_dir = os.getcwd()
MODEL_NAME = 'cifar10-densenet3.h5'
MODEL_FILE_PATH = os.path.join(curr_dir, MODEL_NAME)
cutout_class = Cutout()

# cifar-10 dataset loading
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# normalize images to [0,1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# categorical encoding of labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# DenseNet model loading
dense_net_class = DenseNet3()
model = dense_net_class.dense_net()

# compile with loss and optimiser
model.compile(optimizer='adam',
              loss=CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# callback for model checkpointing
checkpoint = ModelCheckpoint(filepath=MODEL_FILE_PATH,
                             verbose=1, save_best_only=True)
callback = [checkpoint]


# data augmentation
# TODO: optimise with fixed length array filling
def augment_images(train_images: np.ndarray) -> np.ndarray:
    cutout_train_images = np.array([cutout_class.cutout(im) for im in train_images])
    return cutout_train_images


# # visualise the cutout implementation
# sample_16 = x_train[:16]
# cutout_sample_16 = np.array([cutout_class.cutout(im).astype('uint8') for im in sample_16])
# # 16 original pictures concatenated horizontally
# original_concated = get_concat_h_multi_resize(sample_16)
# # 16 cutout pictures concatenated horizontally
# cutout_concated = get_concat_h_multi_resize(cutout_sample_16)
# # both images above concatenated vertically
# get_concat_v_multi_resize([original_concated, cutout_concated]).save('cutout.png')

# apply cutout into x_train
x_train = augment_images(x_train)

history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=callback)


# test set performance in terms of classification accuracy versus the epochs.
for i, acc in enumerate(history.history['accuracy']):
    print(f"Epoch {i+1} => test accuracy: {acc}")

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Final test accuracy after training: {test_acc}")

# TODO: Visualise your results, by saving to a PNG file “result.png”, a montage of 36 test images
#  with captions indicating the ground-truth and the predicted classes for each.
#  CAN VISUALIZE WITH SAVED MODEL AND MAKE PNG FILE AFTER...


# load the saved model
model = load_model(MODEL_NAME)

# model prediction
n_sample = 36
x_test_samples = x_test[:n_sample]
y_test_samples = y_test[:n_sample]

y_class_names = []
y_hat_class_names = []

for x_sample, y_sample in zip(x_test_samples, y_test_samples):
    image = x_sample.reshape(1, 32, 32, 3)
    # make prediction (y_hat)
    y_hat = model.predict(image)
    y_hat = int(np.argmax(y_hat[0]))

    y_hat_class_name = class_names[y_hat]
    y_class_name = class_names[y_sample[0]]

    y_class_names.append(y_class_name)
    y_hat_class_names.append(y_hat_class_name)

print(f'y_class_names: {y_class_names}\n')
print(f'y_hat_class_names: {y_hat_class_names}')

