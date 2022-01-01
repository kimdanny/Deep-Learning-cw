import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from PIL import Image
import os
# from task2.densenet3 import DenseNet3
# from task2.cutout import Cutout
from cutout import Cutout
from densenet3 import DenseNet3
from utils import get_concat_h_multi_resize, get_concat_v_multi_resize

# Global parameters
EPOCHS = 3
# set model saving path
curr_dir = os.getcwd()
model_name = 'cifar10-densenet3.h5'
MODEL_FILE_PATH = os.path.join(curr_dir, model_name)
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

# def plot_metrics(self, history, metric_name='loss', title='Loss history', filename='loss_plot.png'):
# 	if not os.path.exists(self.plot_dir):
# 		os.makedirs(self.plot_dir)
#
# 	train_metric = history.history['loss']
# 	plt.plot(train_metric, color='blue', label=metric_name)
#
# 	if self.is_validation:
# 		val_metric = history.history['val_loss']
# 		plt.plot(val_metric, color='green', label='val_' + metric_name)
#
# 	plt.title(title)
# 	plt.legend()
# 	plt.savefig(f'{self.plot_dir}/{filename}')

# TODO: need to print: Report the test set performance in terms of classification accuracy versus the epochs.
print(history.history['loss'])
print(history.history['val_loss'])
print(history.history['accuracy'])

test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_loss, test_acc)

# TODO: Visualise your results, by saving to a PNG file “result.png”, a montage of 36 test images
#  with captions indicating the ground-truth and the predicted classes for each.
