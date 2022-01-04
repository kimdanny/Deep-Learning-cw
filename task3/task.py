import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tutorialnet import TutorialNet
from lenet import LeNet
from copy import deepcopy
import os
from time import time
from cutout import Cutout
from densenet3 import DenseNet3

print("CUTOUT data augmentation Ablation Study using Cross-Validation (CV)")

# Global parameters
EPOCHS = 10
K_FOLD = 3
# set model saving path
curr_dir = os.getcwd()
MODEL_NAME = 'cifar10-densenet3-no-aug.h5'
MODEL_NAME_AUG = 'cifar10-densenet3-aug.h5'

MODEL_FILE_PATH = os.path.join(curr_dir, MODEL_NAME)
MODEL_FILE_PATH_AUG = os.path.join(curr_dir, MODEL_NAME_AUG)


TUTORIALNET_NAME = 'cifar10-tutorialnet.h5'
LENET_NAME = 'cifar10-lenet.h5'

TUTORIALNET_FILE_PATH = os.path.join(curr_dir, TUTORIALNET_NAME)
LENET_FILE_PATH = os.path.join(curr_dir, LENET_NAME)

cutout_class = Cutout()

# cifar-10 dataset loading
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# data augmentation
def augment_images(train_images: np.ndarray) -> np.ndarray:
	"""
	Apply Cutout method to all images

	:param train_images: training images in np.ndarray
	:return: Augmented images in np.ndarray
	"""
	cutout_train_images = np.array([cutout_class.cutout(im) for im in train_images])
	return cutout_train_images


# x_train_aug: cutout images for training
x_train_aug = augment_images(x_train)

# change datatype to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize images to [0,1]
x_train = x_train / 255.0
x_train_aug = x_train_aug / 255.0
x_test = x_test / 255.0

# categorical encoding of labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Randomly pre-shuffles data to give randomness to Cross Validation data split
shuffler = np.random.permutation(len(x_train))
x_train = x_train[shuffler]
x_train_aug = x_train_aug[shuffler]
y_train, y_train_aug = y_train[shuffler], y_train[shuffler]


# callback for model checkpointing
checkpoint_no_aug = ModelCheckpoint(filepath=MODEL_FILE_PATH, verbose=1, save_best_only=True)
callback_no_aug = [checkpoint_no_aug]

# callback for aug model checkpointing
checkpoint_aug = ModelCheckpoint(filepath=MODEL_FILE_PATH_AUG, verbose=1, save_best_only=True)
callback_aug = [checkpoint_aug]

####################
# Cross Validation #
####################

histories_no_aug = []
histories_aug = []

time_histories_no_aug = []
time_histories_aug = []

offset = 0
for fold in range(K_FOLD):
	vali_size = len(x_train) // K_FOLD

	# get validation set (no-aug)
	vali_cv = x_train[offset: offset + vali_size]
	# get validation set (aug)
	vali_aug_cv = x_train_aug[offset: offset + vali_size]

	# get train set (no-aug)
	x_train_copy = list(deepcopy(x_train))
	del x_train_copy[offset: offset + vali_size]
	train_cv = np.array(x_train_copy)

	# get train set (aug)
	x_train_aug_copy = list(deepcopy(x_train_aug))
	del x_train_aug_copy[offset: offset + vali_size]
	train_aug_cv = np.array(x_train_aug_copy)

	# get labels of vali_cv (no-aug)
	y_vali_cv = y_train[offset: offset + vali_size]
	# get labels of vali_aug_cv (aug)
	y_vali_aug_cv = y_train_aug[offset: offset + vali_size]

	# get labels of train_cv (no-aug)
	y_train_copy = list(deepcopy(y_train))
	del y_train_copy[offset: offset + vali_size]
	y_train_cv = np.array(y_train_copy)

	# get labels of train_aug_cv (aug)
	y_train_aug_copy = list(deepcopy(y_train_aug))
	del y_train_aug_copy[offset: offset + vali_size]
	y_train_aug_cv = np.array(y_train_aug_copy)

	del x_train_copy, x_train_aug_copy, y_train_copy

	# print summary of CV dataset
	print(f"FOLD: {fold + 1} / {K_FOLD}")
	print(f"CV => train set shape: {train_cv.shape}")
	print(f"CV => validation set shape: {vali_cv.shape}")

	print(f"CV => train (aug) set shape: {train_aug_cv.shape}")
	print(f"CV => validation (aug) set shape: {vali_aug_cv.shape}")

	print(f"CV => labels for train shape: {y_train_cv.shape}")
	print(f"CV => labels for validation shape: {y_vali_cv.shape}")

	print(f"CV => labels for train (aug) shape: {y_train_aug_cv.shape}")
	print(f"CV => labels for validation (aug) shape: {y_vali_aug_cv.shape}")
	print()

	# DenseNet model loading
	dense_net_class = DenseNet3()
	# model_no_aug: model that is to be trained without augmentation
	model_no_aug = dense_net_class.dense_net()
	# model_aug: model that is to be trained WITH augmentation
	model_aug = dense_net_class.dense_net()

	for model in [model_aug, model_no_aug]:
		# compile with cross entropy loss and adam optimiser, and
		# added mse and accuracy as metrics for monitoring during training
		model.compile(optimizer='adam',
		              loss=CategoricalCrossentropy(from_logits=True),
		              metrics=['accuracy', 'mse'])

	# train two models
	start = time()
	history_no_aug = model_no_aug.fit(train_cv, y_train_cv, epochs=EPOCHS,
	                                  validation_data=(vali_cv, y_vali_cv), callbacks=callback_no_aug)
	end = time()
	elapsed_sec_no_aug = end - start
	print(f"Fold {fold+1} Non-Augmented model CV time: {elapsed_sec_no_aug // 60} min {elapsed_sec_no_aug % 60} sec")

	start = time()
	history_aug = model_aug.fit(train_aug_cv, y_train_aug_cv, epochs=EPOCHS,
	                                  validation_data=(vali_aug_cv, y_vali_aug_cv), callbacks=callback_aug)
	end = time()
	elapsed_sec_aug = end - start
	print(f"Fold {fold+1} Augmented model CV time: {elapsed_sec_aug // 60} min {elapsed_sec_aug % 60} sec")

	# save each fold's histories
	histories_no_aug.append(history_no_aug)
	histories_aug.append(history_aug)

	time_histories_no_aug.append(elapsed_sec_no_aug)
	time_histories_aug.append(elapsed_sec_aug)

	# increase index offset by validation size for the next fold of CV
	offset += vali_size


def report_cv_summary(history_list, time_list, is_aug):
	if is_aug:
		print("Cross Validation report of models trained with Augmented Dataset")
	else:
		print("Cross Validation report of models trained with Non-Augmented Dataset")

	for fold, his in enumerate(history_list):
		print(f"FOLD {fold + 1}")

		print("TRAIN LOSS")
		train_losses = his.history['loss']
		for i, train_loss in enumerate(train_losses):
			print(f"Epoch {i + 1}: {round(train_loss, 3)}")

		print("TRAIN ACCURACY")
		train_accuracies = his.history['accuracy']
		for i, train_accuracy in enumerate(train_accuracies):
			print(f"Epoch {i + 1}: {round(train_accuracy, 3)}")

		print("TRAIN MSE")
		train_mses = his.history['mse']
		for i, train_mse in enumerate(train_mses):
			print(f"Epoch {i + 1}: {round(train_mse, 3)}")

		validation_loss = his.history['val_loss']
		validation_accuracy = his.history['val_accuracy']
		validation_mse = his.history['val_mse']

		print(f"VALIDATION LOSS: {validation_loss[-1]}")
		print(f"VALIDATION ACCURACY: {validation_accuracy[-1]}")
		print(f"VALIDATION MSE: {validation_mse[-1]}")

		print(len(time_list))
		fold_time = time_list[fold]
		print(f"Cross Validating speed: {fold_time // 60} min {fold_time % 60} sec")

		print()


# Report the cv summary
report_cv_summary(histories_no_aug, time_histories_no_aug, is_aug=False)
report_cv_summary(histories_aug, time_histories_aug, is_aug=True)


##########################################
# Retrain two models with entire dev set #
##########################################
print('Retrain two ablation models with entire dev set')

# Train both (ablation) models
# DenseNet model loading
dense_net_class = DenseNet3()
# model_no_aug: model that is to be trained without augmentation
model_no_aug = dense_net_class.dense_net()
# model_aug: model that is to be trained WITH augmentation
model_aug = dense_net_class.dense_net()

for model in [model_no_aug, model_aug]:
	# compile with cross entropy loss and adam optimiser, and
	# added mse and accuracy as metrics for monitoring during training
	model.compile(optimizer='adam',
	              loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy', 'mse'])

# train two models
history_no_aug = model_no_aug.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=callback_no_aug)
history_aug = model_aug.fit(x_train_aug, y_train_aug, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=callback_aug)

# test set performance in terms of classification accuracy versus the epochs.
print("Non-Augmented Model Holdout set performance")
for i, (acc, mse) in enumerate(zip(history_no_aug.history['accuracy'], history_no_aug.history['mse'])):
	print(f"Epoch {i+1} => test accuracy: {acc} || test mse: {mse}")

print("Augmented Model Holdout set performance")
for i, (acc, mse) in enumerate(zip(history_aug.history['accuracy'], history_aug.history['mse'])):
	print(f"Epoch {i+1} => test accuracy: {acc} || test mse: {mse}")

# test performance
test_loss, test_acc, test_mse = model_no_aug.evaluate(x_test, y_test)
print(f"Non-Augmented model => Final test accuracy: {test_acc} || test mse: {test_mse} || test loss: {test_loss}")
test_loss, test_acc, test_mse = model_aug.evaluate(x_test, y_test)
print(f"Augmented model => Final test accuracy: {test_acc} || test mse: {test_mse} || test loss: {test_loss}")


##################################################
# Retrain two further models with entire dev set #
##################################################

# cifar-10 dataset loading
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# preprocessing
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


############
# LeNet    #
############
print("LeNet Model training")
lenet = LeNet()
lenet.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
lenet_checkpoint = ModelCheckpoint(filepath=LENET_FILE_PATH, verbose=1, save_best_only=True)
lenet_callback = [lenet_checkpoint]

lenet.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=lenet_callback)

test_loss, test_acc = lenet.evaluate(x_test, y_test)
print(f"LeNet model => Final test accuracy: {test_acc} || test loss: {test_loss}")


###############
# TutorialNet #
###############
print("TutorialNet Model training")
tutorial_net = TutorialNet()
tutorial_net.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
tutorialnet_checkpoint = ModelCheckpoint(filepath=TUTORIALNET_FILE_PATH, verbose=1, save_best_only=True)
tutorialnet_callback = [tutorialnet_checkpoint]

tutorial_net.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=tutorialnet_callback)

test_loss, test_acc = tutorial_net.evaluate(x_test, y_test)
print(f"TutorialNet model => Final test accuracy: {test_acc} || test loss: {test_loss}")


