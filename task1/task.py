import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError
from random import choice
from statistics import mean, stdev
from time import time


def get_rmse(y, y_hat):
	metric = RootMeanSquaredError()
	metric.update_state(y, y_hat)
	return float(metric.result())


# function to transform X
def polynomial_transform(X, degree):
	m, _ = X.shape

	stack_list = []
	for i in range(degree+1):
		stack_list.append(tf.pow(X, i+1))
	x_transform = tf.stack(stack_list, axis=1)
	x_transform = tf.reshape(x_transform, [m, degree+1])

	return x_transform


# predicts y_hat
def polynomial_fun(X, weight):
	return tf.matmul(X, weight)


# model fitting with normal equation ( ||t-y||^2 is minimised )
def fit_polynomial_ls(X, Y, degree):
	X_transform = polynomial_transform(X, degree)

	start = time()

	X_T = tf.transpose(X_transform)
	temp = tf.linalg.inv(tf.matmul(X_T, X_transform))
	weights = tf.matmul(tf.matmul(temp, X_T), Y)

	end = time()

	return weights, end-start


# model training with stochastic gradient descent
def fit_polynomial_sgd(X, Y, degree, learning_rate=1e-20, iterations=1000, batch_size=10):
	X_transform = polynomial_transform(X, degree)

	model = tf.keras.Sequential()
	model.add(tf.keras.Input(shape=(degree+1)))
	model.add(tf.keras.layers.Dense(degree+1))
	model.add(tf.keras.layers.Dense(1))

	sgd = tf.keras.optimizers.SGD(lr=learning_rate)
	model.compile(optimizer=sgd,
	              loss=tf.keras.losses.MeanSquaredError())

	start = time()
	model.fit(X_transform, Y, epochs=iterations, batch_size=batch_size)
	end = time()

	return model.get_weights()[-2], end-start


if __name__ == '__main__':
	#####################
	# Dataset generation
	#####################
	weight = tf.constant([[1], [2], [3], [4]], dtype=tf.dtypes.float32)

	x_train = tf.transpose(tf.constant([[choice(range(-20, 21)) for _ in range(100)]], dtype=tf.dtypes.float32))
	x_test = tf.transpose(tf.constant([[choice(range(-20, 21)) for _ in range(50)]], dtype=tf.dtypes.float32))
	# print(x_train.shape, x_test.shape)  # (100, 1) (50, 1)

	# polynomial transform of x_train and x_test
	x_train_transform_for_data = polynomial_transform(x_train, degree=3)
	x_test_transform_for_data = polynomial_transform(x_test, degree=3)
	# print(x_train_transform_for_data.shape, x_test_transform_for_data.shape)  # (100, 4) (50, 4)

	# generate y_train with noise
	y_train = polynomial_fun(x_train_transform_for_data, weight)
	noise = tf.random.normal(shape=tf.shape(y_train), stddev=0.2, dtype=tf.dtypes.float32)
	y_train = y_train + noise

	# generate y_test with noise
	y_test = polynomial_fun(x_test_transform_for_data, weight)
	noise = tf.random.normal(shape=tf.shape(y_test), stddev=0.2, dtype=tf.dtypes.float32)
	y_test = y_test + noise

	# print(y_train.shape, y_test.shape)  # (100, 1) (50, 1)

	##########
	# LS
	#########
	print("Least Squares")

	# transformed train and test to 4 degree
	x_train_transform = polynomial_transform(x_train, degree=4)
	x_test_transform = polynomial_transform(x_test, degree=4)

	# Least Squares polynomial fitting
	weight_hat_train, elapsed_time_ls = fit_polynomial_ls(x_train, y_train, degree=4)
	# print(weight_hat_train.shape)  # (5, 1)
	print(f"Elapsed time for fitting Polynomial Least Squared model: {elapsed_time_ls} seconds")

	# get y_hat for both training and test set
	y_hat_train = polynomial_fun(x_train_transform, weight_hat_train)
	y_hat_test = polynomial_fun(x_test_transform, weight_hat_train)
	# print(y_hat_train.shape, y_hat_test.shape)  # (100, 1) (50, 1)


	diff = []
	for p, t in zip(y_hat_train, y_train):
		pred, true = float(tf.reshape(p, shape=[1, ])), float(tf.reshape(t, shape=[1, ]))
		diff.append(abs(pred - true))

	print(f"a) Train set mean difference : {mean(diff)}")
	print(f"a) Train set std of differences  : {stdev(diff)}")

	diff = []
	for p, t in zip(y_hat_test, y_test):
		pred, true = float(tf.reshape(p, shape=[1, ])), float(tf.reshape(t, shape=[1, ]))
		diff.append(abs(pred - true))

	print(f"b) Test set mean difference : {mean(diff)}")
	print(f"b) Test set std of differences  : {stdev(diff)}")

	print()

	####
	# SGD
	####
	print("SGD")

	weight_hat_train, elapsed_time_sgd = fit_polynomial_sgd(x_train, y_train, degree=4)
	print(f"Elapsed time for fitting Polynomial SGD: {elapsed_time_sgd} seconds")
	weight_hat_train = tf.constant(weight_hat_train)  # shape=(5, 1)

	# get y_hat for both training and test set
	y_hat_train = polynomial_fun(x_train_transform, weight_hat_train)
	y_hat_test = polynomial_fun(x_test_transform, weight_hat_train)

	diff = []
	for p, t in zip(y_hat_train, y_train):
		pred, true = float(tf.reshape(p, shape=[1, ])), float(tf.reshape(t, shape=[1, ]))
		diff.append(abs(pred - true))

	print(f"a) Train set mean difference : {mean(diff)}")
	print(f"a) Train set std of differences  : {stdev(diff)}")

	diff = []
	for p, t in zip(y_hat_test, y_test):
		pred, true = float(tf.reshape(p, shape=[1, ])), float(tf.reshape(t, shape=[1, ]))
		diff.append(abs(pred - true))

	print(f"b) Test set mean difference : {mean(diff)}")
	print(f"b) Test set std of differences  : {stdev(diff)}")

