import tensorflow as tf


def get_mse(y, y_hat):
	errors = y - y_hat
	squared_errors = tf.math.square(errors)
	mse = sum(squared_errors) / len(squared_errors)
	return mse


# function to transform X
def transform(X, degree):
	m, n = X.shape
	# initialize X_transform
	X_transform = tf.ones((m, 1))

	for j in range(degree + 1):
		if j != 0:
			x_pow = tf.pow(X, j)
			# append x_pow to X_transform 2-D array
			X_transform = np.append(X_transform, x_pow.reshape(-1, 1), axis=1)

	return X_transform


# model fitting with normal equation ( ||t-y||^2 is minimised )
def fit_polynomial_ls(X, Y, degree):
	X_transform = transform(X, degree)
	weights = tf.linalg.inv(X_transform.T.dot(X_transform)).dot(X_transform.T).dot(Y)

	return weights


# model training with stochastic gradient descent
def fit_polynomial_sgd(X, Y, degree, learning_rate=0.01, iterations=10000, batch_size=None):
	m, n = X.shape

	X_transform = transform(X, degree)

	weights = tf.zeros(degree + 1)
	for i in range(iterations):
		y_hat = polynomial_fun(X, weights, degree)
		error = y_hat - Y
		# update weights by derivative of MSE -> ||t-y||^2 is minimised
		weights = weights - learning_rate * (2 / m) * tf.matmul(X_transform.T, error)

	return weights


def sgd_regressor(X, y, learning_rate=0.2, n_epochs=1000, k=40):
	w = tf.zeros(degree + 1)  # Randomly initializing weights
	b = np.random.randn(1, 1)  # Random intercept value

	epoch = 1

	while epoch <= n_epochs:

		temp = X.sample(k)

		X_tr = temp.iloc[:, 0:13].values
		y_tr = temp.iloc[:, -1].values

		Lw = w
		Lb = b

		loss = 0
		y_pred = []
		sq_loss = []

		for i in range(k):
			Lw = (-2 / k * X_tr[i]) * (y_tr[i] - np.dot(X_tr[i], w.T) - b)

			w = w - learning_rate * Lw

			y_predicted = np.dot(X_tr[i], w.T)
			y_pred.append(y_predicted)

		loss = get_mse(y_pred, y_tr)

		print("Epoch: %d, Loss: %.3f" % (epoch, loss))
		epoch += 1
		learning_rate = learning_rate / 1.02

	return w, b


# predicts y_hat
def polynomial_fun(X, weights, degree):
	X_transform = transform(X, degree)
	return tf.matmul(X_transform, weights)


if __name__ == "__main__":

