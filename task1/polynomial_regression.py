import tensorflow as tf


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


# predicts y_hat
def polynomial_fun(X, weights, degree):
	X_transform = transform(X, degree)
	return tf.matmul(X_transform, weights)


if __name__ == "__main__":
	X = np.array([[1, 2, 3, 4]])
	X = X.T
	Y = np.array([3, 2, 0, 5])

	# model training
	model = PolynomialRegression(degree=3)

	model.fit(X, Y)
	print(model.weights)

	# Prediction on training set
	Y_pred = model.predict(X)
