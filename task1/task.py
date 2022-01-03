# from polynomial_regression import polynomial_fun
import tensorflow as tf


# predicts y_hat
def polynomial_fun(X, weights):
	return tf.matmul(X, weights)




if __name__ == '__main__':
