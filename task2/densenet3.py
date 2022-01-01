"""
Implementation of DenseNet with modifications
paper: https://arxiv.org/pdf/1608.06993.pdf
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, ZeroPadding2D, Dense, Dropout, Activation, Convolution2D, Reshape
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization


class DenseNet3:
	"""
	Each dense block contains 4 convolutional layers. The network has 3 of these dense blocks.
	:param nb_layers: type is list.
					Length of the list indicates the number of dense blocks,
					while each number in the list indicates the number of convolutional layers

	Parameters below follows the paper.
	:param growth_rate: set to 32
	:param compression: set to 0.5
	:param dropout_rate: set to 0.2
	:param batch_size: set to 64
	"""
	def __init__(self, nb_layers=[4, 4, 4], classes=10, shape=(32, 32, 3), growth_rate=32, compression=0.5,
	             dropout_rate=0.2, batch_size=64, with_output_block=True, with_se_layers=False):
		self.dropout_rate = dropout_rate
		self.with_se_layers = with_se_layers
		self.with_output_block = with_output_block
		self.batch_size = batch_size
		self.shape = shape
		self.classes = classes
		self.compression = compression
		self.nb_layers = nb_layers
		self.growth_rate = growth_rate

	def dense_net(self, nb_filter=64):
		"""

		:param nb_filter:
		:return:
		"""
		# compression factor is set to 0.5 following the experiment in the paper
		"""To further improve model compactness, we can reduce the number of feature-maps at transition layers. If a 
		dense block contains m feature-maps, we let the following transition layer generate ⌊θm⌋ output feature maps, 
		where 0 < θ ≤ 1 is referred to as the compression factor. When θ = 1, the number of feature-maps across 
		transition layers remains unchanged. We refer the DenseNet with θ < 1 as DenseNet-C, and we set θ = 0.5 in 
		our experiment. 
		When both the bottleneck and transition layers with θ < 1 are used, we refer to our model as DenseNet-BC. 
		(from paper)
		"""

		nb_dense_block = len(self.nb_layers)

		img_input = Input(shape=self.shape, name='data')

		x = ZeroPadding2D((3, 3), name='conv1_zeropadding', batch_size=self.batch_size)(img_input)
		x = Convolution2D(nb_filter, 7, 2, name='conv1', use_bias=False)(x)
		x = BatchNormalization(name='conv1_bn')(x)
		x = Activation('relu', name='relu1')(x)
		x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
		x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

		stage = 0
		# Add dense blocks
		for block_idx in range(nb_dense_block - 1):
			stage = block_idx + 2
			x, nb_filter = self.dense_block(x, stage, self.nb_layers[block_idx], nb_filter, self.growth_rate,
			                                dropout_rate=self.dropout_rate)

			if self.with_se_layers:
				x = self.se_block(x, stage, 'dense', nb_filter)

			# Add transition_block
			x = self.transition_block(x, stage, nb_filter, compression=self.compression, dropout_rate=self.dropout_rate)
			nb_filter = int(nb_filter * self.compression)

			if self.with_se_layers:
				x = self.se_block(x, stage, 'transition', nb_filter)

		final_stage = stage + 1
		x, nb_filter = self.dense_block(x, final_stage, self.nb_layers[-1], nb_filter, self.growth_rate,
		                                dropout_rate=self.dropout_rate)

		if self.with_se_layers:
			x = self.se_block(x, final_stage, 'dense', nb_filter)

		x = BatchNormalization(name='conv_final_blk_bn')(x)
		x = Activation('relu', name='relu_final_blk')(x)

		# TODO: can erase this and erase self.with_output_block
		if not self.with_output_block:
			return Model(inputs=img_input, outputs=x)

		x = GlobalAveragePooling2D(name='pool_final')(x)
		x = Dense(self.classes, name='fc6')(x)
		output = Activation('softmax', name='prob')(x)

		return Model(inputs=img_input, outputs=output)

	def conv_block(self, x, stage, branch, nb_filter, dropout_rate=None):
		conv_name_base = 'conv' + str(stage) + '_' + str(branch)
		relu_name_base = 'relu' + str(stage) + '_' + str(branch)

		"""
		Bottleneck layer
		1×1 convolution can be introduced as bottleneck layer before each 3×3 convolution to reduce the number of 
		input feature-maps, and thus to improve computational efficiency. (from paper)
		"""
		inter_channel = nb_filter * 4
		x = BatchNormalization(name=conv_name_base + '_x1_bn')(x)
		x = Activation('relu', name=relu_name_base + '_x1')(x)
		x = Convolution2D(inter_channel, 1, 1, name=conv_name_base + '_x1', use_bias=False)(x)

		if dropout_rate:
			x = Dropout(dropout_rate)(x)

		# 3x3 Convolution
		x = BatchNormalization(name=conv_name_base + '_x2_bn')(x)
		x = Activation('relu', name=relu_name_base + '_x2')(x)
		x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(x)
		x = Convolution2D(nb_filter, 3, 1, name=conv_name_base + '_x2', use_bias=False)(x)

		if dropout_rate:
			x = Dropout(dropout_rate)(x)
		return x

	def se_block(self, x, stage, previous, nb_filter, ratio=16):
		se_name = 'se' + str(stage) + '_' + previous
		init = x
		x = GlobalAveragePooling2D(name='global_average_pooling_2d_' + se_name)(x)
		x = Dense(nb_filter // ratio, name='dense_relu_' + se_name)(x)
		x = Activation('relu', name='relu_' + se_name)(x)
		x = Dense(nb_filter, name='dense_sigmoid_' + se_name)(x)
		x = Activation('sigmoid', name='sigmoid_' + se_name)(x)
		x = tf.expand_dims(x, 1)
		x = init * tf.expand_dims(x, 1)
		return x

	def dense_block(self, x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, grow_nb_filters=True):
		concat_feat = x
		for i in range(nb_layers):
			branch = i + 1
			x = self.conv_block(concat_feat, stage, branch, growth_rate, dropout_rate)
			concat_feat = tf.concat([concat_feat, x], -1)

			if grow_nb_filters:
				nb_filter += growth_rate

		return concat_feat, nb_filter

	def transition_block(self, x, stage, nb_filter, compression=0.5, dropout_rate=None):
		"""
		To reduce the size, DenseNet uses transition layers.
		These layers contain convolution with kernel size = 1 followed by
		2x2 average pooling with stride = 2.
		It reduces height and width dimensions but leaves feature dimension the same.
		:param x:
		:param stage:
		:param nb_filter:
		:param compression:
		:param dropout_rate:
		:return:
		"""
		conv_name_base = 'tran_conv' + str(stage) + '_blk'
		relu_name_base = 'tran_relu' + str(stage) + '_blk'
		pool_name_base = 'tran_pool' + str(stage)

		x = BatchNormalization(name=conv_name_base + '_bn')(x)
		x = Activation('relu', name=relu_name_base)(x)
		x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, use_bias=False)(x)

		if dropout_rate:
			x = Dropout(dropout_rate)(x)

		x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

		return x


if __name__ == '__main__':
	dense_net_class = DenseNet3(classes=10)
	dense_net = dense_net_class.dense_net()
	dense_net.summary()

