"""
Implementation of Cutout with modifications
https://arxiv.org/pdf/1708.04552.pdf
"""
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.datasets import cifar10
from utils import get_concat_h_multi_resize, get_concat_v_multi_resize
from PIL import Image, ImageFont, ImageDraw


class Cutout(Layer):
	"""
	Use square masks with variable size and location.
	Add an additional parameter s, such that the mask size can be uniformly sampled from [0, s].
	Location should be sampled uniformly in the image space.
	N.B. care needs to be taken around the boundaries, so the sampled mask maintains its size.
	"""

	def __init__(self, s=16 * 16, n_holes=1, **kwargs):
		super(Cutout, self).__init__(**kwargs)
		self.s = s
		self.n_holes = n_holes

	def call(self, image, training=None):
		if not training:
			return image

		return self.cutout(image)

	def cutout(self, input_img) -> np.ndarray:
		"""
		Cutout implementation. Cuts out square(s) from an image and return it.

		:param input_img: source image
		:param s: cutout size (area) that will be uniformly sampled from [0, s]
		:param n_holes: number of masks to make
		:return: cutout-processed image
		"""
		img_h, img_w, _ = input_img.shape

		mask = np.ones((img_h, img_w), np.float32)

		for n in range(self.n_holes):
			while True:
				size = np.random.uniform(0, self.s)
				length = int(size ** 0.5)
				# randint returns random integers from the "discrete uniform" distribution of the specified dtype
				# in the “half-open” interval [low, high)
				y = np.random.randint(img_h)
				x = np.random.randint(img_w)

				y1 = y - length // 2
				y2 = y + length // 2
				x1 = x - length // 2
				x2 = x + length // 2

				# condition that is not allowed (box out of image border)
				if y1 < 0 or y2 > img_h or x1 < 0 or x2 > img_w:
					pass
				# if condition is met, mask the part with 0.
				else:
					mask[y1: y2, x1: x2] = 0.
					break

		# broadcast the mask to all channels
		mask = mask[:, :, np.newaxis]
		result_img = input_img * mask

		return result_img


if __name__ == '__main__':
	cutout_class = Cutout()

	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	# sample_16 = x_train[:16]
	# cutout_sample_16 = np.array([cutout_class.cutout(im).astype('uint8') for im in sample_16])
	#
	# # 16 original pictures concatenated horizontally
	# original_concated = get_concat_h_multi_resize(sample_16)
	# # 16 cutout pictures concatenated horizontally
	# cutout_concated = get_concat_h_multi_resize(cutout_sample_16)
	# # both images above concatenated vertically
	# get_concat_v_multi_resize([original_concated, cutout_concated]).save('cutout.png')
	class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

	from tensorflow.keras.models import load_model

	model = load_model('./cifar10-densenet3.h5')

	# model prediction
	n_sample = 6
	x_test_samples = x_test[:n_sample]
	y_test_samples = y_test[:n_sample]
	y_class_names = []
	y_hat_class_names = []

	for x_sample, y_sample in zip(x_test_samples, y_test_samples):
		image = x_sample.reshape(1, 32, 32, 3)
		y_hat = model.predict(image)
		y_hat = int(np.argmax(y_hat[0]))
		# print(y_hat)
		# print(y_sample)
		y_hat_class_name = class_names[y_hat]
		y_class_name = class_names[y_sample[0]]

		y_class_names.append(y_class_name)
		y_hat_class_names.append(y_hat_class_names)

	# Draw the image with captions
	# TODO: complete visualising the result.png
	image_list = []
	for i in range(n_sample):
		im = Image.fromarray(x_test_samples[i])
		title_text = f'truth: {y_hat_class_names[i]} || pred: {y_hat_class_names[i]}'

		# draw captions
		image_editable = ImageDraw.Draw(im)
		image_editable.text((15, 15), title_text, (237, 230, 211))
		image_list.append(im)

	image_list[1].show()