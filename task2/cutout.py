"""
Implementation of Cutout with modifications
https://arxiv.org/pdf/1708.04552.pdf
"""
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.datasets import cifar10
from utils import get_concat_h_multi_resize, get_concat_v_multi_resize



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


	(x_train, _), (_, _) = cifar10.load_data()
	sample_16 = x_train[:16]
	cutout_sample_16 = np.array([cutout_class.cutout(im).astype('uint8') for im in sample_16])

	# 16 original pictures concatenated horizontally
	original_concated = get_concat_h_multi_resize(sample_16)
	# 16 cutout pictures concatenated horizontally
	cutout_concated = get_concat_h_multi_resize(cutout_sample_16)
	# both images above concatenated vertically
	get_concat_v_multi_resize([original_concated, cutout_concated]).save('cutout.png')
