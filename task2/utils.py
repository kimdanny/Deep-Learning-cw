from PIL import Image


# For Visulaization of cutouts
def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
	min_height = min(Image.fromarray(im).height for im in im_list)
	im_list_resize = [Image.fromarray(im).resize(
		(int(Image.fromarray(im).width * min_height / Image.fromarray(im).height), min_height), resample=resample)
		for im in im_list]
	total_width = sum(im.width for im in im_list_resize)
	dst = Image.new('RGB', (total_width, min_height))
	pos_x = 0
	for im in im_list_resize:
		dst.paste(im, (pos_x, 0))
		pos_x += im.width
	return dst


def get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
	min_width = min(im.width for im in im_list)
	im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)), resample=resample)
	                  for im in im_list]
	total_height = sum(im.height for im in im_list_resize)
	dst = Image.new('RGB', (min_width, total_height))
	pos_y = 0
	for im in im_list_resize:
		dst.paste(im, (0, pos_y))
		pos_y += im.height
	return dst