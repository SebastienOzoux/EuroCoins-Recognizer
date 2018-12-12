from scipy.ndimage import rotate
from scipy.misc import imread, imsave
import os
import numpy as np
from tqdm import tqdm
from skimage import exposure

train_data='/home/sebastien/Euros_Recognition/data2'


for i in tqdm(os.listdir(train_data)):
	path = os.path.join(train_data, i)
    img = imread(path)
	label=(i.split('.')[0]+ '.' + i.split('.')[1])

	# Contrast stretching
	p2, p98 = np.percentile(img, (2, 98))
	img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
	imsave('/home/sebastien/Euros_Recognition/data/' + label + '-mod1' + '.jpeg', img_rescale)

	# Histogram Equalization
	img_eq = exposure.equalize_hist(img)
	imsave('/home/sebastien/Euros_Recognition/data/' + label + '-mod2' + '.jpeg', img_eq)

	# Adaptive Equalization
	img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
	imsave('/home/sebastien/Euros_Recognition/data/' + label + '-mod3' + '.jpeg', img_adapteq)

	rotate_img = rotate(img, 90)
	imsave('/home/sebastien/Euros_Recognition/data/' + label + '-90' + '.jpeg', rotate_img)

	rotate_img = rotate(img, 180)
	imsave('/home/sebastien/Euros_Recognition/data/' + label + '-180' + '.jpeg', rotate_img)

	rotate_img = rotate(img, 270)
	imsave('/home/sebastien/Euros_Recognition/data/' + label + '-270' + '.jpeg', rotate_img)

	#flip_img = np.fliplr(img)
	#imsave('/home/sebastien/Euros_Recognition/train/' + label + '-flip' + '.jpeg', flip_img)




