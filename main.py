import pathlib2 as path
import gzip
import shutil
import numpy as np
import matplotlib.pyplot as plt

TEST_IMAGES_FILENAME = "data/t10k-images-idx3-ubyte.gz"
TEST_LABELS_FILENAME = 'data/t10k-labels-idx1-ubyte.gz'
TRAINING_IMAGES_FILENAME = "data/train-images-idx3-ubyte.gz"
TRAINING_LABELS_FILENAME = 'data/train-labels-idx1-ubyte.gz'
CWD = path.Path.cwd()


'''
data from http://yann.lecun.com/exdb/mnist/
images for the MNIST are stored is a gzip format. 
read_images returns the data at three dimensional np array where each image is 28x28. 
Contains integers from 0 - 256, representing gray scale
Training : Boolean -> read in 60k training images or 10k trial images
'''


def read_images(training=True, num_of_images=None):
	if training:
		filepath = path.Path.joinpath(CWD, TRAINING_IMAGES_FILENAME)
		labelspath = path.Path.joinpath(CWD, TRAINING_LABELS_FILENAME)
		num_images = 60000
	else:
		filepath = path.Path.joinpath(CWD, TEST_IMAGES_FILENAME)
		labelspath = path.Path.joinpath(CWD, TEST_LABELS_FILENAME)
		num_images = 10000

	# a smaller specified amount of images to read in by the user
	if num_of_images:
		num_images = num_of_images

	with gzip.open(filepath, 'r') as data:

		image_size = 28
		data.read(16)  # skipping empty buffer
		buf = data.read(image_size * image_size * num_images)
		images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		images = images.reshape(num_images, image_size, image_size)

	with gzip.open(labelspath, 'r') as data:
		data.read(8)  # skipping empty buffer
		buf = data.read(num_images)
		labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
		print(labels)
		labels = labels.reshape(-1, 1)

	'''
	The images in the array \'images\' has one extra dimension than is necessary.
	For convenience, lets use the squeeze function
	'''

	return images, labels


def visualize_images(images, labels):
	i = 0
	for im in images:
		plt.imshow(im)
		plt.text(2.5, .5, labels[i], ha='center', )
		plt.show()
		i +=1

