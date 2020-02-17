import numpy as np
from main import read_images, visualize_images

'''
Input array is 28x28=784
this is my first attempt at a NN from scratch. 
Not sure how to set up a convolution NN
but i'll figure this out

With 784 inputs connected to 16 neurons -> 12544 synapses
16 neurons connected to 16 neurons -> 256 synapses
16 neurons connected to one output neuron -> 16 synapses
In total there is 12816 weights and biases to updates and refine

I ..S1.. H1 ..S2.. H2 ..S3.. O
NN diagram
'''


class NN:
	def __init__(self, input_size):
		# left input_size as variable. However, in context of problem, it will be 28x28=784
		self.in_layer = np.zeros(input_size, np.float32)
		# 16 hidden layer neurons
		self.hidden_layer_1 = np.zeros(16, np.float32)
		self.hidden_layer_2 = np.zeros(16, np.float32)
		# activation bias to change the tendency of a neuron to fire
		self.activation_bias_layer1 = np.random.random(16)*np.random.randint(0, 5)
		self.activation_bias_layer2 = np.random.random(16) * np.random.randint(0, 5)
		self.activation_bias_layer3 = np.random.random(10) * np.random.randint(0, 5)
		# 10 output layer neurons representing digits 0 - 9
		self.out_layer = np.zeros(10, np.float32)

		self.synapses1 = np.random.random(self.hidden_layer_1.size*self.in_layer.size).reshape(self.in_layer.size, -1)
		self.synapses2 = np.random.random(self.hidden_layer_1.size*self.hidden_layer_2.size).reshape(self.hidden_layer_1.size, -1)
		self.synapses3 = np.random.random(self.hidden_layer_2.size*self.out_layer.size).reshape(self.hidden_layer_2.size, -1)

		# initializing object variables for later
		self.image_data = np.array([1])
		self.images_labels = np.array([1])

	# derivative is used for gradient decent
	def sigmoid(self, val, deriv=False):
		if deriv:
			return val * (1 - val)
		else:
			return 1.0 / (1 + np.exp(-val))

	# pulls in image matrices from main
	def load_images(self, training=True, num_images: int=None):
		if not num_images:
			self.image_data, self.images_labels = read_images(training)
		else:
			self.image_data, self.images_labels = read_images(training, num_images)

	# makes an array of zeros and the correct index marked as 1 to compared in error function
	def make_correct_output(self, num):
		correct = np.zeros(10, dtype=np.float32)
		correct[self.images_labels[num]] = 1.0
		return correct

	# begins training, gradient decent, and back propagation
	def train(self):
		self.load_images(True, 60000)
		# for i in range(self.image_data.shape[0]):
		for i in range(self.image_data.shape[0]):
			curr_im = self.image_data[i]
			l0 = self.image_data[i].flatten()
			self.hidden_layer_1 = self.sigmoid(np.dot(l0, self.synapses1) + self.activation_bias_layer1)
			self.hidden_layer_2 = self.sigmoid(np.dot(self.hidden_layer_1, self.synapses2) + self.activation_bias_layer2)
			self.out_layer = self.sigmoid(np.dot(self.hidden_layer_2, self.synapses3) + self.activation_bias_layer3)

			correct = self.make_correct_output(i)
			output_error = correct - self.out_layer
			output_delta = output_error * self.sigmoid(self.out_layer, True)

			if (i % 100) == 0:
				print(f"%Error: {np.mean(np.abs(output_error))} {i} of {self.image_data.shape[0]}")

			hidden2_error = output_delta.dot(self.synapses3.T)
			hidden2_delta = hidden2_error * self.sigmoid(self.hidden_layer_2, True)

			hidden1_error = hidden2_delta.dot(self.synapses2.T)
			hidden1_delta = hidden1_error * self.sigmoid(self.hidden_layer_1, True)

			# update weights
			self.synapses1 += self.hidden_layer_1.T.dot(hidden1_delta)
			self.synapses2 += self.hidden_layer_2.T.dot(hidden2_delta)
			self.synapses3 += self.out_layer.T.dot(output_delta)

			# update biases (subtracting to reduce the
			self.activation_bias_layer1 += self.hidden_layer_1.T.dot(hidden1_delta)
			self.activation_bias_layer2 += self.hidden_layer_2.T.dot(hidden2_delta)
			self.activation_bias_layer3 += self.out_layer.T.dot(output_delta)

	def test(self, num_images: int):
		self.load_images(False, num_images)
		for i in range(num_images):
			# visualize_images(self.image_data[i:i+1], self.images_labels[i:i+1])
			l0 = self.image_data[i].flatten()
			self.hidden_layer_1 = self.sigmoid(np.dot(l0, self.synapses1) + self.activation_bias_layer1)
			self.hidden_layer_2 = self.sigmoid(
				np.dot(self.hidden_layer_1, self.synapses2) + self.activation_bias_layer2)
			self.out_layer = self.sigmoid(np.dot(self.hidden_layer_2, self.synapses3) + self.activation_bias_layer3)

			print(f"test {i}: real answer -> {self.images_labels[i]}, guess -> {np.argmax(self.out_layer)}")


nn = NN(784)
nn.train()
nn.test(50)
