import numpy as np
from layers.core import Layer

class LinearLayer(Layer):
	def __init__(self, input_dim, output_dim):
		self.mu = 0
		self.sigma = 1
		self.input_dim = input_dim
		self.output_dim = output_dim
		if output_dim == 1:
			shape = (input_dim,)
		else:
			shape = (output_dim, input_dim)
		self.W = np.random.normal(self.mu, self.sigma, shape)

	def forward(self, x):
		return np.dot(self.W, x)

	def backward(self, x, grad):
		pass

	def __str__(self):
		string = "LinearLayer: input_dim={0}, output_dim={1}".format(self.input_dim, self.output_dim)
		return string
