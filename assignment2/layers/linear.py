import numpy as np
from layers.core import Layer

class LinearLayer(Layer):
	DEFAULT_INITIAL_MU = 0
	DEFAULT_INITIAL_SIGMA = 0.1

	def __init__(self, input_dim, output_dim, **kwargs):
		Layer.__init__(self)

		self.input_dim = input_dim
		self.output_dim = output_dim

		self.mu = kwargs["mu"] if "mu" in kwargs else LinearLayer.DEFAULT_INITIAL_MU
		self.sigma = kwargs["sigma"] if "sigma" in kwargs else LinearLayer.DEFAULT_INITIAL_SIGMA

		if output_dim == 1:
			shape = (input_dim,)
		else:
			shape = (output_dim, input_dim)
		self.W = np.random.normal(self.mu, self.sigma, shape)

		if output_dim == 1:
			self.b = 0
		else:
			self.b = np.zeros(output_dim)

	def forward(self, x):
		output = np.dot(self.W, x) + self.b
		self.output = output
		return output

	def backward(self, x, grad):
		return x

	def __str__(self):
		string = "LinearLayer: input_dim={0}, output_dim={1}".format(self.input_dim, self.output_dim)
		return string
