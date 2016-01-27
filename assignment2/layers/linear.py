import numpy as np
from layers.core import Layer

class LinearLayer(Layer):
	def __init__(self, shape):
		self.mu = 0
		self.sigma = 1
		self.W = np.random.normal(self.mu, self.sigma, shape)

	def forward(self, x):
		return np.dot(self.W, x)

	def backward(self, x):
		pass
