import numpy as np
from layers.core import Layer

class SoftMaxLayer(Layer):
	def __init__(self):
		pass

	def forward(self, x):
		x_exp = np.exp(x)
		return 1.*x_exp / np.sum(x_exp)

	def backward(self, x, grad):
		return x #TODO

	def __str__(self):
		string = "SoftMaxLayer"
		return string
