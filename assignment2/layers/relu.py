import numpy as np
from layers.core import Layer

class ReluLayer(Layer):
	def __init__(self):
		pass

	def forward(self, x):
		return x * (x > 0)

	def backward(self, x):
		return 1 * (x > 0) + 0 * (x < 0) + np.random.uniform(0, 1, x.shape) * (x == 0)

	def __str__(self):
		string = "ReluLayer"
		return string
