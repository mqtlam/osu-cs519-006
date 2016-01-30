import numpy as np
from layers.core import Layer

class ReluLayer(Layer):
	def __init__(self):
		Layer.__init__(self)

	def forward(self, x):
		output = x * (x > 0)
		self.output = output
		return output

	def backward(self, x, grad):
		return 1 * (x > 0) + 0 * (x < 0) + np.random.uniform(0, 1, x.shape) * (x == 0)

	def __str__(self):
		string = "ReluLayer"
		return string
