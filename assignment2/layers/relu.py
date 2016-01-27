import numpy as np
from layers.layer import Layer

class ReluLayer(Layer):
	def __init__():
		pass

	def forward(x):
		return x * (x > 0)

	def backward(x):
		return 1 * (x > 0) + 0 * (x < 0) + np.random.uniform(0, 1, x.shape) * (x == 0)
