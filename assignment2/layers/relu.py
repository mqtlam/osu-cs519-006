import numpy as np
from layers.core import Layer

class ReluLayer(Layer):
	def __init__(self):
		Layer.__init__(self)

	def computeOutput(self, input):
		return input * (input > 0)

	def computeGradInput(self, input, out, gradOut):
		# TODO
		return 1 * (input > 0) + 0 * (input < 0) + np.random.uniform(0, 1, input.shape) * (input == 0)

	def updateParams(self, solver):
		# relu has no parameters
		pass

	def __str__(self):
		string = "ReluLayer"
		return string
