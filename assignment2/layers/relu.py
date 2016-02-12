import numpy as np
from layers.core import Layer

class ReluLayer(Layer):
	def __init__(self):
		Layer.__init__(self)

	def computeOutput(self, input):
		"""Compute the output of layer from forward pass."""
		return input * (input > 0)

	def reluGrad(self, input, gradOut):
		"""Compute the gradient w.r.t the input.

		Args:
			input: input data
			gradOut: gradient from output

		Returns:
			input gradient
		"""
		gradIn = np.diag(1 * (input > 0) + 0 * (input < 0) + np.random.uniform(0, 1, input.shape) * (input == 0))
		return np.dot(gradOut, gradIn)

	def computeGradInput(self, input, out, gradOut):
		"""Compute the input gradient of layer from backward pass."""
		input_unbatched = self.__unbatch__(input)
		gradOut_unbatched = self.__unbatch__(gradOut)
		output = []
		for i in range(len(gradOut_unbatched)):
			output.append(self.reluGrad(input_unbatched[i],
						  gradOut_unbatched[i]))
		output = self.__batch__(output)
		return output

	def updateParams(self, solver):
		# relu has no parameters
		pass

	def __str__(self):
		string = "ReluLayer"
		return string
