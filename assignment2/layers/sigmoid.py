import numpy as np
from scipy.special import expit
from layers.core import Layer

class SigmoidLayer(Layer):
	def __init__(self):
		Layer.__init__(self)

	def computeOutput(self, input):
		"""Compute the output of layer from forward pass."""
		# expit is sigmoid
		return expit(input)

	def sigmoidGrad(self, input, gradOut):
		"""Compute the gradient w.r.t the input.

		Args:
			input: input data
			gradOut: gradient from output

		Returns:
			input gradient
		"""
		gradIn = np.diag(expit(input)*(1-expit(input)))
		return np.dot(gradOut, gradIn)

	def computeGradInput(self, input, out, gradOut):
		"""Compute the input gradient of layer from backward pass."""
		input_unbatched = self.__unbatch__(input)
		gradOut_unbatched = self.__unbatch__(gradOut)
		output = []
		for i in range(len(gradOut_unbatched)):
			output.append(self.sigmoidGrad(input_unbatched[i],
						  gradOut_unbatched[i]))
		output = self.__batch__(output)
		return output

	def updateParams(self, solver):
		# sigmoid has no parameters
		pass

	def __str__(self):
		string = "SigmoidLayer"
		return string
