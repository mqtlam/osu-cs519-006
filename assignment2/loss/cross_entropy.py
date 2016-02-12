import numpy as np
from loss.core import Loss

class CrossEntropyLoss(Loss):
	"""Cross entropy loss function.
	"""

	def __init__(self):
		pass

	def crossEntropy(self, output, target):
		"""Computes the cross entropy.

		Args:
			output: cx1 numpy array from soft max layer
				c = num of classes
			target: groundtruth label (number)

		Returns:
			cross entropy computation (number)
		"""
		return -np.log(output[target])

	def forward(self, output, target):
		"""Computes the loss of output and target."""
		output_unbatched = self.__unbatch__(output)
		for i in range(len(output_unbatched)):
			output_unbatched[i] = self.crossEntropy(output_unbatched[i], target[0,0,i])
		output = self.__batch__(output_unbatched)
		return output

	def crossEntropyGradient(self, output, target):
		"""Computes the gradient of cross entropy.

		Args:
			output: cx1 numpy array from soft max layer
				c = num of classes
			target: groundtruth label (number)

		Returns:
			cross entropy gradient computation (number)
		"""
		gradInput = np.zeros(output.shape)
		gradInput[target] = -1./output[target]
		return gradInput

	def backward(self, output, target):
		"""Computes the loss gradient of output and target."""
		output_unbatched = self.__unbatch__(output)
		for i in range(len(output_unbatched)):
			output_unbatched[i] = self.crossEntropyGradient(output_unbatched[i], target[0,0,i])
		output = self.__batch__(output_unbatched)
		return output

	def __str__(self):
		string = "CrossEntropyLoss"
		return string
