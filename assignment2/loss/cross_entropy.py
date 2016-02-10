import numpy as np
from loss.core import Loss

class CrossEntropyLoss(Loss):
	def __init__(self):
		pass

	def crossEntropy(self, output, target):
		return -np.log(output[target])

	def forward(self, output, target):
		output_unbatched = self.__unbatch__(output)
		for i in range(len(output_unbatched)):
			output_unbatched[i] = self.crossEntropy(output_unbatched[i], target[0,0,i])
		output = self.__batch__(output_unbatched)
		return output

	def crossEntropyGradient(self, output, target):
		gradInput = np.zeros(output.shape)
		gradInput[target] = -1./output[target]
		return gradInput

	def backward(self, output, target):
		output_unbatched = self.__unbatch__(output)
		for i in range(len(output_unbatched)):
			output_unbatched[i] = self.crossEntropyGradient(output_unbatched[i], target[0,0,i])
		output = self.__batch__(output_unbatched)
		return output

	def __str__(self):
		string = "CrossEntropyLoss"
		return string
