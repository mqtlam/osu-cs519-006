import numpy as np
from loss.core import Loss

class CrossEntropyLoss(Loss):
	def __init__(self):
		pass

	def forward(self, output, target):
		return -np.log(output[target])

	def backward(self, output, target):
		gradInput = np.zeros(output.shape)
		gradInput[target] = -1./output[target]
		return gradInput

	def __str__(self):
		string = "CrossEntropyLoss"
		return string
