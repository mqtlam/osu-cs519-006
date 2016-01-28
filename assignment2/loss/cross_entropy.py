import numpy as np
from loss.core import Loss

class CrossEntropyLoss(Loss):
	def __init__(self):
		pass

	def forward(self, output, target):
		# return -1.*np.log(np.exp(output[target]) / (np.sum(np.exp(output)))
		return -output[target] + np.log(np.sum(np.exp(output)))

	def backward(self, output, target):
		return 0.1 # TODO

	def __str__(self):
		string = "CrossEntropyLoss"
		return string
