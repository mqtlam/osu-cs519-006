import numpy as np
from layers.core import Layer

class SoftMaxLayer(Layer):
	def __init__(self):
		Layer.__init__(self)

	def computeOutput(self, input):
		input_exp = np.exp(input)
		return 1.*input_exp / np.sum(input_exp)

	def computeGradInput(self, input, out, gradOut):
		d = out.shape[0]
		gradIn = np.zeros((d, d))
		for i in range(d):
			for j in range(d):
				gradIn[i][j] = out[i]*(int(i==j)-out[j])
		return gradOut*gradIn

	def updateParams(self, solver):
		# soft max has no parameters
		pass

	def __str__(self):
		string = "SoftMaxLayer"
		return string
