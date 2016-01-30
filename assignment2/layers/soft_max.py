import numpy as np
from layers.core import Layer

class SoftMaxLayer(Layer):
	def __init__(self):
		Layer.__init__(self)

	def forward(self, x):
		x_exp = np.exp(x)
		output = 1.*x_exp / np.sum(x_exp)
		self.output = output
		return output

	def backward(self, x, grad):
		d = self.output.shape[0]
		gradInput = np.zeros((d, d))
		for i in range(d):
			for j in range(d):
				gradInput[i][j] = self.output[i]*(int(i==j)-self.output[j])
		return grad*gradInput

	def __str__(self):
		string = "SoftMaxLayer"
		return string
