from scipy.special import expit
from layers.core import Layer

class SigmoidLayer(Layer):
	def __init__(self):
		Layer.__init__(self)

	def forward(self, x):
		# expit is sigmoid
		output = expit(x)
		self.output = output
		return output

	def backward(self, x, grad):
		return expit(x)*(1-expit(x))

	def __str__(self):
		string = "SigmoidLayer"
		return string
