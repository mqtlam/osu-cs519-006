from scipy.special import expit
from layers.core import Layer

class SigmoidLayer(Layer):
	def __init__(self):
		pass

	def forward(self, x):
		# expit is sigmoid
		return expit(x)

	def backward(self, x):
		return expit(x)*(1-expit(x))
