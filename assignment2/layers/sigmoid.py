from scipy.special import expit
from layers.layer import Layer

class SigmoidLayer(Layer):
	def __init__():
		pass

	def forward(x):
		# expit is sigmoid
		return expit(x)

	def backward(x):
		return expit(x)*(1-expit(x))
