from scipy.special import expit
from layers.core import Layer

class SigmoidLayer(Layer):
	def __init__(self):
		Layer.__init__(self)

	def computeOutput(self, input):
		# expit is sigmoid
		return expit(input)

	def computeGradInput(self, input, out, gradOut):
		# TODO
		return expit(input)*(1-expit(input))

	def updateParams(self, solver):
		# sigmoid has no parameters
		pass

	def __str__(self):
		string = "SigmoidLayer"
		return string
