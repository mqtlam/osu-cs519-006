import numpy as np
from layers.core import Layer

class SoftMaxLayer(Layer):
	def __init__(self):
		Layer.__init__(self)

	def softmax(self, input):
		input_exp = np.exp(input)
		return 1.*input_exp / np.sum(input_exp)

	def computeOutput(self, input):
		input_unbatched = self.__unbatch__(input)
		for i in range(len(input_unbatched)):
			input_unbatched[i] = self.softmax(input_unbatched[i])
		output = self.__batch__(input_unbatched)
		return output

	def softmaxGrad(self, out, gradOut):
		d = out.shape[0]
		gradIn = np.zeros((d, d))
		for i in range(d):
			for j in range(d):
				gradIn[i][j] = out[i]*(int(i==j)-out[j])
		return np.dot(gradOut, gradIn)

	def computeGradInput(self, input, out, gradOut):
		out_unbatched = self.__unbatch__(out)
		gradOut_unbatched = self.__unbatch__(gradOut)
		output = []
		for i in range(len(gradOut_unbatched)):
			output.append(self.softmaxGrad(out_unbatched[i].reshape(-1),
						  gradOut_unbatched[i].reshape(-1)))
		output = self.__batch__(output)
		return output

	def updateParams(self, solver):
		# soft max has no parameters
		pass

	def __str__(self):
		string = "SoftMaxLayer"
		return string
