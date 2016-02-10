from module import Module

class Layer(Module):
	"""Abstract Layer class.
	"""

	def __init__(self):
		"""Initializes a Layer.

		The inputs, outputs, input gradients
		and output gradients are stored for
		computational efficiency.
		"""
		self.out = None
		self.gradIn = None
		self.input = None
		self.gradOut = None

	def forward(self, input):
		"""Perform a forward pass.

		Computes a forward pass and also stores the
		input and outputs.

		Args:
			input: input data to layer

		Returns:
			output of forward pass
		"""
		self.input = input
		self.out = self.computeOutput(input)
		return self.out

	def backward(self, gradOut):
		"""Performs a backward pass.

		Computes a backward pass and also stores the
		output gradients and input gradients.

		Note: assumes forward() was called before this otherwise it
		doesn't compute the gradients correctly!

		Args:
			gradOut: output gradient from backprop process

		Returns:
			gradient of input to layer for continuing backprop
		"""
		self.gradOut = gradOut
		self.gradIn = self.computeGradInput(self.input, self.out, gradOut)
		return self.gradIn

	def updateParams(self, solver):
		"""Update the parameters of the layer after backward pass.

		Specific layers derived from Layer are responsible
		for implementing this function.

		Some layers do not have parameters so simply override
		with a function that does nothing.

		Args:
			solver: solver object for computing weight updates
		"""
		raise NotImplementedError

	def computeOutput(self, input):
		"""Compute the output of layer from forward pass.

		Specific layers derived from Layer are responsible
		for implementing this function.

		Args:
			input: input data to layer

		Returns:
			output of forward pass
		"""
		raise NotImplementedError

	def computeGradInput(self, input, out, gradOut):
		"""Compute the input gradient of layer from backward pass.

		Specific layers derived from Layer are responsible
		for implementing this function.

		Args:
			input: input data to layer
			output: output data from forward pass
			gradOut: output gradient from backprop process

		Returns:
			gradient of input to layer for continuing backprop
		"""
		raise NotImplementedError

	def __str__(self):
		string =  "Layer (abstract)"
		return string
