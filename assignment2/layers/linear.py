import numpy as np
from layers.core import Layer

class LinearLayer(Layer):
	# default normal mean for random weights initialization
	DEFAULT_INITIAL_MU = 0
	# default normal standard deviation for random weights initialization
	DEFAULT_INITIAL_SIGMA = 0.1

	def __init__(self, input_dim, output_dim, **kwargs):
		"""Initialization.

		Weights are randomly initialized from a normal distribution.
		Biases are initialized to zero.

		Args:
			input_dim: input dimensionality to layer
			output_dim: output dimensionality from layer
			mu (kwargs): set the mean for random initialization
			sigma (kwargs): set the standard deviation for random initialization
		"""
		Layer.__init__(self)

		self.input_dim = input_dim
		self.output_dim = output_dim

		self.mu = kwargs["mu"] if "mu" in kwargs else LinearLayer.DEFAULT_INITIAL_MU
		self.sigma = kwargs["sigma"] if "sigma" in kwargs else LinearLayer.DEFAULT_INITIAL_SIGMA

		# initialize weights
		if output_dim == 1:
			shape = (input_dim,)
		else:
			shape = (output_dim, input_dim)
		self.W = np.random.normal(self.mu, self.sigma, shape)

		# initialize biases
		if output_dim == 1:
			self.b = 0
		else:
			self.b = np.zeros(output_dim)

	def linear(self, input):
		"""Performs linear operation: W*x+b

		Args:
			input: mx1 numpy array
				m = input dimensionality

		Returns:
			nx1 numpy array (n = output dimensionality)
		"""
		return np.dot(self.W, input) + self.b

	def computeOutput(self, input):
		"""Compute the output of layer from forward pass."""
		input_unbatched = self.__unbatch__(input)
		for i in range(len(input_unbatched)):
			input_unbatched[i] = self.linear(input_unbatched[i].reshape(-1))
		output = self.__batch__(input_unbatched)
		return output

	def linearGrad(self, gradOut):
		"""Compute the gradient w.r.t the input.

		Args:
			gradOut: gradient from output

		Returns:
			input gradient
		"""
		return np.dot(gradOut, self.W)

	def computeGradInput(self, input, out, gradOut):
		"""Compute the input gradient of layer from backward pass."""
		gradOut_unbatched = self.__unbatch__(gradOut)
		output = []
		for i in range(len(gradOut_unbatched)):
			output.append(self.linearGrad(gradOut_unbatched[i].reshape(-1)))
		output = self.__batch__(output)
		return output

	def gradW(self, input, gradOut):
		"""Compute the gradient w.r.t. the weights.

		Args:
			input: input data
			gradOut: gradient from output

		Returns:
			gradient w.r.t. weights
		"""
		gradOut_tile = np.tile(gradOut[...,None], (1, self.input_dim))
		input_tile = np.tile(input, (self.output_dim, 1))
		W_grad = gradOut_tile * input_tile
		return W_grad

	def gradB(self, gradOut):
		"""Compute the gradient w.r.t. the biases.

		Args:
			gradOut: gradient from output

		Returns:
			gradient w.r.t. biases
		"""
		b_grad = np.dot(gradOut, np.identity(self.output_dim))
		return b_grad

	def updateParams(self, solver):
		"""Update the parameters of the layer after backward pass."""
		input_unbatched = self.__unbatch__(self.input)
		gradOut_unbatched = self.__unbatch__(self.gradOut)
		W_grad = []
		b_grad = []
		for i in range(len(gradOut_unbatched)):
			W_grad.append(self.gradW(input_unbatched[i].reshape(-1),
									 gradOut_unbatched[i].reshape(-1)))
			b_grad.append(self.gradB(gradOut_unbatched[i].reshape(-1)))
		W_grad = self.__batch__(W_grad)
		b_grad = self.__batch__(b_grad)

		self.W = solver.update(self.W, W_grad, id(self))
		self.b = solver.update(self.b, b_grad, id(self)+1)

	def __str__(self):
		string = "LinearLayer: input_dim={0}, output_dim={1}".format(self.input_dim, self.output_dim)
		return string
