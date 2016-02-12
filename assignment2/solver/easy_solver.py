import numpy as np
from solver.core import Solver

class EasySolver(Solver):
	"""EasySolver only uses a learning rate to udpate weights.
	"""

	# default learning rate
	DEFAULT_LEARNING_RATE = 0.001

	def __init__(self, learning_rate=DEFAULT_LEARNING_RATE):
		"""initialization.

		Args:
			learning_rate: set learning rate otherwise use default
		"""
		self.learning_rate = learning_rate

	def update(self, weights, gradient, layer_id):
		"""Computes a weight update.

		Simply uses a learning rate for updates.

		Args:
			weights: current weights
			gradient: gradient of weights
			layer_id: id of layer currently updating (not used here)

		Returns:
			updated weights
		"""
		# compute batch gradient average
		batch_size = gradient.shape[2]
		gradient_sum = np.sum(gradient, 2)
		batch_gradient = 1./batch_size * gradient_sum
		if batch_gradient.shape[1] == 1:
			batch_gradient = batch_gradient[:,0]

		# learning rate update
		result = weights - self.learning_rate * batch_gradient
		return result

	def reset(self):
		"""Resets parameters of solver."""
		pass
