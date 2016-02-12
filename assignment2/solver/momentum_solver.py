import numpy as np

from solver.core import Solver

class MomentumSolver(Solver):
	# default learning rate alpha
	DEFAULT_ALPHA = 0.001

	# default momentum mu
	DEFAULT_MU = 0.7

	def __init__(self, **kwargs):
		"""Initialization.

		Args:
			lr (kwargs): learning rate otherwise use default
			mu (kwargs): momentum otherwise use default
		"""
		self.alpha = kwargs["lr"] if "lr" in kwargs else MomentumSolver.DEFAULT_ALPHA
		self.mu = kwargs["mu"] if "mu" in kwargs else MomentumSolver.DEFAULT_MU
		self.reset()

	def update(self, weights, gradient, layer_id):
		"""Computes a weight update.

		Uses momentum to update weights.

		Args:
			weights: current weights
			gradient: gradient of weights
			layer_id: id of layer currently updating

		Returns:
			updated weights
		"""
		if layer_id not in self.momentum:
			self.momentum[layer_id] = np.zeros(weights.shape)

		# compute batch gradient average
		batch_size = gradient.shape[2]
		gradient_sum = np.sum(gradient, 2)
		batch_gradient = 1./batch_size * gradient_sum
		if batch_gradient.shape[1] == 1:
			batch_gradient = batch_gradient[:,0]

		# momentum update
		self.momentum[layer_id] = self.mu * self.momentum[layer_id] - self.alpha * batch_gradient
		result = weights + self.momentum[layer_id]
		return result

	def reset(self):
		"""Resets parameters of solver."""
		self.momentum = {}
