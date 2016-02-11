import numpy as np

from solver.core import Solver

class MomentumSolver(Solver):
	DEFAULT_ALPHA = 0.001
	DEFAULT_MU = 0.7

	def __init__(self, **kwargs):
		self.alpha = kwargs["lr"] if "lr" in kwargs else MomentumSolver.DEFAULT_ALPHA
		self.mu = kwargs["mu"] if "mu" in kwargs else MomentumSolver.DEFAULT_MU
		self.reset()

	def update(self, weights, gradient, layer_id):
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
		self.momentum = {}
