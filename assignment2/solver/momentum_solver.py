import numpy as np

from solver.core import Solver

class MomentumSolver(Solver):
	DEFAULT_ALPHA = 0.01
	DEFAULT_MU = 0.6

	def __init__(self, **kwargs):
		self.alpha = kwargs["lr"] if "lr" in kwargs else MomentumSolver.DEFAULT_ALPHA
		self.mu = kwargs["mu"] if "mu" in kwargs else MomentumSolver.DEFAULT_MU
		self.reset()

	def update(self, weights, gradient, layer_id):
		if layer_id not in self.momentum:
			self.momentum[layer_id] = np.zeros(gradient.shape)

		self.momentum[layer_id] = self.mu * self.momentum[layer_id] - self.alpha * gradient
		return weights + self.momentum[layer_id]

	def reset(self):
		self.momentum = {}
