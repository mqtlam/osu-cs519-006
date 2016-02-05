class Solver:
	def update(self, weights, gradient):
		"""Computes a weight update.

		Specific solvers derived from Solver are responsible 
		for implementing this function.

		Args:
			weights: current weights
			gradient: gradient of weights

		Returns:
			updated weights
		"""
		raise NotImplementedError
