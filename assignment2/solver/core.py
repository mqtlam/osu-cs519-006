class Solver:
	def update(self, weights, gradient, layer_id):
		"""Computes a weight update.

		Specific solvers derived from Solver are responsible 
		for implementing this function.

		Args:
			weights: current weights
			gradient: gradient of weights
			layer_id: id of layer currently updating
				(useful for some solvers)

		Returns:
			updated weights
		"""
		raise NotImplementedError

	def reset(self):
		"""Resets parameters of solver.

		Specific solvers derived from Solver are responsible 
		for implementing this function.

		Some solvers do not need to reset its state so 
		simply override with a function that does nothing.
		"""
		raise NotImplementedError
