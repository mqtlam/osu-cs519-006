from solver.core import Solver

class EasySolver(Solver):
	DEFAULT_LEARNING_RATE = 0.1

	def __init__(self, learning_rate=DEFAULT_LEARNING_RATE):
		self.learning_rate = learning_rate

	def update(self, weights, gradient, layer_id):
		return weights - self.learning_rate * gradient

	def reset(self):
		pass
