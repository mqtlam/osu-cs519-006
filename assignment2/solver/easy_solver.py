import numpy as np
from solver.core import Solver

class EasySolver(Solver):
	DEFAULT_LEARNING_RATE = 0.0001

	def __init__(self, learning_rate=DEFAULT_LEARNING_RATE):
		self.learning_rate = learning_rate

	def update(self, weights, gradient, layer_id):
		batch_size = gradient.shape[2]
		gradient_sum = np.sum(gradient, 2)
		batch_gradient = 1./batch_size * gradient_sum
		if batch_gradient.shape[1] == 1:
			batch_gradient = batch_gradient[:,0]
		result = weights - self.learning_rate * batch_gradient
		return result

	def reset(self):
		pass
