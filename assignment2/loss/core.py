from module import Module

class Loss(Module):
	"""Abstract Loss class.
	"""

	def forward(output, target):
		"""Computes the loss of output and target.

		Args:
			output: predictions data to compare to ground truth
				cx1xb numpy array
					c = num of classes
					b = batch size
			target: ground truth labels (must be 0-indexed)
				1x1xb numpy array
					b = batch size

		Returns:
			loss values as 1x1xb numpy array
		"""
		raise NotImplementedError

	def backward(output, target):
		"""Computes the loss gradient of output and target.
		This is the first step for backpropagation.

		Args:
			output: predictions data to compare to ground truth
				cx1xb numpy array
					c = num of classes
					b = batch size
			target: ground truth labels (must be 0-indexed)
				1x1xb numpy array
					b = batch size

		Returns:
			loss gradient
		"""
		raise NotImplementedError
