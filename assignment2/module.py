import numpy as np

class Module:
	"""Module is an abstract class.
	It handles numpy array data.

	Assumes working with consistent data format
	for all inputs and outputs:
		mxnxb numpy array
			b refers to batch
			m,n are arbitrary
	"""

	def __unbatch__(self, input):
		"""Takes a mxnxb numpy array and returns
		a list of b mxn numpy arrays.

		This is useful for separating the 3-d numpy array into
		a list of 2-d arrays to perform some operation on each 2-d array.

		Args:
			input: mxnxb numpy array

		Returns:
			list of b mxn numpy arrays
		"""
		batch_size = input.shape[2]
		output = np.dsplit(input, batch_size)
		for i in range(batch_size):
			output[i] = output[i][:,:,0]
		return output

	def __batch__(self, input):
		"""Takes a list of b mxn numpy arrays and returns
		a mxnxb numpy array.

		This is used after calling __unbatch__ and applying some operation.

		Args:
			input: list of b mxn numpy arrays

		Returns:
			mxnxb numpy array
		"""
		batch_size = len(input)
		for i in range(batch_size):
			input[i] = input[i][..., None]
		output = np.dstack(input)
		return output
