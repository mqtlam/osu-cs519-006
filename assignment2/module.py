import numpy as np

class Module:
	def __unbatch__(self, input):
		"""Takes a mxnxb numpy array and returns
		a list of b mxn numpy arrays.

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
