import numpy as np

class DataPreprocessing:
	"""Utility functions for preprocessing data.
	"""

	# constants
	RGB_MIN = 0
	RGB_MAX = 255
	NORMALIZED_MIN = 0
	NORMALIZED_MAX = 1

	@staticmethod
	def convert_range(data, old_min, old_max, new_min, new_max):
		"""Convert data from old range to new range.

		Args:
			data: data to convert
			old_min: old minimum in range
			old_max: old maximum in range
			new_min: new minimum in range
			new_max: new maximum in range

		Returns:
			data converted from old range to new range
		"""
		old_range = old_max - old_min
		new_range = new_max - new_min
		return (((1.*data - old_min) * new_range) / old_range) + new_min

	@staticmethod
	def rgb2normalized(data):
		"""Normalize RGB data [0, 255] to [0,1].

		Args:
			data: RGB data

		Returns:
			normalized data
		"""
		return DataPreprocessing.convert_range(data, DataPreprocessing.RGB_MIN, DataPreprocessing.RGB_MAX, DataPreprocessing.NORMALIZED_MIN, DataPreprocessing.NORMALIZED_MAX)

	@staticmethod
	def normalized2rgb(data):
		"""Convert normalized data [0,1] to RGB [0, 255].

		Args:
			data: normalized data

		Returns:
			RGB data
		"""
		return DataPreprocessing.convert_range(data, DataPreprocessing.NORMALIZED_MIN, DataPreprocessing.NORMALIZED_MAX, DataPreprocessing.RGB_MIN, DataPreprocessing.RGB_MAX)

	@staticmethod
	def compute_mean(data):
		"""Compute mean of data along the first axis.

		Args:
			data: assumes mxn numpy array
				m = number of examples
				n = feature dimension of examples

		Returns:
			mean of data (mx1) numpy array
		"""
		return np.mean(data, axis=0)

	@staticmethod
	def compute_std(data):
		"""Compute standard deviation of data along the first axis.

		Args:
			data: assumes mxn numpy array
				m = number of examples
				n = feature dimension of examples

		Returns:
			standard deviation of data (mx1) numpy array
		"""
		return np.std(data, axis=0)

	@staticmethod
	def normalize_data(data, **kwargs):
		"""Normalize the data using the mean and standard deviation.

		Args:
			data: assumes mxn numpy array
				m = number of examples
				n = feature dimension of examples
			mean (kwargs): use the mean provided
			std (kwargs): use the standard deviation provided

		Returns:
			normalized data
		"""
		mean = kwargs["mean"] if "mean" in kwargs else DataPreprocessing.compute_mean(data)
		std = kwargs["std"] if "std" in kwargs else DataPreprocessing.compute_std(data)
		return (1.*data - mean) / std
