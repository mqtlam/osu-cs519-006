import numpy as np

class DataPreprocessing:
	RGB_MIN = 0
	RGB_MAX = 255
	NORMALIZED_MIN = 0
	NORMALIZED_MAX = 1

	@staticmethod
	def convert_range(data, old_min, old_max, new_min, new_max):
		old_range = old_max - old_min
		new_range = new_max - new_min
		return (((1.*data - old_min) * new_range) / old_range) + new_min

	@staticmethod
	def rgb2normalized(data):
		return DataPreprocessing.convert_range(data, DataPreprocessing.RGB_MIN, DataPreprocessing.RGB_MAX, DataPreprocessing.NORMALIZED_MIN, DataPreprocessing.NORMALIZED_MAX)

	@staticmethod
	def normalized2rgb(data):
		return DataPreprocessing.convert_range(data, DataPreprocessing.NORMALIZED_MIN, DataPreprocessing.NORMALIZED_MAX, DataPreprocessing.RGB_MIN, DataPreprocessing.RGB_MAX)

	@staticmethod
	def compute_mean(data):
		return np.mean(data, axis=0)

	@staticmethod
	def compute_std(data):
		return np.std(data, axis=0)

	@staticmethod
	def normalize_data(data, **kwargs):
		mean = kwargs["mean"] if "mean" in kwargs else DataPreprocessing.compute_mean(data)
		std = kwargs["std"] if "std" in kwargs else DataPreprocessing.compute_std(data)
		return (1.*data - mean) / std
