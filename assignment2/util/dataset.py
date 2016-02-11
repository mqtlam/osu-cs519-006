try:
	import cPickle as pickle
except:
	import pickle

from util.data_preprocessing import DataPreprocessing
from sklearn.utils import shuffle

class Dataset:
	"""Abstract class for loading and accessing datasets.
	"""
	def __init__(self):
		pass

	def load(self, dataset_path):
		"""Load a dataset from a file.

		Args:
			dataset_path: path to dataset to read
		"""
		pass

	def save(self, dataset_path):
		"""Save the dataset to a file.

		Args:
			dataset_path: path to dataset to write
		"""
		pass

class CifarDataset(Dataset):
	"""Load and access the CIFAR dataset.
	It is mean to load the cifar_2class_py2.p data file.
	This class also contains functions to access the data in batches.
	"""
	def __init__(self):
		self.original_data = {}
		self.data = {}

	def load(self, dataset_path):
		"""Load a dataset from a file.

		Args:
			dataset_path: path to dataset to read
		"""
		with open(dataset_path, 'rb') as f:
			self.original_data = pickle.load(f)
		self.data = self.preprocess(self.original_data)
		self.data = self.shuffle_data(self.data)
		self.data = self.format_data(self.data)

	def save(self, dataset_path, protocol=3):
		"""Save the dataset to a file.

		Args:
			dataset_path: path to dataset to write
		"""
		with open(dataset_path, 'wb') as f:
			pickle.dump(self.original_data, f, protocol)

	def preprocess(self, data):
		"""Preprocess the training and test data by normalizing
		with the mean and standard deviation.

		Args:
			data: cifar data

		Returns:
			data preprocessed
		"""
		meanval = DataPreprocessing.compute_mean(data['train_data'])
		stdval = DataPreprocessing.compute_std(data['train_data'])
		data['train_data'] = DataPreprocessing.normalize_data(data['train_data'], mean=meanval, std=stdval)
		data['test_data'] = DataPreprocessing.normalize_data(data['test_data'], mean=meanval, std=stdval)
		return data

	def shuffle_data(self, data):
		"""Randomly shuffle the training and test data.

		Args:
			data: cifar data

		Returns:
			data shuffled
		"""
		data['train_data'], data['train_labels'] = shuffle(data['train_data'], data['train_labels'])
		data['test_data'], data['test_labels'] = shuffle(data['test_data'], data['test_labels'])
		return data

	def format_data(self, data):
		"""Format the data into "canonical form."
		This is how data is managed throughout the entire program:
			data: dx1xb numpy array
				d = feature dimension
				b = batch size
			labels: 1x1xb numpy array
				b = batch size
		Note that the last axis is always the batch.

		Args:
			data: cifar data

		Returns:
			data formatted
		"""
		data['train_data'] = data['train_data'][..., None]
		data['train_data'] = data['train_data'].transpose([1,2,0])
		data['train_labels'] = data['train_labels'].reshape((1,1,-1))
		data['test_data'] = data['test_data'][..., None]
		data['test_data'] = data['test_data'].transpose([1,2,0])
		data['test_labels'] = data['test_labels'].reshape((1,1,-1))
		return data

	def get_train_batches(self, batch_size):
		"""Generator for training batches.

		Args:
			batch_size: number of images in a batch_size

		Returns:
			generator for training batches
		"""
		num_train = self.get_num_train()
		for i in xrange(0, num_train, batch_size):
			# get batch
			train_data = self.get_train_data()[:,:,i:min(i+batch_size, num_train-1)]
			train_labels = self.get_train_labels()[:,:,i:min(i+batch_size, num_train-1)]
			yield (train_data, train_labels)

	def get_train_data(self):
		"""Get the entire training data.

		Returns:
			training data as dx1xb numpy array
		"""
		return self.data['train_data']

	def get_train_labels(self):
		"""Get the entire training labels.

		Returns:
			training labels as 1x1xb numpy array
		"""
		return self.data['train_labels']

	def get_test_data(self):
		"""Get the entire test data.

		Returns:
			test data as dx1xb numpy array
		"""
		return self.data['test_data']

	def get_test_labels(self):
		"""Get the entire test labels.

		Returns:
			test labels as 1x1xb numpy array
		"""
		return self.data['test_labels']

	def get_data_dim(self):
		"""Get the number of features of the training data.

		Returns:
			number of training features
		"""
		return self.data['train_data'].shape[0]

	def get_num_train(self):
		"""Get the number of training examples.

		Returns:
			number of training examples
		"""
		return self.data['train_data'].shape[2]

	def get_num_test(self):
		"""Get the number of test examples.

		Returns:
			number of test examples
		"""
		return self.data['test_data'].shape[2]
