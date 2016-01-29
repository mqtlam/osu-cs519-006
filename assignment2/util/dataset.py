try:
	import cPickle as pickle
except:
	import pickle

from util.data_preprocessing import DataPreprocessing

class Dataset:
	def __init__(self):
		pass

	def load(self, dataset_path):
		pass

	def save(self, dataset_path):
		pass

class CifarDataset(Dataset):
	def __init__(self):
		self.original_data = dict
		self.data = dict

	def load(self, dataset_path):
		with open(dataset_path, 'rb') as f:
			self.original_data = pickle.load(f)
		self.data = self.preprocess(self.original_data)

	def save(self, dataset_path, protocol=3):
		with open(dataset_path, 'wb') as f:
			pickle.dump(self.original_data, f, protocol)

	def preprocess(self, data):
		meanval = DataPreprocessing.compute_mean(data['train_data'])
		stdval = DataPreprocessing.compute_std(data['train_data'])
		data['train_data'] = DataPreprocessing.normalize_data(data['train_data'], mean=meanval, std=stdval)
		data['test_data'] = DataPreprocessing.normalize_data(data['test_data'], mean=meanval, std=stdval)
		return data

	def get_train_data(self):
		return self.data['train_data']

	def get_train_labels(self):
		return self.data['train_labels']

	def get_test_data(self):
		return self.data['test_data']

	def get_test_labels(self):
		return self.data['test_labels']

	def get_data_dim(self):
		return self.data['train_data'].shape[1]

	def get_num_train(self):
		return self.data['train_data'].shape[0]

	def get_num_test(self):
		return self.data['test_data'].shape[0]
