try:
	import cPickle as pickle
except:
	import pickle

from util.data_preprocessing import DataPreprocessing
from sklearn.utils import shuffle

class Dataset:
	def __init__(self):
		pass

	def load(self, dataset_path):
		pass

	def save(self, dataset_path):
		pass

class CifarDataset(Dataset):
	def __init__(self):
		self.original_data = {}
		self.data = {}

	def load(self, dataset_path):
		with open(dataset_path, 'rb') as f:
			self.original_data = pickle.load(f)
		self.data = self.preprocess(self.original_data)
		self.data = self.shuffle_data(self.data)
		self.data = self.format_data(self.data)

	def save(self, dataset_path, protocol=3):
		with open(dataset_path, 'wb') as f:
			pickle.dump(self.original_data, f, protocol)

	def preprocess(self, data):
		meanval = DataPreprocessing.compute_mean(data['train_data'])
		stdval = DataPreprocessing.compute_std(data['train_data'])
		data['train_data'] = DataPreprocessing.normalize_data(data['train_data'], mean=meanval, std=stdval)
		data['test_data'] = DataPreprocessing.normalize_data(data['test_data'], mean=meanval, std=stdval)
		return data

	def shuffle_data(self, data):
		data['train_data'], data['train_labels'] = shuffle(data['train_data'], data['train_labels'])
		data['test_data'], data['test_labels'] = shuffle(data['test_data'], data['test_labels'])
		return data

	def format_data(self, data):
		data['train_data'] = data['train_data'][..., None]
		data['train_data'] = data['train_data'].transpose([1,2,0])
		data['train_labels'] = data['train_labels'].reshape((1,1,-1))
		data['test_data'] = data['test_data'][..., None]
		data['test_data'] = data['test_data'].transpose([1,2,0])
		data['test_labels'] = data['test_labels'].reshape((1,1,-1))
		return data

	def get_train_batches(self, batch_size):
		num_train = self.get_num_train()
		for i in xrange(0, num_train, batch_size):
			# get batch
			train_data = self.get_train_data()[:,:,i:min(i+batch_size, num_train-1)]
			train_labels = self.get_train_labels()[:,:,i:min(i+batch_size, num_train-1)]
			yield (train_data, train_labels)

	def get_train_data(self):
		return self.data['train_data']

	def get_train_labels(self):
		return self.data['train_labels']

	def get_test_data(self):
		return self.data['test_data']

	def get_test_labels(self):
		return self.data['test_labels']

	def get_data_dim(self):
		return self.data['train_data'].shape[0]

	def get_num_train(self):
		return self.data['train_data'].shape[2]

	def get_num_test(self):
		return self.data['test_data'].shape[2]
