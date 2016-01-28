try:
	import cPickle as pickle
except:
	import pickle

class Dataset:
	def __init__(self):
		pass

	def load(self, dataset_path):
		pass

	def save(self, dataset_path):
		pass

class CifarDataset(Dataset):
	def __init__(self):
		self.data = dict

	def load(self, dataset_path):
		with open(dataset_path, 'rb') as f:
			self.data = pickle.load(f)

	def save(self, dataset_path, protocol=3):
		with open(dataset_path, 'wb') as f:
			pickle.dump(self.data, f, protocol)

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
