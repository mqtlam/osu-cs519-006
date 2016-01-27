try:
	import cPickle as pickle
except:
	import pickle

class DatasetLoader:
	@staticmethod
	def load_cifar(dataset_path):
		with open(dataset_path, 'rb') as f:
			data = pickle.load(f)
		return data

	@staticmethod
	def save_cifar(data, dataset_path, protcol=3):
		with open(dataset_path, 'wb') as f:
			pickle.save(data, f, protocol)
