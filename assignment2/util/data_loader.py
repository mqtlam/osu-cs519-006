try
	import cPickle as pickle
except:
	import pickle

def unpickle_cifar():
	fo = open("cifar_2class", 'rb')
	dict = pickle.load(fo)
	fo.close()
	return dict
