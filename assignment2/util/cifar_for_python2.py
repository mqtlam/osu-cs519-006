#!/usr/bin/env python
from data_loader import DatasetLoader

def main():
	"""Make sure this script is running in Python 3!"""

	load_dataset_path = "../data/cifar_2class"
	new_dataset_path = "../data/cifar_2class.protocol2"

	print("Loading protocol version 3...")
	data = DatasetLoader.load_cifar(load_dataset_path)
	print("Saving to protocol version 2...")
	DatasetLoader.save_cifar(data, new_dataset_path, 2)
	print("Done.")
main()
