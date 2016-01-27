import numpy as np

class Sequential:
	def __init__(self):
		self.graph = []

	def size(self):
		return len(self.graph)

	def get(self, index):
		return self.graph[index]

	def add(self, layer):
		self.graph.append(layer)
		# print("Added layer: [{0}] {1}".format(len(self.graph)-1, layer))

	def remove(self, index=None):
		if index is None:
			self.graph.pop()
		else:
			self.graph.pop(index)
		# print("Removed layer from end: {0}".format(layer))

	def insert(self, layer, index):
		self.graph.insert(layer, index)
		# print("Inserted layer: [{0}] {1}".format(index, layer))

	def forward(self, x):
		z = x
		for layer in self.graph:
			z = layer.forward(z)
		return z

	def backward(self, x, grad):
		log_g = np.log(grad)
		for layer in reversed(self.graph):
			g = np.exp(log_g)
			result = layer.backward(x, g)
			log_g = log_g + np.log(result)
		return np.exp(log_g)

	def __str__(self):
		string = "Sequential Network: "
		for index, layer in enumerate(self.graph):
			string += "\n\t[{0}] {1}".format(index, layer)
		return string
