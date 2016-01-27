import math

class Sequential:
	def __init__(self):
		self.graph = []

	def size(self):
		return len(self.graph)

	def get(self, index):
		return self.graph[index]

	def add(self, layer):
		self.graph.append(layer)

	def remove(self, index=None):
		if index is None:
			self.graph.pop()
		else:
			self.graph.pop(index)

	def insert(self, index):
		self.graph.insert(index)

	def forward(self, x):
		z = x
		for layer in self.graph:
			z = layer.forward(z)
		return z

	def backward(self, x, grad):
		log_g = math.log(grad)
		for layer in reversed(self.graph):
			g = math.exp(log_g)
			result = layer.backward(x, g)
			log_g = log_g + math.log(result)
		return math.exp(log_g)

	def __str__(self):
		string = "Sequential Network: \n"
		for index, layer in enumerate(self.graph):
			string += "\t[{0}] {1}\n".format(index, layer)
		return string
