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

	def backward(self, x):
		pass

	def __str__(self):
		string = "Sequential Network: \n"
		for index, layer in enumerate(self.graph):
			string += "\t[{0}] {1}\n".format(index, layer)
		return string
