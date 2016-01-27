class Sequential:
	def __init__(self):
		self.graph = []

	def add(self, layer):
		self.graph.append(layer)

	def remove(self, index=None):
		if index is None:
			self.graph.pop()
		else:
			self.graph.pop(index)

	def forward(self, x):
		z = x
		for layer in self.graph:
			z = layer.forward(z)
		return z

	def backward(self, x):
		pass
