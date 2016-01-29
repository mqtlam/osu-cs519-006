import numpy as np

class Sequential:
	def __init__(self, **kwargs):
		self.debug = "debug" in kwargs and kwargs["debug"]
		self.graph = []

	def size(self):
		return len(self.graph)

	def get(self, index):
		return self.graph[index]

	def add(self, layer):
		self.graph.append(layer)

		self.__debug_print__("Added layer: [{0}] {1}".format(len(self.graph)-1, layer))

	def remove(self, index=None):
		if index is None:
			self.graph.pop()
		else:
			self.graph.pop(index)

		self.__debug_print__("Removed layer from end: {0}".format(layer))

	def insert(self, layer, index):
		self.graph.insert(layer, index)

		self.__debug_print__("Inserted layer: [{0}] {1}".format(index, layer))

	def forward(self, x):
		self.__debug_print__("Running forward pass...\n")
		self.__debug_print__("Input={0}\n".format(x))

		z = x
		for index, layer in enumerate(self.graph):
			self.__debug_print__("[{0}] {1}".format(index, layer))
			self.__debug_print__("\tInput=\n\t\t{0}".format(z))

			z = layer.forward(z)

			self.__debug_print__("\tOutput=\n\t\t{0}\n".format(z))

		self.__debug_print__("Output={0}\n".format(z))
		self.__debug_print__("Done with forward pass.")

		return z

	def backward(self, x, grad):
		self.__debug_print__("Running backward pass...\n")
		self.__debug_print__("Input={0}\n".format(x))
		self.__debug_print__("Input={1}\n".format(grad))

		log_g = np.log(grad)
		for index, layer in enumerate(reversed(self.graph)):
			g = np.exp(log_g)
			result = layer.backward(x, g)
			log_g = log_g + np.log(result)
		return np.exp(log_g)

	def __str__(self):
		string = "Sequential Network: "
		for index, layer in enumerate(self.graph):
			string += "\n\t[{0}] {1}".format(index, layer)
		return string

	def __debug_print__(self, string):
		if self.debug:
			print(string)
