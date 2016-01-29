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
		self.__debug_print__("[forward] Running forward pass...\n")
		self.__debug_print__("[forward] Input={0}\n".format(x))

		z = x
		for index, layer in enumerate(self.graph):
			self.__debug_print__("[forward] [{0}] {1}".format(index, layer))
			self.__debug_print__("[forward] Input=\n\t\t{0}".format(z))

			z = layer.forward(z)

			self.__debug_print__("[forward] Output=\n\t\t{0}\n".format(z))

		self.__debug_print__("[forward] Output={0}\n".format(z))
		self.__debug_print__("[forward] Done with forward pass.")

		return z

	def backward(self, x, grad):
		self.__debug_print__("[backward] Running backward pass...\n")
		self.__debug_print__("[backward] Input x={0}\n".format(x))
		self.__debug_print__("[backward] Input grad={0}\n".format(grad))

		log_g = np.log(grad)
		for index, layer in enumerate(reversed(self.graph)):
			g = np.exp(log_g)

			self.__debug_print__("[backward] [{0}] {1}".format(index, layer))
			self.__debug_print__("[backward] Input=\n\t\t{0}".format(g))

			result = layer.backward(x, g)

			self.__debug_print__("[backward] Output=\n\t\t{0}\n".format(result))

			log_g = log_g + np.log(result)

		result = np.exp(log_g)
		self.__debug_print__("[backward] Output={0}\n".format(result))
		self.__debug_print__("[backward] Done with backward pass.")

		return result

	def __str__(self):
		string = "Sequential Network: "
		for index, layer in enumerate(self.graph):
			string += "\n\t[{0}] {1}".format(index, layer)
		return string

	def __debug_print__(self, string):
		if self.debug:
			print(string)
