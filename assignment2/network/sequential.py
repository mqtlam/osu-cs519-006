import numpy as np

from util.debug import Debug

class Sequential:
	def __init__(self, **kwargs):
		self.debug = Debug("debug" in kwargs and kwargs["debug"])
		self.graph = []

	def size(self):
		return len(self.graph)

	def get(self, index):
		return self.graph[index]

	def add(self, layer):
		self.graph.append(layer)

		self.debug.disp("Added layer: [{0}] {1}".format(len(self.graph)-1, layer))

	def remove(self, index=None):
		if index is None:
			self.graph.pop()
		else:
			self.graph.pop(index)

		self.debug.disp("Removed layer from end: {0}".format(layer))

	def insert(self, layer, index):
		self.graph.insert(layer, index)

		self.debug.disp("Inserted layer: [{0}] {1}".format(index, layer))

	def forward(self, x):
		self.debug.disp("[forward] Running forward pass...\n")
		self.debug.disp("[forward] Initial Input={0}\n".format(x))
		self.debug.disp("[forward] Initial Input Shape={0}\n".format(x.shape))

		z = x
		for index, layer in enumerate(self.graph):
			self.debug.disp("[forward] [{0}] {1}".format(index, layer))
			self.debug.disp("[forward] Input=\n\t\t{0}".format(z))
			self.debug.disp("[forward] Input Shape=\n\t\t{0}".format(z.shape))

			z = layer.forward(z)

			self.debug.disp("[forward] Output=\n\t\t{0}\n".format(z))
			self.debug.disp("[forward] Output Shape=\n\t\t{0}\n".format(z.shape))

		self.debug.disp("[forward] Final Output={0}\n".format(z))
		self.debug.disp("[forward] Final Output Shape={0}\n".format(z.shape))
		self.debug.disp("[forward] Done with forward pass.")

		return z

	def backward(self, x, grad):
		self.debug.disp("[backward] Running backward pass...\n")
		self.debug.disp("[backward] Initial Input x={0}\n".format(x))
		self.debug.disp("[backward] Initial Input x shape={0}\n".format(x.shape))
		self.debug.disp("[backward] Initial Input grad={0}\n".format(grad))
		self.debug.disp("[backward] Initial Input grad shape={0}\n".format(grad.shape))

		g = grad
		for index, layer in enumerate(reversed(self.graph)):
			self.debug.disp("[backward] [{0}] {1}".format(index, layer))
			self.debug.disp("[backward] Input=\n\t\t{0}".format(g))
			self.debug.disp("[backward] Input Shape=\n\t\t{0}".format(g.shape))

			g = layer.backward(g)

			self.debug.disp("[backward] Output=\n\t\t{0}\n".format(g))
			self.debug.disp("[backward] Output Shape=\n\t\t{0}\n".format(g.shape))

		self.debug.disp("[backward] Final Output={0}\n".format(g))
		self.debug.disp("[backward] Final Output Shape={0}\n".format(g.shape))
		self.debug.disp("[backward] Done with backward pass.")

		return g

	def updateParams(self, solver):
		self.debug.disp("[updateParams] Updating network params...\n")

		for index, layer in enumerate(reversed(self.graph)):
			self.debug.disp("[updateParams] [{0}] {1}".format(index, layer))
			layer.updateParams(solver)

		self.debug.disp("\n[updateParams] Done updating params.")

	def __str__(self):
		string = "Sequential Network: "
		for index, layer in enumerate(self.graph):
			string += "\n\t[{0}] {1}".format(index, layer)
		return string
