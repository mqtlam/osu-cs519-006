import numpy as np

from util.debug import Debug

class Sequential:
	"""Sequential is a type of network container that organizes
	modules in a sequential network.

	Assumes working with consistent data format
	for all inputs and outputs:
		mxnxb numpy array
			b refers to batch
			m,n are arbitrary
	"""

	def __init__(self, **kwargs):
		"""Initialization.

		Args:
			debug (kwargs): boolean indicating debug mode
		"""
		self.debug = Debug("debug" in kwargs and kwargs["debug"])
		self.graph = []

	def size(self):
		"""Returns the number of layers in the network.

		Returns:
			number of layers
		"""
		return len(self.graph)

	def get(self, index):
		"""Get the layer at position index.

		Args:
			index: index to query

		Returns:
			layer object
		"""
		return self.graph[index]

	def add(self, layer):
		"""Add a layer to the end of the network.

		Args:
			layer: layer object
		"""
		self.graph.append(layer)

		self.debug.disp("Added layer: [{0}] {1}".format(len(self.graph)-1, layer))

	def remove(self, index=None):
		"""Remove a layer at the end of the network or at the specified index.

		Args:
			index: index to remove layer, otherwise remove last layer
		"""
		if index is None:
			self.graph.pop()
		else:
			self.graph.pop(index)

		self.debug.disp("Removed layer from end: {0}".format(layer))

	def insert(self, layer, index):
		"""Insert a layer at the specified index.

		Args:
			layer: layer object
			index: index to insert
		"""
		self.graph.insert(layer, index)

		self.debug.disp("Inserted layer: [{0}] {1}".format(index, layer))

	def forward(self, x):
		"""Performs a forward pass.

		Calls forward() on all layers from in a forward sequence.
		This is basically inference.

		Args:
			x: input data

		Returns:
			result of forward pass (last layer)
		"""
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
		"""Performs a backward pass.

		Calls backward() on all layers in a backward sequence.

		Does not update weights!
		This is basically computing gradients in advance for backpropagation.

		Also, forward() MUST BE CALLED BEFORE backward()!

		Args:
			x: input dataset
			grad: output gradient (computed from loss function)

		Returns:
			gradient w.r.t. to input
		"""
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
		"""Update the parameters of the network.

		Backpropagation is basically backward() followed by updateParams().
		backward() MUST BE CALLED BEFORE updateParams()!

		Args:
			solver: solver object for updating weights
		"""
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
