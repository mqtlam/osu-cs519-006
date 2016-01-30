class Layer:
	def __init__(self):
		self.output = None

	def forward(self, x):
		pass

	def backward(self, x, grad):
		pass

	def __str__(self):
		string =  "Layer (abstract)"
		return string
