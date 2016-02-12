class Debug:
	"""Contains useful debugging utilities.
	"""
	def __init__(self, debug):
		"""Initialization.

		Args:
			debug: boolean for debug mode
		"""
		self.debug = debug

	def disp(self, string):
		"""Print a string or not depending on debug mode.

		Args:
			string: string to print
		"""
		if self.debug:
			print(string)
