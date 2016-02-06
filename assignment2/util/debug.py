class Debug:
	def __init__(self, debug):
		self.debug = debug

	def disp(self, string):
		if self.debug:
			print(string)
