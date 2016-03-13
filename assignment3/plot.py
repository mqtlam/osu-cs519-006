import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

class Plot:
	def __save_figure__(self, output_file):
		plt.savefig(output_file, dpi=72)
		plt.clf()

	def plot_loss(self, data, output_file):
		epochs = range(len(data['loss']))
		train_loss = data['loss']
		val_loss = data['val_loss']

		plt.plot(epochs, train_loss, color="blue", linewidth=2.5, linestyle="-", label="Training Loss")
		plt.plot(epochs, val_loss, color="red", linewidth=2.5, linestyle="-", label="Validation Loss")
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.title("Training/Validation Loss vs. Epoch")

		self.__save_figure__(output_file)

	def plot_error(self, data, output_file):
		epochs = range(len(data['acc']))
		train_error = data['acc']
		val_error = data['val_acc']
		for i in epochs:
			train_error[i] = 1-train_error[i]
			val_error[i] = 1-val_error[i]

		plt.plot(epochs, train_error, color="blue", linewidth=2.5, linestyle="-", label="Training Error")
		plt.plot(epochs, val_error, color="red", linewidth=2.5, linestyle="-", label="Validation Error")
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Epoch")
		plt.ylabel("Error")
		plt.title("Training/Validation Error vs. Epoch")

		self.__save_figure__(output_file)

def main():
	parser = argparse.ArgumentParser(description='plot')
	parser.add_argument('history', help='history json file')
	parser.add_argument('prefix', help='history json file')
	args = parser.parse_args()

	with open(args.history, 'r') as f:
		data = json.load(f)

	plot = Plot()
	plot.plot_loss(data, args.prefix + '_loss.png')
	plot.plot_error(data, args.prefix + '_error.png')

if __name__ == "__main__":
	main()
