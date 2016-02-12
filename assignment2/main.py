"""Main script for training and evaluating a neural network.
Results will be stored in the logs folder.

Usage: python main.py experiment_name
	Can also set hyperparameters:
		--num_hidden_units
		--learning_rate
		--momentum_mu
		--mini_batch_size
		--num_epoch

Run 'python main.py -h for help'
"""

import numpy as np
import argparse

from layers.relu import ReluLayer
from layers.soft_max import SoftMaxLayer
from layers.linear import LinearLayer
from network.sequential import Sequential
from util.dataset import CifarDataset
from loss.cross_entropy import CrossEntropyLoss
from solver.momentum_solver import MomentumSolver

from util.debug import Debug
from util.metrics import ErrorRate
from util.metrics import Objective
from util.monitor import Monitor
from util.benchmark import Timer
timer = Timer()

def main():
	# set random seed
	np.random.seed(13141)

	# debug mode
	debug_mode = False
	debug = Debug(debug_mode)

	# parse arguments
	parser = argparse.ArgumentParser(description='Train and test neural network on cifar dataset.')
	parser.add_argument('experiment_name', help='used for outputting log files')
	parser.add_argument('--num_hidden_units', type=int, help='number of hidden units')
	parser.add_argument('--learning_rate', type=float, help='learning rate for solver')
	parser.add_argument('--momentum_mu', type=float, help='mu for momentum solver')
	parser.add_argument('--mini_batch_size', type=int, help='mini batch size')
	parser.add_argument('--num_epoch', type=int, help='number of epochs')
	args = parser.parse_args()

	# experiment name
	experiment_name = args.experiment_name
	iter_log_file = "logs/{0}_iter_log.txt".format(experiment_name)
	epoch_log_file = "logs/{0}_epoch_log.txt".format(experiment_name)

	# load data
	print("Loading dataset...")
	timer.begin("dataset")
	DATASET_PATH = 'cifar-2class-py2/cifar_2class_py2.p'
	data = CifarDataset()
	data.load(DATASET_PATH)
	print("Loaded dataset in {0:2f}s.".format(timer.getElapsed("dataset")))

	# get data stats
	num_training = data.get_num_train()
	num_test = data.get_num_test()
	input_dim = data.get_data_dim()

	# hyperparameters
	num_hidden_units = 100 if args.num_hidden_units is None else args.num_hidden_units
	learning_rate = 0.01 if args.learning_rate is None else args.learning_rate
	momentum_mu = 0.6 if args.momentum_mu is None else args.momentum_mu
	mini_batch_size = 256 if args.mini_batch_size is None else args.mini_batch_size
	num_epoch = (1000 if not debug_mode else 1) if args.num_epoch is None else args.num_epoch

	# print hyperparameters
	print("num_hidden_units: {0}".format(num_hidden_units))
	print("learning_rate: {0}".format(learning_rate))
	print("momentum_mu: {0}".format(momentum_mu))
	print("mini_batch_size: {0}".format(mini_batch_size))
	print("num_epoch: {0}".format(num_epoch))

	# network
	net = Sequential(debug=debug_mode)
	net.add( LinearLayer(input_dim, num_hidden_units) )
	net.add( ReluLayer() )
	net.add( LinearLayer(num_hidden_units, 2) )
	net.add( SoftMaxLayer() )

	print("{0}\n".format(net))

	# loss
	loss = CrossEntropyLoss()

	# error metrics
	training_objective = Objective(loss)
	test_objective = Objective(loss)
	errorRate = ErrorRate()

	print("Loss function: {0}\n".format(loss))

	# solver
	solver = MomentumSolver(lr=learning_rate, mu=momentum_mu)

	# training loop
	monitor = Monitor()
	monitor.createSession(iter_log_file, epoch_log_file)
	cum_iter = 0
	for epoch in range(num_epoch):
		print("Training epoch {0}...".format(epoch))
		timer.begin("epoch")
		# training
		for iter, batch in enumerate(data.get_train_batches(mini_batch_size)):
			if iter > 1 and debug_mode:
				break

			timer.begin("iter")

			# get batch
			(x, target) = batch
			batch_size = x.shape[2]

			# forward
			z = net.forward(x)
			Debug.disp("\toutput: {0}".format(z))
			Debug.disp("\toutput shape: {0}".format(z.shape))

			# loss
			if debug_mode:
				l = loss.forward(z, target)
				Debug.disp("\tloss: {0}".format(l))
				Debug.disp("\tloss shape: {0}".format(l.shape))

			# backward loss
			gradients = loss.backward(z, target)
			Debug.disp("\tgradients: {0}".format(gradients))
			Debug.disp("\tgradients shape: {0}".format(gradients.shape))

			# backward
			grad_x = net.backward(x, gradients)
			Debug.disp("\tgrad_x: {0}".format(grad_x))
			Debug.disp("\tgrad_x: {0}".format(grad_x.shape))

			# update parameters
			net.updateParams(solver)

			# metrics and timing
			loss_avg = training_objective.compute(z, target)
			elapsed = timer.getElapsed("iter")

			# logging
			print("\t[iter {0}]\tloss: {1}\telapsed: {2}".format(iter, loss_avg, elapsed))
			monitor.recordIteration(cum_iter, loss_avg, elapsed)

			cum_iter += 1

		# evaluation on test set
		target = data.get_test_labels()
		x = data.get_test_data()
		output = net.forward(x)
		loss_avg_test = test_objective.compute(output, target)
		error_rate_test = errorRate.compute(output, target)

		# evaluation on training set
		target = data.get_train_labels()
		x = data.get_train_data()
		output = net.forward(x)
		loss_avg_train = training_objective.compute(output, target)
		error_rate_train = errorRate.compute(output, target)

		# timing
		elapsed = timer.getElapsed("epoch")

		# logging
		print("End of epoch:\ttest objective: {0}\ttrain objective: {1}".format(loss_avg_test,
												 								loss_avg_train))
		print("\t\ttest error rate: {0}\ttrain error rate: {1}".format(error_rate_test,
												 					   error_rate_train))
		print("Finished epoch {1} in {0:2f}s.\n".format(elapsed, epoch))
		monitor.recordEpoch(epoch, loss_avg_train, loss_avg_test,
							error_rate_train, error_rate_test, elapsed)

	monitor.finishSession()

if __name__ == '__main__':
	main()
