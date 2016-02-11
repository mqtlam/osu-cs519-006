import numpy as np
import argparse

from layers.relu import ReluLayer
from layers.soft_max import SoftMaxLayer
from layers.linear import LinearLayer
from network.sequential import Sequential
from util.dataset import CifarDataset
from loss.cross_entropy import CrossEntropyLoss
from solver.easy_solver import EasySolver
from solver.momentum_solver import MomentumSolver

from util.monitor import Monitor
from util.benchmark import Timer
timer = Timer()

def main():
	# set random seed
	np.random.seed(13141)

	# debug mode
	debug_mode = False

	# parse arguments
	parser = argparse.ArgumentParser(description='Train and test neural network on cifar dataset.')
	parser.add_argument('experiment_name', help='used for outputting log files')
	parser.add_argument('--num_hidden_units', type=int, help='number of hidden units')
	parser.add_argument('--learning_rate', type=float, help='learning rate for solver')
	parser.add_argument('--momentum_mu', type=float, help='mu for momentum solver')
	parser.add_argument('--mini_batch_size', type=int, help='mini batch size')
	parser.add_argument('--num_epoch', type=int, help='number of epochs')
	args = parser.parse_args()
	import sys
	sys.exit()

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
	num_hidden_units = 50 if args.num_hidden_units is None else args.num_hidden_units
	learning_rate = 0.001 if args.learning_rate is None else args.learning_rate
	momentum_mu = 0.7 if args.momentum_mu is None else args.momentum_mu
	mini_batch_size = 256 if args.mini_batch_size is None else args.mini_batch_size
	num_epoch = (50 if not debug_mode else 1) if args.num_epoch is None else args.num_epoch

	# network
	net = Sequential(debug=debug_mode)
	net.add( LinearLayer(input_dim, num_hidden_units) )
	net.add( ReluLayer() )
	net.add( LinearLayer(num_hidden_units, 2) )
	net.add( SoftMaxLayer() )

	print("{0}\n".format(net))

	# loss
	loss = CrossEntropyLoss()

	print("Loss function: {0}\n".format(loss))

	# solver
	# solver = EasySolver(learning_rate)
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
			if debug_mode:
				print("\toutput: {0}".format(z))
				print("\toutput shape: {0}".format(z.shape))

			# loss
			l = loss.forward(z, target)
			l_avg = 1./batch_size*l.sum(2)[0,0]
			if debug_mode:
				print("\tloss: {0}".format(l))
				print("\tloss shape: {0}".format(l.shape))

			# backward loss
			gradients = loss.backward(z, target)
			if debug_mode:
				print("\tgradients: {0}".format(gradients))
				print("\tgradients shape: {0}".format(gradients.shape))

			# backward
			grad_x = net.backward(x, gradients)
			if debug_mode:
				print("\tgrad_x: {0}".format(grad_x))
				print("\tgrad_x: {0}".format(grad_x.shape))

			# update parameters
			net.updateParams(solver)

			# timing
			elapsed = timer.getElapsed("iter")
			print("\tloss: {0}\telapsed: {1}".format(l_avg, elapsed))

			# logging
			monitor.recordIteration(cum_iter, l_avg, elapsed)
			cum_iter += 1

		# evaluation on test set
		target = data.get_test_labels()
		x = data.get_test_data()
		batch_size = x.shape[2]

		z = net.forward(x)
		l = loss.forward(z, target)
		l_test_avg = 1./batch_size*l.sum(2)[0,0]

		# evaluation on training set
		target = data.get_train_labels()
		x = data.get_train_data()
		batch_size = x.shape[2]

		z = net.forward(x)
		l = loss.forward(z, target)
		l_train_avg = 1./batch_size*l.sum(2)[0,0]

		print("Evaluation: test: {0}\ttrain: {1}".format(l_test_avg, l_train_avg))

		# timing
		elapsed = timer.getElapsed("epoch")
		print("Finished epoch {1} in {0:2f}s.".format(elapsed, epoch))

		# logging
		monitor.recordEpoch(epoch, l_train_avg, l_test_avg, elapsed)

	monitor.finishSession()

if __name__ == '__main__':
	main()
