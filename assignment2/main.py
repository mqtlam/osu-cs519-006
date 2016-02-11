import numpy as np
from layers.relu import ReluLayer
from layers.soft_max import SoftMaxLayer
from layers.linear import LinearLayer
from network.sequential import Sequential
from util.dataset import CifarDataset
from loss.cross_entropy import CrossEntropyLoss
from solver.easy_solver import EasySolver
from solver.momentum_solver import MomentumSolver

# set random seed
np.random.seed(13141)

# debug mode
debug_mode = False

# load data
DATASET_PATH = 'data/cifar_2class_py2.p'
data = CifarDataset()
data.load(DATASET_PATH)
num_training = data.get_num_train()
num_test = data.get_num_test()
input_dim = data.get_data_dim()

# hyperparameters
num_hidden_units = 10
learning_rate = 0.001
momentum_mu = 0.6
mini_batch_size = 256
num_epoch = 25 if not debug_mode else 1

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
for epoch in range(num_epoch):
	print("Training epoch {0}...".format(epoch))
	# training
	for iter, batch in enumerate(data.get_train_batches(mini_batch_size)):
		if iter > 1 and debug_mode:
			break

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
		l_avg = 1./batch_size*l.sum(2)
		print("\tloss: {0}".format(l_avg))
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

	# evaluation
	target = data.get_test_labels()
	x = data.get_test_data()
	batch_size = x.shape[2]

	z = net.forward(x)
	l = loss.forward(z, target)
	l_avg = 1./batch_size*l.sum(2)
	print("Evaluation: {0}".format(l_avg))
