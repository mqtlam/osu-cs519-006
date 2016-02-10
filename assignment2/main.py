import numpy as np
from layers.relu import ReluLayer
from layers.soft_max import SoftMaxLayer
from layers.linear import LinearLayer
from network.sequential import Sequential
from util.dataset import CifarDataset
from loss.cross_entropy import CrossEntropyLoss
from solver.momentum_solver import MomentumSolver

# set random seed
np.random.seed(13141)

# debug mode
debug_mode = True

# load data
DATASET_PATH = 'data/cifar_2class.protocol2'
# DATASET_PATH = 'data/cifar_2class'
data = CifarDataset()
data.load(DATASET_PATH)
num_training = data.get_num_train()
num_test = data.get_num_test()
input_dim = data.get_data_dim()

# hyperparameters
num_hidden_units = 10
learning_rate = 0.01
mini_batch_size = 256
momentum = 0.1
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
solver = MomentumSolver(lr=learning_rate, mu=0.6)

def test_propagations():
	# forward propagation test
	train_image_index = 0
	target = data.get_train_labels()[train_image_index, 0]
	x = data.get_train_data()[0]
	z = net.forward(x)
	l = loss.forward(z, target)

	print("input={0}".format(x))
	print("output={0}".format(z))
	print("loss={0}".format(l))

	# backward propagation test
	gradients = loss.backward(z, target)
	grad_x = net.backward(x, gradients)

	print("input gradient={0}".format(grad_x))

	# update params test
	net.updateParams(solver)

# training loop
for epoch in range(num_epoch):
	print("Training epoch {0}...".format(epoch))
	# training
	for batch in data.get_train_batches(mini_batch_size):
		# get batch
		(x, target) = batch
		debug_index = -1
		x = x[debug_index]
		target = target[debug_index][0]

		# forward
		z = net.forward(x)
		if debug_mode:
			print("\toutput: {0}".format(z))
			print("\toutput shape: {0}".format(z.shape))

		# loss
		l = loss.forward(z, target)
		print("\tloss: {0}".format(l))
		if debug_mode:
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
	test_image_index = 0
	target = data.get_test_labels()[test_image_index, 0]
	x = data.get_test_data()[0]

	z = net.forward(x)
	l = loss.forward(z, target)
	print("Evaluation: {0}".format(l))
