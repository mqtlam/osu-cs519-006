import numpy as np
from tqdm import tqdm
from layers.relu import ReluLayer
from layers.soft_max import SoftMaxLayer
from layers.linear import LinearLayer
from network.sequential import Sequential
from util.dataset import CifarDataset
from loss.cross_entropy import CrossEntropyLoss

# set random seed
np.random.seed(13141)

# load data
DATASET_PATH = 'data/cifar_2class.protocol2'
# DATASET_PATH = 'data/cifar_2class'
data = CifarDataset()
data.load(DATASET_PATH)
num_training = data.get_num_train()
num_test = data.get_num_test()
input_dim = data.get_data_dim()

# hyperparameters
num_hidden_units = 20
learning_rate = 0.01
mini_batch_size = 256
momentum = 0.1
num_epoch = 25

num_batches = num_training // mini_batch_size

# network
net = Sequential(debug=True)
net.add( LinearLayer(input_dim, num_hidden_units) )
net.add( ReluLayer() )
net.add( LinearLayer(num_hidden_units, 2) )
net.add( SoftMaxLayer() )

print("{0}\n".format(net))

# loss
loss = CrossEntropyLoss()

print("Loss function: {0}\n".format(loss))

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

# # training loop
# for epoch in range(num_epoch):
# 	print("Training epoch {0}...".format(epoch))
# 	# training
# 	for batch in tqdm(range(num_batches)):
# 		# get batch
# 		train_image_index = 0
# 		target = data.get_train_labels()[train_image_index, 0]
# 		x = data.get_train_data()[0]
# 
# 		# forward
# 		z = net.forward(x)
# 		l = loss.forward(z, target)
# 
# 		# backward
# 		gradients = loss.backward(z, target)
# 		grad_x = net.backward(x, gradients)
# 
# 		# update
# 
# 	# evaluation
# 	test_image_index = 0
# 	target = data.get_test_labels()[test_image_index, 0]
# 	x = data.get_test_data()[0]
# 
# 	z = net.forward(x)
# 	l = loss.forward(z, target)
# 	print("Evaluation: {0}".format(l))
