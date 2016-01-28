import numpy as np
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

# network
net = Sequential()
net.add( LinearLayer(input_dim, num_hidden_units) )
net.add( ReluLayer() )
net.add( LinearLayer(num_hidden_units, 2) )
net.add( SoftMaxLayer() )

print("{0}\n".format(net))

# loss
loss = CrossEntropyLoss()

print("Loss function: {0}\n".format(loss))

# forward propagation test
target = 0
x = np.random.rand(input_dim)
z = net.forward(x)
l = loss.forward(z, target)

print("input={0}".format(x))
print("output={0}".format(z))
print("loss={0}".format(l))

# backward propagation test
gradients = loss.backward(z, target)
grad_x = net.backward(x, gradients)

print("input gradient={0}".format(grad_x))
