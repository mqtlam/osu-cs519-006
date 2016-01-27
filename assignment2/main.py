import numpy as np
from layers.relu import ReluLayer
from layers.sigmoid import SigmoidLayer
from layers.linear import LinearLayer
from network.sequential import Sequential
from util.data_loader import DatasetLoader
import pickle

# load data
# DATASET_PATH = 'data/cifar_2class.protocol2'
DATASET_PATH = 'data/cifar_2class'
data = DatasetLoader.load_cifar(DATASET_PATH)
num_training = data['train_data'].shape[0]
num_test = data['test_data'].shape[0]
input_dim = data['train_data'].shape[1]

# hyperparameters
num_hidden_units = 10
learning_rate = 0.01
mini_batch_size = 256
momentum = 0.1
num_epoch = 25

# network
net = Sequential()
net.add( LinearLayer(input_dim, num_hidden_units) )
net.add( ReluLayer() )
net.add( LinearLayer(num_hidden_units, 1) )
net.add( SigmoidLayer() )

print(net)

# forward propagation test
x = np.random.rand(input_dim)
z = net.forward(x)

print(x)
print(z)

# backward propagation test
grad = np.random.rand(1)
grad_x = net.backward(x, grad)

print(grad_x)
