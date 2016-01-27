import numpy as np
from layers.relu import ReluLayer
from layers.sigmoid import SigmoidLayer
from layers.linear import LinearLayer
from network.sequential import Sequential

# hyperparameters
num_hidden_units = 10
learning_rate = 0.01
mini_batch_size = 256
momentum = 0.1
num_epoch = 25

# variables
input_dim = 20

# network
net = Sequential()
net.add( LinearLayer(input_dim, num_hidden_units) )
net.add( ReluLayer() )
net.add( LinearLayer(num_hidden_units, 1) )
net.add( SigmoidLayer() )

print net

# test
x = np.random.rand(input_dim)
z = net.forward(x)
