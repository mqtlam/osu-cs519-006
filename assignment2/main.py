import numpy as np
from layers.relu import ReluLayer
from layers.sigmoid import SigmoidLayer
from layers.linear import LinearLayer

from scipy.special import expit

# hyperparameters
num_hidden_units = 10
learning_rate = 0.01
mini_batch_size = 256
momentum = 0.1

# variables
input_dim = 10

# network
linear1 = LinearLayer((num_hidden_units, input_dim))
relu = ReluLayer()
linear2 = LinearLayer((num_hidden_units,))
sigmoid = SigmoidLayer()

# test
x = np.random.rand(input_dim)

s1 = linear1.forward(x)
z1 = relu.forward(s1)
s2 = linear2.forward(z1)
z2 = sigmoid.forward(s2)
