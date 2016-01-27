import numpy as np
from scipy.special import expit

# hyperparameters
num_hidden_units = 10
learning_rate = 0.01
mini_batch_size = 256
momentum = 0.1

mu = 0
sigma = 1

# useful functions
def relu(x):
	return x * (x > 0)

def relu_grad(x):
	return 1 * (x > 0) + 0 * (x < 0) + np.random.uniform(0, 1, x.shape) * (x == 0)

def sigmoid(x):
	return expit(x)

def sigmoid_grad(x):
	return sigmoid(x)*(1-sigmoid(x))

def linear(W, x):
	return np.dot(W, x)

def linear_grad(W, x):
	pass

# variables
input_dim = 10

# construct simple network
W1 = np.random.normal(mu, sigma, (num_hidden_units, input_dim))
w2 = np.random.normal(mu, sigma, num_hidden_units)

# test
x = np.random.rand(input_dim)

s1 = linear(W1, x)
z1 = relu(s1)

s2 = linear(w2, z1)
z2 = sigmoid(s2)
