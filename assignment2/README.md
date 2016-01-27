# Assignment #2

This is assignment #2 for the deep learning class CS 519-006 winter 2016.

## Overview

This assignment implements a one hidden layer fully connected neural network in python (from scratch).

## Dependencies
- Python 3
- numpy 
- scipy 

See requirements.txt for complete list.

## Directory Structure
```
	data/				where to store dataset
	layers/				Python package containing layer modules
		layer.py			abstract layer
		linear.py			linear layer (W*x+b)
		relu.py				RELU activation layer
		sigmoid.py			sigmoid layer
		soft_max.py			soft max layer
		cross_entropy.py		cross entropy loss layer
	network/			network container types
		sequential.py			sequential network container
	test/				simple unit tests
	util/				Python utilities for learning and inference
		benchmark.py			benchmarking tools
		data_loader.py			load data
		hyperparameter.py		hyperparameter tuning
		monitor.py			training monitoring
	bootstrap.sh			Run first to set up environment
	main.py				main script
	solver.py			momentum solver
```

## Quick Start
Set up the python virtual environment and extract dataset using `./bootstrap.sh cifar-2class-py.zip`

Run `main.py` for training and test
