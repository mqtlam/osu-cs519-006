# Assignment #2

This is assignment #2 for the deep learning class CS 519-006 winter 2016.

## Overview

This assignment implements a one hidden layer fully connected neural network in python (from scratch).

## Dependencies
- Python 2
- pip (to install packages)
- virtualenv (for setting up environments easily)

### Libraries
- numpy 
- scipy 
- sklearn

See requirements.txt for complete list.

## Directory Structure
```
	data/				where to store dataset
	layers/				Python package containing layer modules
		core.py				abstract layer
		linear.py			linear layer (W*x+b)
		relu.py				RELU activation layer
		sigmoid.py			sigmoid layer
		soft_max.py			soft max layer
	loss/				Python package containing loss modules
		core.py				abstract loss
		cross_entropy.py		cross entropy loss layer
	network/			network container types
		sequential.py			sequential network container
	solver/				Python package containing training solvers
		core.py				abstract solver
		easy_solver.py			fixed learning rate
		momentum_solver.py		weight updates via momentum
	util/				Python utilities for learning and inference
		benchmark.py			benchmarking tools
		data_preprocessing.py		image data preprocessing
		dataset.py			loading and handling datasets
		debug.py			utilities useful for debugging
		hyperparameter.py		hyperparameter tuning
		monitor.py			training monitoring
	bootstrap.sh			Run first to set up environment
	main.py				main script
	module.py			abstract class for module
```

## Quick Start
1. Download the cifar dataset (Python 2 version!) to this folder.
2. Set up the python virtual environment and extract dataset using `./bootstrap.sh cifar-2class-py2.zip` (this might take a while)
3. Activate the virtual environment (`source venv/bin/activate`)
4. Run `main.py` for training and test
