# Assignment #2

This is assignment #2 for the deep learning class CS 519-006 winter 2016.

## Overview

This assignment implements a one hidden layer fully connected neural network in python from "scratch." No deep learning libraries were used.

## Running
1. Download the cifar dataset (Python 2 version!) to this folder.
2. Set up the python virtual environment and extract dataset using `./bootstrap.sh cifar-2class-py2.zip` (this might take a while)
3. Activate the virtual environment (`source venv/bin/activate`)
4. Run `main.py` for training and test
5. It's also possible to run multiple experiments in parallel on a HPC cluster. Run `hyperparameter.py` to run multiple experiments on the cluster for tuning the hyperparameters.

### Requirements
- Python 2
- pip (to install packages)
- virtualenv (for setting up environments easily)

### Libraries
- argparse
- numpy
- scipy
- ipython
- sklearn
- matplotlib

These are automatically installed by running `bootstrap.sh`.

See requirements.txt for complete list.

## Directory Structure
```
	data/ - where to store dataset
	layers/ - Python package containing layer modules
		core.py - abstract layer
		linear.py - linear layer (W*x+b)
		relu.py - RELU activation layer
		sigmoid.py - sigmoid layer
		soft_max.py - soft max layer
	loss/ - Python package containing loss modules
		core.py - abstract loss
		cross_entropy.py - cross entropy loss layer
	network/ - network container types
		sequential.py - sequential network container
	solver/ - Python package containing training solvers
		core.py - abstract solver
		easy_solver.py - fixed learning rate
		momentum_solver.py - weight updates via momentum
	util/ - Python utilities for learning and inference
		benchmark.py - benchmarking tools
		data_preprocessing.py - image data preprocessing
		dataset.py - loading and handling datasets
		debug.py - utilities useful for debugging
		hyperparameter.py - hyperparameter tuning
		metrics.py - metrics for evaluation
		monitor.py - training monitoring
		plot.py - plot loss graphs and more
	analyze_results.py - analyze results after running hyperparameter.py
	bootstrap.sh - run first to set up environment
	hyperparameter.sh - run multiple experiments for tuning hyperparameters
	main.py - main script
	module.py - abstract class for module
```
