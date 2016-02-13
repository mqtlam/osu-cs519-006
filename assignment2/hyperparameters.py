"""This script launches multiple hyperparameter experiments
in parallel. It uses the HPC cluster for launching jobs.
This is useful for experimenting different hyperparameters.
"""

import time
from subprocess import call
import os

# SET UP HYPERPARAMETERS TO EXPERIMENT BELOW
# The first element of the list will be used as a default value.
num_hidden_units_list = [50, 10, 25, 100, 200]
learning_rate_list = [0.01, 0.001, 0.1]
momentum_mu_list = [0, 0.2, 0.4, 0.6]
mini_batch_size_list = [256, 1, 32, 64, 128]
num_epoch = 250

# constants
SUBMIT_SCRIPT = "submit.csh"

def writeScript(name, argumentString):
    """Generate a submit script of the experiment for the cluster.

    Args:
        name: experiment name
        argumentString: arguments for 'python main.py'

    Returns:
        string containing contents of submit script
    """
    return """#!/bin/csh

# use current working directory for input and output
# default is to use the users home directory
#$ -cwd

# name this job
#$ -N {0}

# send stdout and stderror to this file
#$ -o jobs/{0}.txt
#$ -j y

# select queue
#$ -q eecs2

# activate
source venv/bin/activate.csh

# see where the job is being run
hostname

# see which python is used
which python

# print date and time
date

echo "Running name {0}"

# run script
echo "===================="
python main.py {1}
echo "===================="

# print date and time again
date

# deactivate script
deactivate""".format(name, argumentString)

def launchJob(num_hidden_units, learning_rate,
              momentum_mu, mini_batch_size):
    """Launches a job on the cluster for an experiment.

    Writes a submit script and then uses 'qsub' to submit job.
    This only works on HPC clusters that support this!

    Args:
        num_hidden_units: number of hidden units for experiment
        learning_rate: learning rate for experiment
        momentum_mu: momentum for experiment
        mini_batch_size: mini batch size for experiment
    """
    experiment_name = "experiment_{0}_{1}_{2}_{3}".format(num_hidden_units,
                                                          learning_rate,
                                                          momentum_mu,
                                                          mini_batch_size)
    argumentString = "{0} --num_hidden_units {1} --learning_rate {2} --momentum_mu {3} --mini_batch_size {4} --num_epoch {5}".format(experiment_name,
                                              num_hidden_units,
                                              learning_rate,
                                              momentum_mu,
                                              mini_batch_size,
                                              num_epoch)

    # write to file
    if os.path.isfile(SUBMIT_SCRIPT):
        os.remove(SUBMIT_SCRIPT)
    with open(SUBMIT_SCRIPT, "w") as f:
        f.write(writeScript(experiment_name, argumentString))

    # launch job
    call(["qsub", SUBMIT_SCRIPT])
    time.sleep(1)

    # remove script
    if os.path.isfile(SUBMIT_SCRIPT):
        os.remove(SUBMIT_SCRIPT)

def hidden_units_experiment():
    """Experiment of varying the number of hidden units
    while fixing the other hyperparameters.
    """
    for num_hidden_units in num_hidden_units_list:
        # use defaults for other variables
        learning_rate = learning_rate_list[0]
        momentum_mu = momentum_mu_list[0]
        mini_batch_size = mini_batch_size_list[0]

        # launch jobs
        launchJob(num_hidden_units,
                  learning_rate,
                  momentum_mu,
                  mini_batch_size)

def learning_rate_experiment():
    """Experiment of varying the learning rate
    while fixing the other hyperparameters.
    """
    for learning_rate in learning_rate_list:
        # use defaults for other variables
        num_hidden_units = num_hidden_units_list[0]
        momentum_mu = momentum_mu_list[0]
        mini_batch_size = mini_batch_size_list[0]

        # launch jobs
        launchJob(num_hidden_units,
                  learning_rate,
                  momentum_mu,
                  mini_batch_size)

def momentum_experiment():
    """Experiment of varying the momentum mu
    while fixing the other hyperparameters.
    """
    for momentum_mu in momentum_mu_list:
        # use defaults for other variables
        num_hidden_units = num_hidden_units_list[0]
        learning_rate = learning_rate_list[0]
        mini_batch_size = mini_batch_size_list[0]

        # launch jobs
        launchJob(num_hidden_units,
                  learning_rate,
                  momentum_mu,
                  mini_batch_size)

def batch_size_experiment():
    """Experiment of varying the batch size
    while fixing the other hyperparameters.
    """
    for mini_batch_size in mini_batch_size_list:
        # use defaults for other variables
        num_hidden_units = num_hidden_units_list[0]
        learning_rate = learning_rate_list[0]
        momentum_mu = momentum_mu_list[0]

        # launch jobs
        launchJob(num_hidden_units,
                  learning_rate,
                  momentum_mu,
                  mini_batch_size)

def main():
    hidden_units_experiment()
    learning_rate_experiment()
    momentum_experiment()
    batch_size_experiment()

if __name__ == '__main__':
    main()
