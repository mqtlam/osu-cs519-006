from subprocess import call
import os

# constants
SUBMIT_SCRIPT = "submit.csh"

# variables
num_hidden_units_list = [100, 50, 500, 1000, 3000, 5000]
learning_rate_list = [0.01, 0.001, 0.1, 0.5, 1.0]
momentum_mu_list = [0.6, 0.2, 0.4, 0.8]
mini_batch_size_list = [256, 1, 32, 64, 128]
num_epoch = 1000

def writeScript(name, argumentString):
    return """#!/bin/csh

# use current working directory for input and output
# default is to use the users home directory
#$ -cwd

# name this job
#$ -N {0}

# send stdout and stderror to this file
#$ -o {0}.txt
#$ -j y

# select queue
#$ -q eecs,eecs2

# activate
source venv/bin/activate.csh

# see where the job is being run
hostname

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
    experiment_name = "experiment_{0}_{1}_{2}_{3}".format(num_hidden_units,
                                                          learning_rate,
                                                          momentum_mu,
                                                          mini_batch_size)
    argumentString = "{0} --num_hidden_units {1} --learning_rate {2} "
                   + "--momentum_mu {3} --mini_batch_size {4} "
                   + "--num_epoch {5}".format(experiment_name,
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

def main():
    # experiments for num of hidden units
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

if __name__ == '__main__':
    main()
