#!/usr/bin/env python
"""This script analyzes multiple hyperparameter experiments.
It generates objective and misclassification error rate plots.
"""

from util.plot import Plot

# SET UP HYPERPARAMETERS TO EVALUATE BELOW
# The first element of the list will be used as a default value.
num_hidden_units_list = [50, 10, 25, 100, 200]
learning_rate_list = [0.01, 0.001, 0.005]
momentum_mu_list = [0, 0.2, 0.4, 0.6]
mini_batch_size_list = [256, 32, 64, 128]

def analyze(plt, num_hidden_units, learning_rate, momentum_mu, mini_batch_size):
    """Helper function.
    """
    experiment_name = "experiment_{0}_{1}_{2}_{3}".format(num_hidden_units, learning_rate, momentum_mu, mini_batch_size)
    epoch_log_file = "logs/{0}_epoch_log.txt".format(experiment_name)
    objectives_plot = "figures/{0}_objectives.png".format(experiment_name)
    errorrates_plot = "figures/{0}_errorrates.png".format(experiment_name)
    # iter_log_file = "logs/{0}_iter_log.txt".format(experiment_name)
    # training_plot = "figures/{0}_training.png".format(experiment_name)

    # generate plots
    plt.epochPlotObjectives(epoch_log_file, objectives_plot)
    plt.epochPlotErrorRates(epoch_log_file, errorrates_plot)
    # plt.iterationPlot(iter_log_file, training_plot)

    return experiment_name

def analyze_hidden_units(plt):
    """Analyze varying the number of hidden units
    while fixing the other hyperparameters.

    Args:
        plt: matplotlib object
    """
    input_file_list = []
    for num_hidden_units in num_hidden_units_list:
        # use defaults for other variables
        learning_rate = learning_rate_list[0]
        momentum_mu = momentum_mu_list[0]
        mini_batch_size = mini_batch_size_list[0]

        experiment_name = analyze(plt, num_hidden_units, learning_rate, momentum_mu, mini_batch_size)
        input_file_list.append("logs/{0}_epoch_log.txt".format(experiment_name))

    var_title = "Number of Hidden Units"
    var_list = num_hidden_units_list
    output_file = "figures/hiddenunits.png"
    plt.plotTestAccuracy(var_title, var_list, input_file_list, output_file)

def analyze_learning_rate(plt):
    """Analyze varying the learning rate
    while fixing the other hyperparameters.

    Args:
        plt: matplotlib object
    """
    input_file_list = []
    for learning_rate in learning_rate_list:
        # use defaults for other variables
        num_hidden_units = num_hidden_units_list[0]
        momentum_mu = momentum_mu_list[0]
        mini_batch_size = mini_batch_size_list[0]

        experiment_name = analyze(plt, num_hidden_units, learning_rate, momentum_mu, mini_batch_size)
        input_file_list.append("logs/{0}_epoch_log.txt".format(experiment_name))

    var_title = "Learning Rate"
    var_list = learning_rate_list
    output_file = "figures/learningrate.png"
    plt.plotTestAccuracy(var_title, var_list, input_file_list, output_file)

def analyze_momentum(plt):
    """Analyze varying the momentum mu
    while fixing the other hyperparameters.

    Args:
        plt: matplotlib object
    """
    input_file_list = []
    for momentum_mu in momentum_mu_list:
        # use defaults for other variables
        num_hidden_units = num_hidden_units_list[0]
        learning_rate = learning_rate_list[0]
        mini_batch_size = mini_batch_size_list[0]

        experiment_name = analyze(plt, num_hidden_units, learning_rate, momentum_mu, mini_batch_size)
        input_file_list.append("logs/{0}_epoch_log.txt".format(experiment_name))

    var_title = "Momentum"
    var_list = momentum_mu_list
    output_file = "figures/momentum.png"
    plt.plotTestAccuracy(var_title, var_list, input_file_list, output_file)

def analyze_batch_size(plt):
    """Analyze varying the batch size
    while fixing the other hyperparameters.

    Args:
        plt: matplotlib object
    """
    input_file_list = []
    for mini_batch_size in mini_batch_size_list:
        # use defaults for other variables
        num_hidden_units = num_hidden_units_list[0]
        learning_rate = learning_rate_list[0]
        momentum_mu = momentum_mu_list[0]

        experiment_name = analyze(plt, num_hidden_units, learning_rate, momentum_mu, mini_batch_size)
        input_file_list.append("logs/{0}_epoch_log.txt".format(experiment_name))

    var_title = "Mini Batch Size"
    var_list = mini_batch_size_list
    output_file = "figures/batchsize.png"
    plt.plotTestAccuracy(var_title, var_list, input_file_list, output_file)

def main():
    plt = Plot()
    analyze_hidden_units(plt)
    analyze_learning_rate(plt)
    analyze_momentum(plt)
    analyze_batch_size(plt)

if __name__ == '__main__':
    main()
