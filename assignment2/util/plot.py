import numpy as np
import matplotlib.pyplot as plt

class Plot:
    """Class for generating plots.
    """

    E_EPOCH_INDEX = 0
    E_TRAIN_OBJECTIVE_INDEX = 1
    E_TEST_OBJECTIVE_INDEX = 2
    E_TRAIN_ERROR_RATE_INDEX = 3
    E_TEST_ERROR_RATE_INDEX = 4
    E_ELASPSED_INDEX = 5

    I_ITER_INDEX = 0
    I_TRAIN_OBJECTIVE_INDEX = 1
    I_ELASPSED_INDEX = 2

    def __saveFigure__(self, output_file):
        """Helper function to save and clear current figure.

        Args:
            output_file: png file to save figure
        """
        # save figure
        plt.savefig(output_file, dpi=72)
        # plt.show()
        plt.clf()
        # plt.close()

    def epochPlotObjectives(self, input_file, output_file):
        """Plot the training and test objectives at each epoch.

        Args:
            input_file: file containing text data (written from Monitor)
            output_file: png file to save figure
        """
        # load data
        try:
            data = np.loadtxt(input_file)
        except:
            print("error loading {0}".format(input_file))
            return

        try:
            epoch_data = data[:, Plot.E_EPOCH_INDEX]
            train_objective_data = data[:, Plot.E_TRAIN_OBJECTIVE_INDEX]
            test_objective_data = data[:, Plot.E_TEST_OBJECTIVE_INDEX]
        except:
            print("error reading {0}".format(input_file))
            return

        # plot data
        plt.plot(epoch_data, train_objective_data,
                 color="blue", linewidth=2.5, linestyle="-", label="Training")
        plt.plot(epoch_data, test_objective_data,
                 color="red", linewidth=2.5, linestyle="-", label="Testing")
        plt.legend(loc="lower left", frameon=False)
        plt.xlabel("Epoch")
        plt.ylabel("Objective")
        plt.title("Training/Testing Objectives vs. Epoch")

        # save figure
        self.__saveFigure__(output_file)

    def epochPlotErrorRates(self, input_file, output_file):
        """Plot the training and test misclassification error rates at each epoch.

        Args:
            input_file: file containing text data (written from Monitor)
            output_file: png file to save figure
        """
        # load data
        try:
            data = np.loadtxt(input_file)
        except:
            print("error loading {0}".format(input_file))
            return

        try:
            epoch_data = data[:, Plot.E_EPOCH_INDEX]
            train_error_rates_data = data[:, Plot.E_TRAIN_ERROR_RATE_INDEX]
            test_error_rates_data = data[:, Plot.E_TEST_ERROR_RATE_INDEX]
        except:
            print("error reading {0}".format(input_file))
            return

        # plot data
        plt.plot(epoch_data, train_error_rates_data,
                 color="blue", linewidth=2.5, linestyle="-", label="Training")
        plt.plot(epoch_data, test_error_rates_data,
                 color="red", linewidth=2.5, linestyle="-", label="Testing")
        plt.legend(loc="lower left", frameon=False)
        plt.xlabel("Epoch")
        plt.ylabel("Misclassification Error Rate")
        plt.title("Training/Testing Misclassification Error Rates vs. Epoch")

        # save figure
        self.__saveFigure__(output_file)

    def iterationPlot(self, input_file, output_file):
        """Plot the training objectives at each iteration for debugging purposes.

        Args:
            input_file: file containing text data (written from Monitor)
            output_file: png file to save figure
        """
        # load data
        try:
            data = np.loadtxt(input_file)
        except:
            print("error loading {0}".format(input_file))
            return

        try:
            data = np.loadtxt(input_file)
            iter_data = data[:, Plot.I_ITER_INDEX]
            train_objective_data = data[:, Plot.I_TRAIN_OBJECTIVE_INDEX]
        except:
            print("error reading {0}".format(input_file))
            return

        # plot data
        plt.plot(iter_data, train_objective_data,
                 color="blue", linewidth=2.5, linestyle="-", label="Training Objective")
        plt.legend(loc="lower left", frameon=False)
        plt.xlabel("Iteration")
        plt.ylabel("Objectives")
        plt.title("Training Objective vs. Iteration")

        # save figure
        self.__saveFigure__(output_file)
