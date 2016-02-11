class Monitor:
    """Monitor is a class that records iteration and epoch data.
    """

    def __init__(self):
        self.iter_fh = None
        self.epoch_fh = None

    def createSession(self, iteration_log_file, epoch_log_file):
        """Creates a monitor session.

        This opens files for logging statistics.

        Args:
            iteration_log_file: path to log file for saving iteration data
            epoch_log_file: path to log file for saving epoch data
        """
        self.iter_fh = open(iteration_log_file, 'w')
        self.epoch_fh = open(epoch_log_file, 'w')
        self.iter_fh.write("#\titeration\ttraining_loss\n")
        self.epoch_fh.write("#\tepoch\ttraining_loss\ttest_loss\ttraining_error_rate\ttest_error_rate\n")

    def recordIteration(self, iteration, training_loss, elapsed):
        """Record an iteration data point.

        This records the iteration, training loss and elapsed time in
        the log file.

        Args:
            iteration: iteration number (monotonically increasing)
            training_loss: training objective
            elapsed: time it takes to perform an iteration
        """
        self.iter_fh.write("{0}\t{1}\t{2}\n".format(iteration,
                                               training_loss,
                                               elapsed))

    def recordEpoch(self, epoch, training_loss, test_loss,
                    error_rate_train, error_rate_test, elapsed):
        """Record an epoch data point.

        This records the epoch, training loss, test loss,
        training error rate, test error rate and elapsed time in
        the log file.

        Args:
            iteration: iteration number (monotonically increasing)
            training_loss: training objective
            test_lost: test objective
            error_rate_train: misclassification error rate on training examples
            error_rate_test: misclassification error rate on test examples
            elapsed: time it takes to perform an iteration
        """
        self.epoch_fh.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(epoch,
                                                                    training_loss,
                                                                    test_loss,
                                                                    error_rate_train,
                                                                    error_rate_test,
                                                                    elapsed))

    def finishSession(self):
        """Finish a monitor session.

        This properly closes the log files.
        """
        self.iter_fh.close()
        self.epoch_fh.close()
        self.iter_fh = None
        self.epoch_fh = None
