import numpy as np
from module import Module

class Metric(Module):
    """Abstract class for metrics.
    """
    def compute(self, predictions, target):
        """Compute a metric on confidence predictions and groundtruth targets.

        Args:
            predictions: cx1xb numpy array
                c = number of classes (sums to 1 along this dimension)
                    index of c indicates class confidence
                b = batch size

            target: 1x1xb numpy array containing groundtruth labels
                class labels must be zero-indexed

        Returns:
            metric over all batches (number)
        """
        pass

class ErrorRate(Metric):
    """Misclassification Error Rate.
    """
    def predictionToLabel(self, output):
        return np.argmax(output, 0)[...,None]

    def compute(self, predictions, target):
        """Compute the misclassification error rate on confidence predictions
        and groundtruth targets.

        Misclassification rate is when a training example's predicted label
        does not equal the ground truth label, averaged over all examples in
        the batch.

        Args:
            predictions: cx1xb numpy array
                c = number of classes (sums to 1 along this dimension)
                    index of c indicates class confidence
                b = batch size

            target: 1x1xb numpy array containing groundtruth labels
                class labels must be zero-indexed

        Returns:
            misclassification error rate (number)
        """
        # convert confidences to label prediction
        predictions_unbatched = self.__unbatch__(predictions)
        for i in range(len(predictions_unbatched)):
            predictions_unbatched[i] = self.predictionToLabel(predictions_unbatched[i])
        output = self.__batch__(predictions_unbatched)

        # compare labels to get misclassification error rate
        batch_size = target.shape[2]
        errors = (output != target).astype(int)
        return 1./batch_size*errors.sum(2)[0,0]

class Objective(Metric):
    """Use a loss function as a metric. E.g. cross entropy.
    """
    def __init__(self, loss_func):
        """Initialize.

        Args:
            loss_func: loss function (needs to implement Loss)
        """
        self.loss_func = loss_func

    def compute(self, predictions, target):
        """Compute the loss on confidence predictions and groundtruth targets.

        Uses the loss function intialized.

        Args:
            predictions: cx1xb numpy array
                c = number of classes (sums to 1 along this dimension)
                    index of c indicates class confidence
                b = batch size

            target: 1x1xb numpy array containing groundtruth labels
                class labels must be zero-indexed

        Returns:
            loss averaged over batch examples (number)
        """
        batch_size = target.shape[2]
        losses = self.loss_func.forward(predictions, target)
        return 1./batch_size*losses.sum(2)[0,0]
