"""
Written by:
Name: Elita Menezes
Student ID: 21237434
Class: MSc DA
"""

import numpy as np
from .tree import DecisionTreeClassifierScratch


class AdaBoostClassifierScratch:
    """
    This class implements the AdaBoost Classifier from scratch.
    Reference: https://web.stanford.edu/~hastie/Papers/SII-2-3-A8-Zhu.pdf

    Arguments:
        n_estimators: The number of estimators (trees) to be trained.
        learning_rate: Weight applied to each weak classifier at each iteration.
    """

    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = [None] * n_estimators  # Stores the model after each iteration.
        self.alphas = [None] * n_estimators  # Stores the weights after each iteration.
        self.errors = [
            None
        ] * n_estimators  # Stores the training error after each iteration.

    def fit(self, X, y, sample_weights=None):
        """
        Fit the model on the training data.

        Arguments:
            X: Inputs to the model.
            y: The target values.
            sample_weights: Initial sample weights to be used for training.
        """

        # Initally, initalize observation weights wi = 1 / len(N).
        N = len(y)
        sample_weights = np.ones(N) / N if sample_weights is None else sample_weights
        self.classes = np.unique(y)
        k = len(self.classes)

        # For i in the range of n_estimators.
        for i in range(self.n_estimators):

            # Fit the classifier with the updated weights.
            clf = DecisionTreeClassifierScratch(max_depth=1)
            clf.fit(X, y, sample_weight=sample_weights)
            predictions = clf.predict(X)

            # Compute error.
            # For all the incorrect predictions add their respective weights.
            incorrect = predictions != y
            error = sum(sample_weights * incorrect) / sum(sample_weights)
            self.errors[i] = error

            # Compute alpha.
            # Weight of the weak classifier.
            alpha = (self.learning_rate * (np.log((1 - error) / error))) + np.log(k - 1)

            # Update the weights.
            # New weights is the prev weights * exponential of [alpha for all incorrect predictions].
            sample_weights = sample_weights * np.exp(alpha * incorrect)

            # Renormalize sample_weights.
            sample_weights /= sum(sample_weights)

            # Update the list of models and alpha for carrying out predictions.
            self.models[i] = clf
            self.alphas[i] = alpha

    def predict(self, X):
        """
        Predict on unseen values.

        Arguments:
            X: Inputs to the model.
        """
        pred = 0
        for (clf, a) in zip(self.models, self.alphas):
            # Gives us output of shape (n_classes, 1).
            reshaped_classes = clf.classes[:, np.newaxis]
            # Initial shape, (n_classes, *).
            # Transposing gives us a shape of (*, n_classes).
            pred += (clf.predict(X) == np.array(reshaped_classes)).T * a

        # Get the argmax from the predictions.
        pred = np.argmax(pred, axis=1)
        # Returns values in the form of class names
        return [self.classes[i] for i in pred]
