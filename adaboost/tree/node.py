"""
Written by:
Name: Kevlyn Kadamala
Student ID: 21236191
Class: MSc AI
"""


class Node:
    """
    Class for each node in a decision tree.

    Arguments:
        gini: Gini value for the node.
        value: Sample distribution for each class.
        num_samples: Number of samples in the node.
        prediction: Prediction for the node.

    """

    def __init__(self, gini, value, num_samples, prediction):
        self.gini = gini
        self.num_samples = num_samples
        self.value = [float("%.2f" % v) for v in value]  # For neat printing.
        self.prediction = prediction

    def __str__(self):
        """
        To print information about the current node.
        """
        text = ""
        if hasattr(self, "left") and hasattr(self, "right"):
            text += f"{self.feature} <= {self.threshold:.3f}\n"

        text += f"Gini: {self.gini:.3f}\n"
        text += f"Samples: {self.num_samples}\n"
        text += f"Value: {self.value}\n"
        text += f"Class: {self.prediction}\n"

        return text

    def set_split_values(self, feature, threshold, left, right):
        """
        Sets the split values if the node is not a leaf node.

        Arguments:
            feature: Feature for which the split is taking place.
            threshold: The threshold value at which the split is taking place.
            left: Left child (node) of the current node.
            right: Right child (node) of the current node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
