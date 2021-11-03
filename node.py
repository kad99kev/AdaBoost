import numpy as np


class Node:
    def __init__(self, gini, value, y):
        self.gini = gini
        self.num_samples = len(y)
        self.value = value
        self.prediction = y[np.argmax(self.value)]

    def __str__(self):
        text = ""
        if hasattr(self, "left") and hasattr(self, "right"):
            text += f"{self.feature} <= {self.threshold}\n"

        text += f"Gini: {self.gini}\n"
        text += f"Samples: {self.num_samples}\n"
        text += f"Value: {self.value}\n"
        text += f"Class: {self.prediction}\n"

        return text

    def set_split_values(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
