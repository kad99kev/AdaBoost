import numpy as np


class Node:
    def __init__(self, gini, y):
        self.gini = gini
        self.num_samples = len(y)
        _, self.classes_count = np.unique(y, return_counts=True)
        self.prediction = np.argmax(self.classes_count)

    def __str__(self):
        text = ""
        if hasattr(self, "left") and hasattr(self, "right"):
            text += f"{self.feature} <= {self.threshold}\n"

        text += f"Gini: {self.gini}\n"
        text += f"Samples: {self.num_samples}\n"
        text += f"Value: {self.classes_count}\n"
        text += f"Class: {self.prediction}\n"

        return text

    def set_split_values(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
