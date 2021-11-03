import numpy as np
import pandas as pd

from node import Node


class DecisionTreeClassifier:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.root = None

    def _build_tree(self, X, y, sample_weight=None, current_depth=0):

        if sample_weight is not None:
            value = [sample_weight[y == c].sum() for c in self.classes]
        else:
            value = [np.count_nonzero(y == c) for c in self.classes]

        node = Node(self._compute_gini_index(y, sample_weight), value, y)

        # Check if exceeds depth
        if current_depth < self.max_depth:
            best_split = self._get_split(X, y, sample_weight)

            if best_split["idx"] is not None:

                # Use best split to split data for recursive call
                left_indices = X[:, best_split["idx"]] <= best_split["thres"]
                X_left, y_left = X[left_indices], y[left_indices]
                X_right, y_right = X[~left_indices], y[~left_indices]

                sample_weight_left, sample_weight_right = None, None
                if sample_weight is not None:
                    sample_weight_left, sample_weight_right = (
                        sample_weight[left_indices],
                        sample_weight[~left_indices],
                    )

                # Building a node with max leaf nodes = 2
                node.set_split_values(
                    best_split["idx"],
                    best_split["thres"],
                    self._build_tree(
                        X_left, y_left, sample_weight_left, current_depth + 1
                    ),
                    self._build_tree(
                        X_right, y_right, sample_weight_right, current_depth + 1
                    ),
                )

        return node

    def _get_split(self, X, y, sample_weight):

        # Get number of samples and features in current split
        num_samples, num_features = X.shape

        # Calculates the gini index for current split
        best_gini = self._compute_gini_index(y, sample_weight)
        best_dict = {"idx": None, "thres": None}

        # Iterate through each feature in current data
        for idx in range(num_features):
            feature_values = X[:, idx]
            thresholds = np.unique(feature_values)

            # Check threshold values to find split
            for thres in thresholds:
                left_indices = X[:, idx] <= thres
                y_left, y_right = y[left_indices], y[~left_indices]

                sample_weight_left, sample_weight_right = None, None
                if sample_weight is not None:
                    sample_weight_left, sample_weight_right = (
                        sample_weight[left_indices],
                        sample_weight[~left_indices],
                    )

                if len(y_left) > 0 and len(y_right) > 0:
                    # sum((S_v / S) * Gini(S_v))
                    current_gini = self._compute_gini_split(
                        num_samples,
                        y_left,
                        y_right,
                        sample_weight_left,
                        sample_weight_right,
                    )

                    # Check if current gini values are better than the best gini value
                    # If better, then update values
                    if current_gini < best_gini:
                        best_gini = current_gini
                        best_dict["idx"] = idx
                        best_dict["thres"] = thres

        # Return best split
        return best_dict

    def _compute_gini_split(
        self, num_samples, y_left, y_right, sample_weight_left, sample_weight_right
    ):
        # Get weights for left and right split
        weight_left = len(y_left) / num_samples
        weight_right = len(y_right) / num_samples

        # Calculate gain based on split
        gain = weight_left * self._compute_gini_index(
            y_left, sample_weight_left
        ) + weight_right * self._compute_gini_index(y_right, sample_weight_right)

        return gain

    def _compute_gini_index(self, y, sample_weight):
        if sample_weight is not None:
            classes = np.unique(y)
            return 1 - sum(
                (sample_weight[y == c].sum() / sample_weight.sum()) ** 2
                for c in classes
            )
        else:
            _, classes_count = np.unique(y, return_counts=True)
            num_samples = len(y)

            # Gini = 1 - sum(p^2)
            return 1.0 - sum((n / num_samples) ** 2 for n in classes_count)

    def fit(self, X, y, sample_weight=None):
        # Build tree
        X, y = np.array(X), np.array(y)
        self.classes = np.unique(y)
        self.root = self._build_tree(X, y, sample_weight)
        self.print_tree(self.root)

    def predict(self, X):
        return [self._predict(X_) for X_ in np.array(X)]

    def _predict(self, X_):
        node = self.root
        while hasattr(node, "left"):
            if X_[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def print_tree(self, node):
        print(node)
        if not hasattr(node, "left") and not hasattr(node, "right"):
            return
        self.print_tree(node.left)
        self.print_tree(node.right)
