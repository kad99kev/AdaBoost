"""
Written by:
Name: Kevlyn Kadamala
Student ID: 21236191
Class: MSc AI
"""

import numpy as np
from .node import Node
from .tree_viz import plot_tree


class DecisionTreeClassifierScratch:
    """
    This class implements the Decision Tree Classifier from scratch.
    It uses the CART algorithm to build the tree.

    Arguments:
        max_depth: Maximum depth of the tree.
    """

    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.root = None
        self.nodes = []

    def _build_tree(self, X, y, sample_weight=None, current_depth=0):
        """
        Build the Decision Tree.

        Arugments:
            X: The inputs to the tree.
            y: The target values.
            sample_weight: Sample weights to be used for while calculating the Gini Index.
            current_depth: Current depth of the tree (for recursive building).
        """

        # Calculate values based on either sample weights if provided or class counts.
        if sample_weight is not None:
            value = [sample_weight[y == c].sum() for c in self.classes]
        else:
            value = [np.count_nonzero(y == c) for c in self.classes]
        prediction = self.classes[np.argmax(value)]

        # Initially create a leaf node.
        node = Node(
            self._compute_gini_index(y, sample_weight), value, len(y), prediction
        )

        # Check if exceeds depth
        if current_depth < self.max_depth:
            best_split = self._get_split(X, y, sample_weight)

            if best_split["idx"] is not None:

                # Use best split to split data for recursive call
                left_indices = X[:, best_split["idx"]] < best_split["thres"]
                X_left, y_left = X[left_indices], y[left_indices]
                X_right, y_right = X[~left_indices], y[~left_indices]

                sample_weight_left, sample_weight_right = None, None
                if sample_weight is not None:
                    sample_weight_left, sample_weight_right = (
                        sample_weight[left_indices],
                        sample_weight[~left_indices],
                    )

                # Building a node with max leaf nodes = 2
                # Current node will now be a parent node.
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
                # Keep track of edges for plotting.
                self.nodes.extend(
                    [
                        (str(node), str(node.right)),
                        (str(node), str(node.left)),
                    ]
                )

        return node

    def _get_split(self, X, y, sample_weights):
        """
        To find the best split in the data.

        Arguments:
            X: Current input data.
            y: Current target values.
            sample_weights: Current sample weights.
        """

        # Get number of samples and features in current split.
        num_samples, num_features = X.shape

        # Calculates the gini index for current split.
        best_gini = self._compute_gini_index(y, sample_weights)
        best_dict = {"idx": None, "thres": None}

        # Iterate through each feature in current data.
        for idx in range(num_features):
            feature_values = X[:, idx]
            thresholds = np.unique(feature_values)
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2

            # Check threshold values to find split.
            for thres in thresholds:
                left_indices = X[:, idx] <= thres
                y_left, y_right = y[left_indices], y[~left_indices]

                # Keep sample weights initially as None.
                # If sample weights are provided, then split.
                sample_weights_left, sample_weights_right = None, None
                if sample_weights is not None:
                    sample_weights_left, sample_weights_right = (
                        sample_weights[left_indices],
                        sample_weights[~left_indices],
                    )

                if len(y_left) > 0 and len(y_right) > 0:
                    # sum((S_v / S) * Gini(S_v))
                    current_gini = self._compute_gini_split(
                        num_samples,
                        y_left,
                        y_right,
                        sample_weights_left,
                        sample_weights_right,
                    )

                    # Check if current gini values are better than the best gini value.
                    # If better, then update values.
                    if current_gini < best_gini:
                        best_gini = current_gini
                        best_dict["idx"] = idx
                        best_dict["thres"] = thres

        # Return best split.
        return best_dict

    def _compute_gini_split(
        self, num_samples, y_left, y_right, sample_weights_left, sample_weights_right
    ):
        """
        Compute the Gini index for the current split.

        Arguments:
            num_samples: Number of samples in the current split.
            y_left: Left split of the target values.
            y_right: Right split of the target values.
            sample_weights_left: Left split of the sample weights.
            sample_weights_right: Right split of the sample weights.
        """

        # Get weights for left and right split.
        if sample_weights_left is not None and sample_weights_right is not None:
            # If using sample weight, get split weights based on it.
            weight_left = sample_weights_left.sum()
            weight_right = sample_weights_right.sum()
            total_weight = weight_left + weight_right
            weight_left /= total_weight
            weight_right /= total_weight
        else:
            # Else get split weights based on count.
            weight_left = len(y_left) / num_samples
            weight_right = len(y_right) / num_samples

        # Calculate Gini based on split.
        gini = weight_left * self._compute_gini_index(
            y_left, sample_weights_left
        ) + weight_right * self._compute_gini_index(y_right, sample_weights_right)

        return gini

    def _compute_gini_index(self, y, sample_weights):
        """
        Compute Gini index.

        Arguments:
            y: The target values.
            sample_weights: The sample weights.
        """

        # Gini = 1 - sum(p^2)
        if sample_weights is not None:
            classes = np.unique(y)
            return 1 - sum(
                (sample_weights[y == c].sum() / sample_weights.sum()) ** 2
                for c in classes
            )
        else:
            _, classes_count = np.unique(y, return_counts=True)
            num_samples = len(y)

            return 1.0 - sum((n / num_samples) ** 2 for n in classes_count)

    def fit(self, X, y, sample_weight=None):
        """
        Fit the Decision Tree based on given data.

        Arguments:
            X: The inputs to the tree.
            y: The target values.
            sample_weight: Sample weights to be used for while calculating the Gini Index.
        """
        # Build tree
        X, y = np.array(X), np.array(y)
        self.classes = np.unique(y)
        self.root = self._build_tree(X, y, sample_weight)

    def predict(self, X):
        """
        Predict on unseen values.

        Arguments:
            X: Inputs to the tree.
        """
        return [self._predict(X_) for X_ in np.array(X)]

    def _predict(self, X_):
        """
        Recursively iterate through the tree to obtain predictions.

        Arguments:
            X_: Current input split.
        """
        node = self.root
        while hasattr(node, "left"):
            if X_[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def plot_tree(self):
        """
        Plot the tree.
        """
        # Reverse and return so that the left most node is towards the start of the list.
        plot_tree(self.nodes[::-1], str(self.root), self.classes)
