import numpy as np
import pandas as pd

from node import Node


class DecisionTreeClassifier:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.root = None

    def _build_tree(self, X, y, current_depth=0):

        node = Node(self._compute_gini_index(y), y)

        # Check if exceeds depth
        if current_depth < self.max_depth:
            best_split = self._get_split(X, y)

            if best_split["idx"] is not None:

                # Use best split to split data for recursive call
                left_indices = X[:, best_split["idx"]] <= best_split["thres"]
                X_left, y_left = X[left_indices], y[left_indices]
                X_right, y_right = X[~left_indices], y[~left_indices]
                node.set_split_values(
                    best_split["idx"],
                    best_split["thres"],
                    self._build_tree(X_left, y_left, current_depth + 1),
                    self._build_tree(X_right, y_right, current_depth + 1),
                )

        return node

    def _get_split(self, X, y):
        # TODO: Stopping condition of min samples

        # Calculate Gini
        num_samples, num_features = X.shape

        # Calculates the gini index for current node
        best_gini = self._compute_gini_index(y)
        best_dict = {"idx": None, "thres": None}

        # Iterate through each feature in current data
        for idx in range(num_features):
            feature_values = X[:, idx]
            thresholds = np.unique(feature_values)

            # Check threshold values to find split
            for thres in thresholds:
                left_indices = X[:, idx] <= thres
                y_left, y_right = y[left_indices], y[~left_indices]

                if len(y_left) > 0 and len(y_right) > 0:
                    # sum((S_v / S) * Gini(S_v))
                    current_gini = self._compute_gini_split(
                        num_samples, y_left, y_right
                    )

                    # Check if current gini values are better than the best gini value
                    # If better, then update values
                    if current_gini < best_gini:
                        best_gini = current_gini
                        best_dict["idx"] = idx
                        best_dict["thres"] = thres

        return best_dict

    def _compute_gini_split(self, num_samples, y_left, y_right):
        weight_left = len(y_left) / num_samples
        weight_right = len(y_right) / num_samples

        gain = weight_left * self._compute_gini_index(
            y_left
        ) + weight_right * self._compute_gini_index(y_right)

        return gain

    def _compute_gini_index(self, y):
        _, classes_count = np.unique(y, return_counts=True)
        num_samples = len(y)

        # Gini = 1 - sum(p^2)
        return 1.0 - sum((n / num_samples) ** 2 for n in classes_count)

    def fit(self, X, y):
        # Build tree
        self.root = self._build_tree(X.to_numpy(), y.to_numpy())
        self.print_tree(self.root)

    def predict(self):
        pass

    def print_tree(self, node):
        print(node)
        if not hasattr(node, "left") and not hasattr(node, "right"):
            return
        self.print_tree(node.left)
        self.print_tree(node.right)


if __name__ == "__main__":
    df = pd.read_csv("wildfires.txt", sep="\t", header=0)
    X = df.drop(columns=["fire"])
    y = df.loc[:, "fire"]
    y = y.apply(lambda x: x.strip())

    dt = DecisionTreeClassifier()
    dt.fit(X, y)
