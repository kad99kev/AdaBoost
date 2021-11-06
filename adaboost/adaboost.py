"""
Written by:
Name: Elita Menezes
Student ID:
Class: MSc DA
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .tree import DecisionTreeClassifier


class AdaBoostClassifier:
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
        self.training_error = [
            None
        ] * n_estimators  # Stores the training error after each iteration.

    def plot_error_rates(self):
        """
        Plotting the error rate for each iteration.
        """

        fig = go.Figure(
            data=go.Scatter(
                x=[i for i in range(self.n_estimators)], y=self.training_error
            )
        )
        fig.update_layout(
            title="Error rates for each iteration",
            xaxis_title="Iteration",
            yaxis_title="Error",
        )
        fig.show()

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

        # For i in the range of n_estimators.
        for i in range(self.n_estimators):

            # Fit the classifier with the updated weights.
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(X, y, sample_weight=sample_weights)
            predictions = clf.predict(X)

            # Compute error.
            # For all the incorrect predictions add their respective weights.
            incorrect = predictions != y
            error = sum(sample_weights * incorrect) / sum(sample_weights)
            self.training_error[i] = error

            # Compute alpha.
            # Weight of the weak classifier.
            alpha = self.learning_rate * (np.log((1 - error) / error))

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
            self.classes = clf.classes[:, np.newaxis]
            # Initial shape, (2, *).
            # Transposing gives us a shape of (*, 2).
            pred += (clf.predict(X) == np.array(self.classes)).T * a

        # Get the argmax from the predictions.
        pred = np.argmax(pred, axis=1)
        # Returns values in the form of [-1, 1].
        return self.classes.take(pred > 0, axis=0)


if __name__ == "__main__":
    df = pd.read_csv("wildfires.txt", sep="\t", header=0)
    X = df.drop(columns=["fire"])
    y = df.loc[:, "fire"]
    y = y.apply(lambda x: x.strip())

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=int(len(X) / 3)
    )

    y_train = np.array([1 if v == "yes" else -1 for v in y_train])
    y_test = np.array([1 if v == "yes" else -1 for v in y_test])
    clf = AdaBoostClassifier(learning_rate=0.5)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print(f"Accuracy: ", accuracy_score(y_test, pred))

    # clf.plot_error_rates()
    # plot_roc_curve(y_test, pred)

"""
The difference in the paper and sklearn.

1. In the paper implementation, there is no learning rate. \
    Should we implement the learning rate?
2. The implementation of boosting sample weight is different in the paper and sklearn. \
    Changing this formula helps us reproduce the sklearn implementation but deviates from the paper.
"""
