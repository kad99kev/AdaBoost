import numpy as np
import pandas as pd
from tree import DecisionTreeClassifier


class AdaBoost:
    # https://web.stanford.edu/~hastie/Papers/SII-2-3-A8-Zhu.pdf
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.models = [None] * n_estimators  # stores the model after each iteration
        self.alphas = [None] * n_estimators  # stores the weights after each iteration
        self.learning_rate = learning_rate

    def check_y(self, y):
        """
        For adaboost classifier, the values of y must be -1 or 1
        """
        assert set(y) == {-1, 1}, "Values for y should be -1 or 1"
        return y

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model on the training data
        """
        # check if y is -1 and 1 values
        # self.check_y(y)

        # Initally, initalize observation weights wi = 1 / len(N)
        N = len(y)
        sample_weights = np.ones(N) / N if sample_weight is None else sample_weight
        self.classes = np.unique(y)

        # for i in the range of n_estimators
        for i in range(self.n_estimators):

            # fit the classifier with the updated weights
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(X, y, sample_weight=sample_weights)
            predictions = clf.predict(X)

            # compute error
            # for all the incorrect predictions add their respective weights
            incorrect = predictions != y
            error = sum(sample_weights * incorrect) / sum(sample_weights)
            # print("Error ", error)

            # compute alpha
            # weight of the weak classifier
            alpha = self.learning_rate * (np.log((1 - error) / error))
            # print("Alpha ", alpha)

            # update the weights
            # new weights is the prev weights * exponential of [alpha for all incorrect predictions]
            sample_weights = sample_weights * np.exp(alpha * incorrect)

            # renormalize sample_weights
            sample_weights /= sum(sample_weights)
            # print("Weights ", sample_weights)

            # update the list of models and alpha for carrying out predictions
            self.models[i] = clf
            self.alphas[i] = alpha

    def predict(self, X):
        """
        Predict on the test set
        """
        pred = 0
        for (clf, a) in zip(self.models, self.alphas):
            # Gives us output of shape (n_classes, 1)
            self.classes = clf.classes[:, np.newaxis]
            # Initial shape, (2, *)
            # Transposing gives us a shape of (*, 2)
            pred += (clf.predict(X) == np.array(self.classes)).T * a

        # Get the argmax from the predictions
        pred = np.argmax(pred, axis=1)
        # Returns values in the form of [-1, 1]
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
    clf = AdaBoost(learning_rate=0.5)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print(accuracy_score(y_test, pred))

"""
The difference in the paper and sklearn.

1. In the paper implementation, there is no learning rate. \
    Should we implement the learning rate?
2. The implementation of boosting sample weight is different in the paper and sklearn. \
    Changing this formula helps us reproduce the sklearn implementation but deviates from the paper.
"""
