import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost

from adaboost import AdaBoostClassifier

from .plot_tests import plot_history


def train_sklearn_SAMME(X_train, y_train, X_test, y_test, sample_weights=None):
    """
    Training the sklearn model using SAMME algorithm
    """
    ada = SklearnAdaBoost(n_estimators=50, algorithm="SAMME")
    ada.fit(X_train, y_train, sample_weights)
    preds = ada.predict(X_test)
    return accuracy_score(y_test, preds)

def train_sklearn_SAMMER_R(X_train, y_train, X_test, y_test, sample_weights=None):
    """
    Training the sklearn model using SAMME.R algorithm
    """
    ada = SklearnAdaBoost(n_estimators=50, algorithm="SAMME.R")
    ada.fit(X_train, y_train, sample_weights)
    preds = ada.predict(X_test)
    return accuracy_score(y_test, preds)


def train_scratch(X_train, y_train, X_test, y_test, sample_weights=None):
    """
    Training adaboost from scratch
    """
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train, sample_weights)
    preds = ada.predict(X_test)
    return accuracy_score(y_test, preds)


def test_adaboost():
    """
    Test the adaboost algorithm on the sklearn and scratch implementation
    """
    df = pd.read_csv("Iris.csv", sep=",", header=0)
    X = df.drop(columns=["Species"])
    y = df.loc[:, "Species"]
    y = y.apply(lambda x: x.strip())
    # print(y)

    history = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=i, test_size=int(len(X) / 3)
        )

        # y_train = np.array([1 if v == "yes" else -1 for v in y_train])
        # y_test = np.array([1 if v == "yes" else -1 for v in y_test])

        sklearn_acc_samme = train_sklearn_SAMME(X_train, y_train, X_test, y_test)
        sklearn_acc_samme_R = train_sklearn_SAMMER_R(X_train, y_train, X_test, y_test)
        scratch_acc = train_scratch(X_train, y_train, X_test, y_test)
        history.append((sklearn_acc_samme, sklearn_acc_samme_R, scratch_acc))

    history = np.round(history, 4)
    history_mean = np.round(history.mean(axis=0), 4)
    data = np.concatenate(
        (
            history,
            [history_mean],
        ),
        axis=0,
    )
    data = pd.DataFrame(data, columns=["Sklearn (SAMME)", "Sklearn (SAMME.R)", "Scratch"])
    data.insert(0, "Run", [i + 1 if i < 10 else "<b>Mean</b>" for i in range(11)])
    plot_history(data, "adaboost_iris")


if __name__ == "__main__":
    test_adaboost()
