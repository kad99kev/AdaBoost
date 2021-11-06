import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

from adaboost.tree import DecisionTreeClassifier

from .plot_tests import plot_history


def train_scratch_dt(X_train, y_train, X_test, y_test, sample_weights=None):
    dt = DecisionTreeClassifier(max_depth=1)
    dt.fit(X_train, y_train, sample_weights)
    preds = dt.predict(X_test)
    return accuracy_score(y_test, preds)


def train_sklearn_dt(X_train, y_train, X_test, y_test, sample_weights=None):
    dt = SklearnDecisionTree(max_depth=1)
    dt.fit(X_train, y_train, sample_weight=sample_weights)
    preds = dt.predict(X_test)
    return accuracy_score(y_test, preds)


def test_decisiontree():
    df = pd.read_csv("wildfires.txt", sep="\t", header=0)
    X = df.drop(columns=["fire"])
    y = df.loc[:, "fire"]
    y = y.apply(lambda x: x.strip())

    history = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=i, test_size=int(len(X) / 3)
        )

        y_train = np.array([1 if v == "yes" else -1 for v in y_train])
        y_test = np.array([1 if v == "yes" else -1 for v in y_test])

        sklearn_acc = train_sklearn_dt(X_train, y_train, X_test, y_test)
        scratch_acc = train_scratch_dt(X_train, y_train, X_test, y_test)
        history.append((sklearn_acc, scratch_acc))

    history = np.round(history, 4)
    history_mean = np.round(history.mean(axis=0), 4)
    data = np.concatenate(
        (
            history,
            [history_mean],
        ),
        axis=0,
    )
    data = pd.DataFrame(data, columns=["Sklearn", "Scratch"])
    data.insert(0, "Run", [i + 1 if i < 10 else "<b>Mean</b>" for i in range(11)])
    plot_history(data, "decision-tree")


if __name__ == "__main__":
    test_decisiontree()
