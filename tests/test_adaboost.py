import argparse
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost

from adaboost import AdaBoostClassifierScratch
from .plot_tests import (
    plot_history,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_errors,
    plot_weights,
)


def train_sklearn_SAMME(X_train, y_train, X_test, y_test, sample_weights=None):
    """
    Written by:
    Name: Kevlyn Kadamala
    Student ID: 21236191
    Class: MSc AI

    Training the sklearn model using SAMME algorithm.

    Arguments:
        X_train: Training inputs split.
        y_train: Training targets split.
        X_test: Testing inputs split.
        y_test: Testing targets split.
        sample_weights: Sample weights to be used.
    """
    ada = SklearnAdaBoost(n_estimators=50, algorithm="SAMME")
    ada.fit(X_train, y_train, sample_weights)
    preds = ada.predict(X_test)
    return accuracy_score(y_test, preds)


def train_sklearn_SAMMER_R(X_train, y_train, X_test, y_test, sample_weights=None):
    """
    Written by:
    Name: Elita Menezes
    Student ID: 21237434
    Class: MSc DA

    Training the sklearn model using SAMME.R algorithm.

    Arguments:
        X_train: Training inputs split.
        y_train: Training targets split.
        X_test: Testing inputs split.
        y_test: Testing targets split.
        sample_weights: Sample weights to be used.
    """
    ada = SklearnAdaBoost(n_estimators=50, algorithm="SAMME.R")
    ada.fit(X_train, y_train, sample_weights)
    preds = ada.predict(X_test)
    return accuracy_score(y_test, preds)


def train_scratch(X_train, y_train, X_test, y_test, sample_weights=None):
    """
    Written by:
    Name: Kevlyn Kadamala
    Student ID: 21236191
    Class: MSc AI

    Training Adaboost from scratch.

    Arguments:
        X_train: Training inputs split.
        y_train: Training targets split.
        X_test: Testing inputs split.
        y_test: Testing targets split.
        sample_weights: Sample weights to be used.
    """
    ada = AdaBoostClassifierScratch()
    ada.fit(X_train, y_train, sample_weights)
    preds = ada.predict(X_test)
    return accuracy_score(y_test, preds)


def additional_visualisations(X, y, dataset_name):
    """
    Written by:
    Name: Kevlyn Kadamala
    Student ID: 21236191
    Class: MSc AI

    Plot confusion matrices and ROC Curves for comparison.

    Arguments:
        X: Inputs to the model.
        y: The target values.
        dataset_name: Name of the dataset.
    """
    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=100, test_size=int(len(X) / 3)
    )

    # Plot for Scratch.
    file_name = f"scratch_{dataset_name}"
    clf = AdaBoostClassifierScratch()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    plot_errors(clf.n_estimators, clf.errors, "adaboost", file_name)
    plot_weights(clf.n_estimators, clf.alphas, "adaboost", file_name)
    plot_confusion_matrix(y_test, pred, clf.classes, "adaboost", file_name)
    if len(clf.classes) == 2:
        plot_roc_curve(y_test, pred, "adaboost", file_name)

    # Plot for Sklearn.
    file_name = f"sklearn_{dataset_name}"
    clf = SklearnAdaBoost(n_estimators=50, algorithm="SAMME")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    plot_errors(clf.n_estimators, clf.estimator_errors_, "adaboost", file_name)
    plot_weights(clf.n_estimators, clf.estimator_weights_, "adaboost", file_name)
    plot_confusion_matrix(y_test, pred, clf.classes_, "adaboost", file_name)
    if len(clf.classes_) == 2:
        plot_roc_curve(y_test, pred, "adaboost", file_name)


def test_adaboost(dataset):
    """
    Written by:
    Name: Kevlyn Kadamala
    Student ID: 21236191
    Class: MSc AI

    Test the Adaboost algorithm on the sklearn and scratch implementation.

    Arguments:
        dataset: The dataset to test on.
    """
    # Load data based on the dataset provided.
    dataset_name = dataset["dataset"]
    if dataset_name == "wildfire":
        df = pd.read_csv("data/wildfires.txt", sep="\t", header=0)
        X = df.drop(columns=["fire"])
        y = df.loc[:, "fire"]
        y = y.apply(lambda x: x.strip())
    elif dataset_name == "iris":
        X, y = load_iris(return_X_y=True, as_frame=True)
    elif dataset_name == "wine":
        X, y = load_wine(return_X_y=True, as_frame=True)

    # Generate accuracy histories.
    history = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=i, test_size=int(len(X) / 3)
        )

        # Train models and add to history.
        sklearn_acc_samme_R = train_sklearn_SAMMER_R(X_train, y_train, X_test, y_test)
        sklearn_acc_samme = train_sklearn_SAMME(X_train, y_train, X_test, y_test)
        scratch_acc = train_scratch(X_train, y_train, X_test, y_test)
        history.append((sklearn_acc_samme_R, sklearn_acc_samme, scratch_acc))

    # Format data and plot history.
    history = np.round(history, 4)
    history_mean = np.round(history.mean(axis=0), 4)
    data = np.concatenate(
        (
            history,
            [history_mean],
        ),
        axis=0,
    )
    data = pd.DataFrame(
        data, columns=["Sklearn (SAMME.R)", "Sklearn (SAMME)", "Scratch"]
    )
    data.insert(0, "Run", [i + 1 if i < 10 else "<b>Mean</b>" for i in range(11)])
    plot_history(data, "adaboost", dataset_name)

    # Visualise the scratch implementation.
    additional_visualisations(X, y, dataset_name)


if __name__ == "__main__":
    # Get dataset name from command line
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--dataset",
        default="wildfire",
        help="Dataset name",
        choices=["wildfire", "iris", "wine"],
    )
    args = vars(ap.parse_args())
    test_adaboost(args)
