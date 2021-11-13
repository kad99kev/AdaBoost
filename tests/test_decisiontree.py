"""
Written by:
Name: Kevlyn Kadamala
Student ID: 21236191
Class: MSc AI
"""

import argparse
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

from adaboost.tree import DecisionTreeClassifierScratch

from .plot_tests import plot_history, plot_confusion_matrix, plot_roc_curve


def train_scratch_dt(X_train, y_train, X_test, sample_weights=None, return_clf=False):
    """
    Training Decision Tree from scratch.

    Arguments:
        X_train: Training inputs split.
        y_train: Training targets split.
        X_test: Testing inputs split.
        sample_weights: Sample weights to be used.
    """
    dt = DecisionTreeClassifierScratch(max_depth=1)
    dt.fit(X_train, y_train, sample_weights)
    preds = dt.predict(X_test)
    if return_clf:
        return preds, dt
    return preds


def train_sklearn_dt(X_train, y_train, X_test, sample_weights=None, return_clf=False):
    """
    Training the sklearn model.

    Arguments:
        X_train: Training inputs split.
        y_train: Training targets split.
        X_test: Testing inputs split.
        sample_weights: Sample weights to be used.
    """
    dt = SklearnDecisionTree(max_depth=1)
    dt.fit(X_train, y_train, sample_weight=sample_weights)
    preds = dt.predict(X_test)
    if return_clf:
        return preds, dt
    return preds


def run_plots(y_test, preds, classes, file_name):
    """
    Plots the confusion matrix and the ROC curve.

    Arguments:
        y_test: Testing targets split.
        preds: Predictions from classifier.
        classes: The classes identified by the classifier.
        file_name: File name for saving the image.
    """
    plot_confusion_matrix(y_test, preds, classes, "cart", file_name)
    if len(classes) == 2:
        plot_roc_curve(y_test, preds, "cart", file_name)


def additional_visualisations(X, y, dataset_name):
    """
    Plot confusion matrices and ROC Curves for comparison.

    Arguments:
        X: Inputs to the model.
        y: The target values.
        dataset_name: Name of the dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=100, test_size=int(len(X) / 3)
    )

    scratch_preds, scratch_clf = train_scratch_dt(
        X_train, y_train, X_test, return_clf=True
    )
    run_plots(y_test, scratch_preds, scratch_clf.classes, f"scratch_{dataset_name}")

    sklearn_preds, sklearn_clf = train_sklearn_dt(
        X_train, y_train, X_test, return_clf=True
    )
    run_plots(y_test, sklearn_preds, sklearn_clf.classes_, f"sklearn_{dataset_name}")


def test_decisiontree(dataset):
    """
    Test the CART algorithm on the sklearn and scratch implementation.

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
        sklearn_preds = train_sklearn_dt(X_train, y_train, X_test)
        scratch_preds = train_scratch_dt(X_train, y_train, X_test)
        history.append(
            (
                accuracy_score(y_test, sklearn_preds),
                accuracy_score(y_test, scratch_preds),
            )
        )

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
    data = pd.DataFrame(data, columns=["Sklearn", "Scratch"])
    data.insert(0, "Run", [i + 1 if i < 10 else "<b>Mean</b>" for i in range(11)])
    plot_history(data, "cart", dataset_name)

    # Create additional results.
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
    test_decisiontree(args)
