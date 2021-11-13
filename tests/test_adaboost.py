import argparse
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost

from adaboost import AdaBoostClassifierScratch
from adaboost.viz import plt_confusion_matrix, plt_roc_curve
from .plot_tests import (
    plot_history,
    plot_errors,
    plot_weights,
)


def train_sklearn_SAMME(
    X_train, y_train, X_test, sample_weights=None, return_clf=False
):
    """
    Written by:
    Name: Elita Menezes
    Student ID: 21237434
    Class: MSc DA

    Training the sklearn model using SAMME algorithm.

    Arguments:
        X_train: Training inputs split.
        y_train: Training targets split.
        X_test: Testing inputs split.
        sample_weights: Sample weights to be used.
    """
    ada = SklearnAdaBoost(n_estimators=50, algorithm="SAMME")
    ada.fit(X_train, y_train, sample_weights)
    preds = ada.predict(X_test)
    if return_clf:
        return preds, ada
    return preds


def train_sklearn_SAMMER(
    X_train, y_train, X_test, sample_weights=None, return_clf=False
):
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
        sample_weights: Sample weights to be used.
    """
    ada = SklearnAdaBoost(n_estimators=50, algorithm="SAMME.R")
    ada.fit(X_train, y_train, sample_weights)
    preds = ada.predict(X_test)
    if return_clf:
        return preds, ada
    return preds


def train_scratch(X_train, y_train, X_test, sample_weights=None, return_clf=False):
    """
    Written by:
    Name: Elita Menezes
    Student ID: 21237434
    Class: MSc DA

    Training Adaboost from scratch.

    Arguments:
        X_train: Training inputs split.
        y_train: Training targets split.
        X_test: Testing inputs split.
        sample_weights: Sample weights to be used.
    """
    ada = AdaBoostClassifierScratch()
    ada.fit(X_train, y_train, sample_weights)
    preds = ada.predict(X_test)
    if return_clf:
        return preds, ada
    return preds


def run_plots(y_test, preds, classes, n_estimators, errors, alphas, file_name):
    """
    Written by:
    Name: Kevlyn Kadamala
    Student ID: 21236191
    Class: MSc AI

    Plots errors, weights, confusion matrix and the ROC curve.

    Arguments:
        y_test: Testing targets split.
        preds: Predictions from classifier.
        classes: The classes identified by the classifier.
        n_estimators: The number of estimators used by AdaBoost.
        errors: Training errors for each estimator.
        alphas: Weights for each estimator.
        file_name: File name for saving the image.
    """
    plot_errors(n_estimators, errors, "adaboost", file_name)
    plot_weights(n_estimators, alphas, "adaboost", file_name)
    conf_fig = plt_confusion_matrix(y_test, preds, classes)
    conf_fig.write_image(f"images/adaboost/confusion_matrix/{file_name}.png")
    if len(classes) == 2:
        roc_fig = plt_roc_curve(y_test, preds)
        roc_fig.write_image(f"images/adaboost/roc_curve/{file_name}.png")


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

    scratch_preds, scratch_clf = train_scratch(
        X_train, y_train, X_test, return_clf=True
    )
    run_plots(
        y_test,
        scratch_preds,
        scratch_clf.classes,
        scratch_clf.n_estimators,
        scratch_clf.errors,
        scratch_clf.alphas,
        file_name,
    )

    # Plot for Sklearn.
    file_name = f"sklearn_{dataset_name}"

    sklearn_preds, sklearn_clf = train_sklearn_SAMME(
        X_train, y_train, X_test, return_clf=True
    )
    run_plots(
        y_test,
        sklearn_preds,
        sklearn_clf.classes_,
        sklearn_clf.n_estimators,
        sklearn_clf.estimator_errors_,
        sklearn_clf.estimator_weights_,
        file_name,
    )


def test_adaboost(dataset):
    """
    Written by:
    Name: Elita Menezes
    Student ID: 21237434
    Class: MSc DA

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
        preds_sammer = train_sklearn_SAMMER(X_train, y_train, X_test)
        preds_samme = train_sklearn_SAMME(X_train, y_train, X_test)
        preds_scratch = train_scratch(X_train, y_train, X_test)
        history.append(
            (
                accuracy_score(y_test, preds_sammer),
                accuracy_score(y_test, preds_samme),
                accuracy_score(y_test, preds_scratch),
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
    data = pd.DataFrame(
        data, columns=["Sklearn (SAMME.R)", "Sklearn (SAMME)", "Scratch"]
    )
    data.insert(0, "Run", [i + 1 if i < 10 else "<b>Mean</b>" for i in range(11)])
    plot_history(data, "adaboost", dataset_name)

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
    test_adaboost(args)
