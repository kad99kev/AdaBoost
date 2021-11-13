"""
Written by:
Name: Kevlyn Kadamala
Student ID: 21236191
Class: MSc AI
"""

import plotly.graph_objects as go
import plotly.figure_factory as ff


def plot_history(data, classifier, name):
    """
    Plots the histories of the test runs.

    Arguments:
        data: History data.
        classifier: Name of classifier.
        name: Name of the file.
    """
    fig = ff.create_table(data)
    fig.write_image(f"images/{classifier}/history/{name}.png")


def plot_errors(n_estimators, estimator_errors, classifier, name):
    """
    Plotting the error rate for each estimator.

    Arguments:
        n_estimators: Number of estimators.
        estimator_errors: Training errors for each estimator.
        classifier: Name of classifier.
        name: Name of the file.
    """

    fig = go.Figure(
        data=go.Scatter(x=[i for i in range(n_estimators)], y=estimator_errors)
    )
    fig.update_layout(
        title="Error rates for each estimator",
        xaxis_title="Estimator",
        yaxis_title="Error",
    )
    fig.write_image(f"images/{classifier}/estimator_errors/{name}.png")


def plot_weights(n_estimators, estimator_weights, classifier, name):
    """
    Plotting the weights for each estimator.

    Arguments:
        n_estimators: Number of estimators.
        estimator_weights: Weights for each estimator.
        classifier: Name of classifier.
        name: Name of the file.
    """

    fig = go.Figure(
        data=go.Scatter(x=[i for i in range(n_estimators)], y=estimator_weights)
    )
    fig.update_layout(
        title="Weights for each estimator",
        xaxis_title="Estimator",
        yaxis_title="Weight",
    )
    fig.write_image(f"images/{classifier}/estimator_weights/{name}.png")
