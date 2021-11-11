import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.metrics import roc_curve, auc, confusion_matrix


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


def plot_roc_curve(y_test, pred, classifier, name):
    """
    Plots the roc curves for the test and pred.
    Reference: https://plotly.com/python/roc-and-pr-curves/
    Only suitable for binary class labels

    Arguements:
        y_test: list of y true values.
        pred: list of y pred values.
        classifier: Name of classifier.
        name: Name of the file.
    """
    assert len(y_test) == len(pred), "Length of y_test and pred should be equal"
    # sklearn roc curve
    y_test = np.array([1 if v == "yes" else -1 for v in y_test])
    pred = np.array([1 if v == "yes" else -1 for v in pred])
    fpr, tpr, _ = roc_curve(y_test, pred)

    # plotting graph using plotly.express
    fig = px.area(
        x=fpr,
        y=tpr,
        title=f"ROC Curve (AUC={auc(fpr, tpr):.4f})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
        width=450,
        height=450,
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")
    fig.write_image(f"images/{classifier}/roc_curve/{name}.png")


def plot_confusion_matrix(y_test, pred, classes, classifier, name):
    """
    Plots the confusion matrix for the true and predicted values.
    Reference: https://stackoverflow.com/questions/60860121/plotly-how-to-make-an-annotated-confusion-matrix-using-a-heatmap

    Arguments:
        y_test: list of y true values.
        pred: list of predicted values.
        classes: list of labels.
        classifier: Name of classifier.
        name: Name of the file.
    """
    confusion_mat = confusion_matrix(y_test, pred).tolist()
    x = classes.tolist()
    y = classes.tolist()

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in confusion_mat]

    # set up figure
    fig = ff.create_annotated_heatmap(
        confusion_mat, x=x, y=y, annotation_text=z_text, colorscale="Reds"
    )

    # add title
    fig.update_layout(title_text="<i><b>Confusion matrix</b></i>")

    # add custom xaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Predicted value",
            xref="paper",
            yref="paper",
        )
    )

    # add custom yaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=-0.35,
            y=0.5,
            showarrow=False,
            text="Real value",
            textangle=-90,
            xref="paper",
            yref="paper",
        )
    )

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig["data"][0]["showscale"] = True
    fig.write_image(f"images/{classifier}/confusion_matrix/{name}.png")


def plot_error_rates(n_estimators, training_errors, classifier, name):
    """
    Plotting the error rate for each iteration.

    Arguments:
        n_estimators: Number of estimators.
        training_errors: Training errors at each iteration.
        classifier: Name of classifier.
        name: Name of the file.
    """

    fig = go.Figure(
        data=go.Scatter(x=[i for i in range(n_estimators)], y=training_errors)
    )
    fig.update_layout(
        title="Error rates for each iteration",
        xaxis_title="Iteration",
        yaxis_title="Error",
    )
    fig.write_image(f"images/{classifier}/error_rate/{name}.png")
