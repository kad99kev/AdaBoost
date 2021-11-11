"""
Written by:
Name: Elita Menezes
Student ID: 21237434
Class: MSc DA
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plt_roc_curve(y_test, pred):
    """
    Plots the roc curves for the test and pred.
    Reference: https://plotly.com/python/roc-and-pr-curves/
    Only suitable for binary class labels

    Arguements:
        y_test: list of y true values.
        y_pred: list of y pred values.
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
    return fig


def plt_confusion_matrix(y_test, pred, classes):
    """
    Plots the confusion matrix for the true and predicted values.
    Reference: https://stackoverflow.com/questions/60860121/plotly-how-to-make-an-annotated-confusion-matrix-using-a-heatmap

    Arguments:
        y_test: list of y true values.
        pred: list of predicted values.
        classes: list of labels.
    """
    confusion_mat = confusion_matrix(y_test, pred).tolist()
    x = classes.tolist()
    y = classes.tolist()

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in confusion_mat]

    # set up figure 
    fig = ff.create_annotated_heatmap(confusion_mat, x = x, y = y, annotation_text = z_text, colorscale='Reds')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>')

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    return fig