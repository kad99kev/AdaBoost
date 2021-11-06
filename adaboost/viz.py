from numpy.core.fromnumeric import trace
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix

def plt_roc_curve(y_test, pred):
    # https://plotly.com/python/roc-and-pr-curves/
    fpr, tpr, _ = roc_curve(y_test, pred)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=450, height=450
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    st.plotly_chart(fig)

def plt_confusion_matrix(y_test, pred, classes):
    # https://stackoverflow.com/questions/60860121/plotly-how-to-make-an-annotated-confusion-matrix-using-a-heatmap
    confusion_mat = confusion_matrix(y_test, pred)
    print(confusion_mat)
    tp, fp, fn, tn = confusion_mat.ravel()
    print(tp, fp, fn, tn)

    data = go.Heatmap(z = [[fn, tn], [tp, fp]], y = classes[::-1], x = classes, colorscale = "Reds")
    annotations = []
    for i, row in enumerate(confusion_mat.T):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": classes[i],
                    "y": classes[j],
                    "font": {"color": "black", "size" : 18},
                    "text": str(value),
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                }
            )
    layout = {
        "title": "Confusion matrix",
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
        "annotations": annotations
    }
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(width = 450, height = 450)
    st.plotly_chart(fig)

