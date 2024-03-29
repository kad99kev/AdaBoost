"""
Written by:
Name: Elita Menezes
Student ID: 21237434
Class: MSc DA
"""

import pandas as pd
import streamlit as st

from adaboost import AdaBoostClassifierScratch
from sklearn.datasets import load_iris, load_wine
from adaboost.viz import plt_roc_curve, plt_confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier as classifier


def read_files(file):
    """
    Reading files using pandas read_csv.
    Saves the first column as y and the remaining columns as X.

    Arguments:
        file: Path to the file.
    """
    df = pd.read_csv(file, sep="\t", header=0)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    y = y.apply(lambda x: x.strip())
    return X, y


def split_data(X, y):
    """
    Splits the dataset into train and test.

    Arguments:
        X: Inputs values.
        y: Tagret values.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=100, test_size=int(len(X) / 3)
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    st.title("Adaboost")

    st.markdown("### Upload your dataset")

    # Reads the user input files
    uploaded_file = st.file_uploader("Add a csv or txt file")
    if uploaded_file:
        X, y = read_files(uploaded_file)
        # Splits the data into test and train

    st.markdown("### Or use an existing dataset")
    button1, button2, button3 = st.columns(3)
    with button1:
        if st.button("Wildfires Dataset"):
            df = pd.read_csv("./data/wildfires.txt", sep="\t", header=0)
            X = df.iloc[:, 1:]
            y = df.iloc[:, 0]
            y = y.apply(lambda x: x.strip())
    with button2:
        if st.button("Iris Dataset"):
            X, y = load_iris(return_X_y=True, as_frame=True)
    with button3:
        if st.button("Wine Dataset"):
            X, y = load_wine(return_X_y=True, as_frame=True)

    try:
        classes = y.unique()
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Predicts using the model from scratch
        st.subheader("Adaboost from scratch")
        clf_scratch = AdaBoostClassifierScratch(n_estimators=50, learning_rate=0.05)
        clf_scratch.fit(X_train, y_train)
        pred = clf_scratch.predict(X_test)
        st.markdown(f"##### Accuracy: {round(accuracy_score(y_test, pred), 4)}")
        if len(classes) == 2:
            dc_col11, dc_col12 = st.columns(2)

            with dc_col11:
                # Plotting the roc curve
                fig = plt_roc_curve(y_test, pred)
                st.plotly_chart(fig)

            with dc_col12:
                # Plotting the confusion matrix
                fig = plt_confusion_matrix(y_test, pred, classes)
                st.plotly_chart(fig)
        else:
            fig = plt_confusion_matrix(y_test, pred, classes)
            st.plotly_chart(fig)

            # Predicts using the sklearn model
        st.subheader("Adaboost from sklearn (SAMME Algorithm)")
        clf_sklearn = classifier(n_estimators=50, learning_rate=0.05, algorithm="SAMME")
        clf_sklearn.fit(X_train, y_train)
        sklearn_preds = clf_sklearn.predict(X_test)
        st.markdown(
            f"##### Accuracy: {round(accuracy_score(y_test, sklearn_preds), 4)}"
        )

        if len(classes) == 2:
            dc_col21, dc_col22 = st.columns(2)

            with dc_col21:
                # Plotting the roc curve
                fig = plt_roc_curve(y_test, sklearn_preds)
                st.plotly_chart(fig)

            with dc_col22:
                # Plotting the confusion matrix
                fig = plt_confusion_matrix(y_test, sklearn_preds, classes)
                st.plotly_chart(fig)
        else:
            fig = plt_confusion_matrix(y_test, sklearn_preds, classes)
            st.plotly_chart(fig)

        # Predicts using the sklearn model SAMME.R
        st.subheader("Adaboost from sklearn (SAMME.R Algorithm)")
        clf_sklearn = classifier(
            n_estimators=50, learning_rate=0.05, algorithm="SAMME.R"
        )
        clf_sklearn.fit(X_train, y_train)
        sklearn_preds = clf_sklearn.predict(X_test)
        st.markdown(
            f"##### Accuracy: {round(accuracy_score(y_test, sklearn_preds), 4)}"
        )

        if len(classes) == 2:
            dc_col21, dc_col22 = st.columns(2)

            with dc_col21:
                # Plotting the roc curve
                fig = plt_roc_curve(y_test, sklearn_preds)
                st.plotly_chart(fig)

            with dc_col22:
                # Plotting the confusion matrix
                fig = plt_confusion_matrix(y_test, sklearn_preds, classes)
                st.plotly_chart(fig)
        else:
            fig = plt_confusion_matrix(y_test, sklearn_preds, classes)
            st.plotly_chart(fig)
    except Exception as e:
        print(e)
