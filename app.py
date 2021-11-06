
import pandas as pd
import streamlit as st

from adaboost import AdaBoostClassifier
from adaboost.viz import plt_roc_curve, plt_confusion_matrix

from sklearn.ensemble import AdaBoostClassifier as classifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def read_files(file):
    """
    Reading files using pandas read_csv.
    Saves the first column as y and the remaining columns as X.

    Arguments:
        file: Path to the file.
    """
    df = pd.read_csv(file, sep="\t", header=0)
    X = df.iloc[: , 1:]
    y = df.iloc[:, 0]
    y = y.apply(lambda x: x.strip())
    return X, y

def convert_y(y):
    """
    Converts y variable to {-1, 1}.
    Operation suitable for binary classification only.

    Arguments:
        y: list of elements having binary class.
    """
    # find unique values
    classes = y.unique()
    # assign -1 to one variable and 1 to another
    y[y == classes[0]] = -1
    y[y == classes[1]] = 1
    y = y.astype(str).astype(int)
    return y, classes

def split_data(X, y):
    """
    Splits the dataset into train and test.

    Arguments:
        X: Inputs values.
        y: Tagret values.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=int(len(X) / 3)
    )
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    st.title('Adaboost')

    # Reads the user input files
    uploaded_file = st.file_uploader("Add a csv or txt file")
    if uploaded_file:
        X, y = read_files(uploaded_file)
        y, classes = convert_y(y)    

        # Splits the data into test and train
        X_train, X_test, y_train, y_test = split_data(X, y)


        # Predicts using the model from scratch
        st.subheader('Adaboost from scratch')
        clf_scratch = AdaBoostClassifier(n_estimators = 100, learning_rate=0.05)
        clf_scratch.fit(X_train, y_train)
        pred = clf_scratch.predict(X_test)
        scratch_preds = pd.Series((p[0] for p in pred))
        st.markdown(f"##### Accuracy: {round(accuracy_score(y_test, scratch_preds), 4)}")

        dc_col11, dc_col12 = st.columns(2)

        with dc_col11:
            # Plotting the roc curve
            plt_roc_curve(y_test, scratch_preds)
            
        with dc_col12:
            # Plotting the confusion matrix
            plt_confusion_matrix(y_test, scratch_preds, classes)

        # Predicts using the sklearn model
        st.subheader('Adaboost from sklearn')
        clf_sklearn = classifier(n_estimators = 100, learning_rate=0.05)
        clf_sklearn.fit(X_train, y_train)
        sklearn_preds = clf_sklearn.predict(X_test)
        st.markdown(f"##### Accuracy: {round(accuracy_score(y_test, sklearn_preds), 4)}")

        dc_col21, dc_col22 = st.columns(2)

        with dc_col21:
            # Plotting the roc curve
            plt_roc_curve(y_test, sklearn_preds)
            
        with dc_col22:
            # Plotting the confusion matrix
            plt_confusion_matrix(y_test, sklearn_preds, classes)
