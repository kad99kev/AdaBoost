
import pandas as pd
import streamlit as st

from adaboost import AdaBoostClassifier
from adaboost.viz import plt_roc_curve, plt_confusion_matrix

from sklearn.ensemble import AdaBoostClassifier as classifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def read_files(filename):
    """
    Reading files using pandas read_csv
    Saving the last column as y and the remaining columns as X
    """
    df = pd.read_csv(filename, sep="\t", header=0)
    X = df.iloc[: , 1:]
    y = df.iloc[:, 0]
    y = y.apply(lambda x: x.strip())
    return X, y

def convert_y(y):
    """
    Converts y variable to -1 to 1
    Suitable for binary data only
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
    Splitting the dataset into train and test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=int(len(X) / 3)
    )
    return X_train, X_test, y_train, y_test

def plt(y_test, preds, classes):
    """
    Getting the accuracy of the data
    Getting the roc curves
    Plotting the confusion matrix
    """
    st.markdown(f"##### Accuracy: {round(accuracy_score(y_test, preds), 4)}")
    plt_roc_curve(y_test, preds)
    plt_confusion_matrix(y_test, preds, classes)


if __name__ == '__main__':
    st.title('Adaboost')

    # reading the files
    uploaded_file = st.file_uploader("Add a csv or txt file")
    if uploaded_file:
        X, y = read_files(uploaded_file)
        y, classes = convert_y(y)    

        # splitting the data into 2/3 train and 1/3 test
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Adaboost from scratch
        st.subheader('Adaboost from scratch')
        clf_scratch = AdaBoostClassifier(n_estimators = 100, learning_rate=0.05)
        clf_scratch.fit(X_train, y_train)
        pred = clf_scratch.predict(X_test)
        preds = pd.Series((p[0] for p in pred))
        plt(y_test, preds, classes)
        
        st.subheader('Adaboost from sklearn')
        clf_sklearn = classifier(n_estimators = 100, learning_rate=0.05, algorithm='SAMME')
        clf_sklearn.fit(X_train, y_train)
        preds = clf_sklearn.predict(X_test)
        plt(y_test, preds, classes)