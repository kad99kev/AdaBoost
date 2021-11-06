import numpy as np
import pandas as pd
import streamlit as st
from adaboost import A

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
    return y, classes


if __name__ == '__main__':
    st.title('Adaboost')

    # reading the files
    uploaded_file = st.file_uploader("Add a csv or txt file")
    if uploaded_file:
        X, y = read_files(uploaded_file)
        y, classes = convert_y(y)    

    st.subheader('Adaboost from scratch')
    # using our model to predict 

    
    st.subheader('Adaboost from sklearn')