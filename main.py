"""
Written by:
Name: Elita Menezes
Student ID: 21237434
Class: MSc DA
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from adaboost import AdaBoostClassifierScratch
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost



def get_data(filename, target_index, sep = ','):
    """
    Stores data in a dataframe
    Uses the index to find the target index (y)
    Seperates by seperator

    Arguments:
        filename: path to the file
        target_index: index of y or target values
        sep: seperator used to seperate the csv files (can be comma or tab seperated values)
    """
    df = pd.read_csv(filename, sep=sep, header=0)
    X = df.drop(df.columns[target_index], axis=1)
    y = df.iloc[:, target_index]
    y = y.apply(lambda x: x.strip())
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=int(len(X) /3))
    return X_train, X_test, y_train, y_test
    

def save_output(filename, y_true, scratch, sklearn):
    """
    Saves the true, scratch pred and sklearn pred contents to a csv file

    Arguments:
        filename: name of the output file
        y_true: true values
        scratch: values predicted by scratch implementation
        sklearn: values predicted by sklearn implementation
    """
    data = {'True Values' : y_true, 'Scratch Pred': scratch, 'Sklearn Pred': sklearn}
    df = pd.DataFrame(data)
    df.to_csv('./outputs/' + filename, index = False)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data(filename = './data/wildfires.txt', target_index = 0, sep = '\t')

    # training and testing AdaBoost Scratch classifier
    print("AdaBoost from Scratch")
    clf_scratch = AdaBoostClassifierScratch(n_estimators=50, learning_rate=0.05)
    clf_scratch.fit(X_train, y_train)
    scratch_pred = clf_scratch.predict(X_test)
    print(f"Accuracy from Scratch Implementation: {round(accuracy_score(y_test, scratch_pred), 4)}")

    print('\n')
    # training and testing AdaBoost from Sklearn
    print("AdaBoost from Sklearn")
    clf_sklearn = SklearnAdaBoost(n_estimators=50, learning_rate=0.05)
    clf_sklearn.fit(X_train, y_train)
    sklearn_pred = clf_sklearn.predict(X_test)
    print(f"Accuracy from Scratch Implementation: {round(accuracy_score(y_test, sklearn_pred), 4)}")

    print('\n')
    save_output(filename = 'wildfire.csv', y_true = y_test, scratch = scratch_pred, sklearn = sklearn_pred)
    print("File saved to outputs folder")