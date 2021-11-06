import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tree import DecisionTreeClassifier as DTC

from adaboost import AdaBoost
from sklearn.tree import DecisionTreeClassifier as SKDTC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import plot_tree


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight


def plot_tree_test(clf, X):
    fig = plt.figure(figsize=(10, 5))
    _ = plot_tree(
        clf,
        feature_names=X.columns,
        class_names=["no", "yes"],
        filled=True,
        fontsize=10,
    )
    plt.show()

def plot_roc_curve(y_test, pred):
    """
    Plotting roc curve
    Ref: https://stackoverflow.com/questions/43043271/roc-curve-for-binary-classification-in-python
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test, pred)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    plt.plot(fpr[1], tpr[1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.show()

MAX_DEPTH = 4


def train_scratch_dt(X_train, y_train, X_test, y_test, sample_weights=None):
    dt = DTC(max_depth=MAX_DEPTH)
    dt.fit(X_train, y_train, sample_weights)
    preds = dt.predict(X_test)
    dt.plot_tree()
    return accuracy_score(y_test, preds)


def train_sklearn_dt(X_train, y_train, X_test, y_test, sample_weights=None, viz=False):
    dt = SKDTC(max_depth=MAX_DEPTH)
    dt.fit(X_train, y_train, sample_weight=sample_weights)
    print(dt)
    preds = dt.predict(X_test)
    if viz:
        plot_tree_test(dt, X)
    return accuracy_score(y_test, preds)
    


def train_sklearn_ada(X_train, y_train, X_test, y_test, sample_weights=None):
    ada = AdaBoostClassifier(n_estimators=50, algorithm="SAMME")
    ada.fit(X_train, y_train, sample_weights)
    preds = ada.predict(X_test)
    plot_roc_curve(y_test, preds)
    return accuracy_score(y_test, preds)


def train_scratch_ada(X_train, y_train, X_test, y_test, sample_weights=None):
    ada = AdaBoost()
    ada.fit(X_train, y_train, sample_weights)
    preds = ada.predict(X_test)
    plot_roc_curve(y_test, preds)
    return accuracy_score(y_test, preds)




if __name__ == "__main__":
    df = pd.read_csv("wildfires.txt", sep="\t", header=0)
    X = df.drop(columns=["fire"])
    y = df.loc[:, "fire"]
    y = y.apply(lambda x: x.strip())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=int(len(X) / 3)
    )

    y_train = np.array([1 if v == "yes" else -1 for v in y_train])
    y_test = np.array([1 if v == "yes" else -1 for v in y_test])

    np.random.seed(0)
    # sample_weights = np.arange(len(X_train)) + 1
    # print(sample_weights)
    sample_weights = compute_sample_weight("balanced", y_train)

    print(train_scratch_dt(X_train, y_train, X_test, y_test))
    print(train_sklearn_dt(X_train, y_train, X_test, y_test, viz=True))
    assert train_sklearn_ada(X_train, y_train, X_test, y_test) == train_scratch_ada(
        X_train, y_train, X_test, y_test
    )

    
    print("All tests passed!!!!")



