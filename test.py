import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree import DecisionTreeClassifier as DTC
from sklearn.tree import DecisionTreeClassifier as SKDTC
from sklearn.tree import plot_tree


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight


def train_scratch(X_train, y_train, X_test, y_test, sample_weights=None):
    dt = DTC()
    dt.fit(X_train, y_train, sample_weights)
    preds = dt.predict(X_test)
    print(accuracy_score(y_test, preds))


def train_sklearn(X_train, y_train, X_test, y_test, sample_weights=None):
    dt = SKDTC(max_depth=1, max_leaf_nodes=2)
    dt.fit(X_train, y_train, sample_weight=sample_weights)
    preds = dt.predict(X_test)
    print(accuracy_score(y_test, preds))
    fig = plt.figure(figsize=(10, 5))
    _ = plot_tree(
        dt,
        feature_names=X_train.columns,
        class_names=["no", "yes"],
        filled=True,
        fontsize=10,
    )
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("wildfires.txt", sep="\t", header=0)
    X = df.drop(columns=["fire"])
    y = df.loc[:, "fire"]
    y = y.apply(lambda x: x.strip())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=int(len(X) / 3)
    )

    np.random.seed(0)
    # sample_weights = np.arange(len(X_train)) + 1
    # print(sample_weights)
    sample_weights = compute_sample_weight("balanced", y_train)

    # X = [[0], [1], [2]]  # 3 simple training examples
    # Y = np.array([1, 2, 1])  # class labels
    # sample_weights = np.array([1, 2, 3])
    # classes = np.unique(Y)
    # print(Y[Y == classes[0]])
    # print(
    #     1
    #     - sum(
    #         (sample_weights[Y == c].sum() / sample_weights.sum()) ** 2 for c in classes
    #     )
    # )

    train_scratch(X_train, y_train, X_test, y_test, sample_weights)
    train_sklearn(X_train, y_train, X_test, y_test, sample_weights)
    # train_sklearn(X_train, y_train, X_test, y_test)
