# AdaBoost

## How to run the files

**main.py**

Takes file as input and stores the ground truth, scratch and sklearn predictions in a csv file.
```
python main.py
```

**app.py**

GUI for the adaboost implementation. Takes a file as an input and outputs the accuracy, ROC curves (Binary classes) and the confusion matrix for the scratch and sklearn (SAMME and SAMME.R) implementations.
```
streamlit run app.py
```

**tests**

Compares the Decision Tree and Adaboost implementations for 10 iterations. Also plots the graphs like confusion matrix, estimator errors, estimator weights, history and roc curves. 
You can also choose an existing dataset to test the algorithm implementation.
```
python -m tests.test_adaboost --dataset [dataset_name]
```
The options are: `[wildfire, iris, wine]`.

The default option is `wildfire`.


