import numpy as np
import sklearn.metrics as m
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
"""
    Return Values:
        Good ( g ): 1
        Bad ( b ):  0
"""


def load_dataset():
    return np.loadtxt('ionosphere.data', delimiter=',')


def model_metrics(model, y_test, name):
    f1 = m.f1_score(y_test, model)
    recall = m.recall_score(y_test, model)
    accuracy = m.accuracy_score(y_test, model)
    precision = m.precision_score(y_test, model)
    tn, fp, fn, tp = m.confusion_matrix(y_test, model).ravel()
    specificity = tn/float(tn+fp)
    print("\n\t\tCONFUSION MATRIX FOR " + name)
    print("\t\t\t\t\t\t  Predicted Values")
    print("\t\t\t\t\t\tNegative\tPositive")
    print("Actual\t\tNegative\t{0}\t\t{1}".format(tn,fp))
    print("Values\t\tPositive\t{0}\t\t{1}".format(fn,tp))
    print("\nF1-score: {0}\nRecall: {1}\nAccuracy: {2}\nPrecision: {3}\nSpecificity: {4}".format(f1,recall,accuracy,precision,specificity))


def data_scaler(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,
                                                        random_state=0,
                                                        stratify=y)
    scaler = StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
