import numpy as np
import sklearn.metrics as m
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys

sys.path.append(os.path.abspath('./Algorithms'))
sys.path.append(os.path.abspath('..'))
"""
    Return Values:
        Good ( g ): 1
        Bad ( b ):  0
"""


def load_dataset():
    """
    Read the data file and return a numpy array
    :return: numpy.ndarray
    """
    file = np.loadtxt(fname='ionosphere.data.txt', delimiter=',')

    return file[:, :33], file[:, 34]


def model_metrics(model, y_test, name):
    """
    Metricas del modelo pasado como argumento.

    Calcula los puntajes de F1-Score, Recall, Exactitud, Precision, y Espcificidad,
    ademas de mostrar la matriz de confusion.
    """
    f1 = m.f1_score(y_test, model)
    recall = m.recall_score(y_test, model)
    accuracy = m.accuracy_score(y_test, model)
    precision = m.precision_score(y_test, model)
    tn, fp, fn, tp = m.confusion_matrix(y_test, model).ravel()
    specificity = tn/float(tn+fp)

    print("\n\t\tCONFUSION MATRIX FOR " + name, file=fl)
    print("\t\t\t\t\t\t  Predicted Values", file=fl)
    print("\t\t\t\t\t\tNegative\tPositive", file=fl)
    print("Actual\t\tNegative\t{0}\t\t{1}".format(tn,fp), file=fl)
    print("Values\t\tPositive\t{0}\t\t{1}".format(fn,tp), file=fl)
    print("\nF1-score: {0}\nRecall: {1}\nAccuracy: {2}\nPrecision: {3}\nSpecificity: {4}".format(f1, recall, accuracy, precision, specificity), file=fl)
    avg_met = f1 + recall + accuracy + precision + specificity
    print("Average Metrics", avg_met, file=fl)


def data_scaler(X, y, scal):
    """Divide los datos de test y train de cada parametro, los estratifica y normaliza

    :param X: Matriz de features
    :param y: Vector de outputs
    :param scal: Si es True, se normalizan los sets
    :return: Tupla de tests y trains
    """
    if scal:
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.43,
                                                            random_state=0,
                                                            stratify=y)
        scaler = StandardScaler().fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.43,
                                                            random_state=0,
                                                            stratify=y)
    return X_train, X_test, y_train, y_test
