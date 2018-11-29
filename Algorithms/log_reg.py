from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from functions import model_metrics, data_scaler
import numpy as np


def LogRegCv(X, y, scal):
    X_train, X_test, y_train, y_test = data_scaler(X, y, scal)
    c = [10**i for i in range(-3, 4)]
    clf_list = [LogisticRegression(C=i, solver="liblinear", max_iter=1000).fit(X_train, y_train) for i in c]
    score = [cross_val_score(i, X_train, y_train, cv=10).mean() for i in clf_list]
    pos_max_score = np.argmax(score)
    penalty = c[pos_max_score]
    clf = LogisticRegression(C=penalty, solver="liblinear", max_iter=1000).fit(X_train, y_train)
    model = clf.predict(X_test)
    k = ''
    if scal:
        k = ' (DATA SCALED)'
    model_metrics(model, y_test, 'LOGISTIC REGRESSION MODEL' + k)
