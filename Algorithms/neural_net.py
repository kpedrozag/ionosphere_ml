from functions import data_scaler, model_metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


def MultiLayerP(X, y, scal):
    X_train, X_test, y_train, y_test = data_scaler(X, y, scal)
    param = [10**i for i in range(-6, 4)]

    mis_scores = []

    real_scores = []
    for i in range(1, 5):
        aux = []
        layers = tuple([30 for x in range(i)])
        clf_list = [MLPClassifier(hidden_layer_sizes=layers, alpha=i, activation='relu', solver='lbfgs', random_state=0).fit(X_train, y_train) for i in param]
        score = [cross_val_score(i, X_train, y_train, cv=10).mean() for i in clf_list]
        mis_scores.append(score)
        pos_max_score = np.argmax(score)
        aux.append(pos_max_score)
        aux.append(score[pos_max_score])
        real_scores.append(aux)

    best_layer = real_scores[0][0]
    pos_best_layer = 0
    for i in range(len(real_scores)):
        if real_scores[i][1] > best_layer:
            best_layer = real_scores[i][1]
            pos_best_layer = i

    layer = tuple([30 for i in range(pos_best_layer+1)])
    pm = param[real_scores[pos_best_layer][0]]
    clf = MLPClassifier(hidden_layer_sizes=layer, alpha=pm, activation='relu', solver='lbfgs', random_state=0).fit(X_train, y_train)
    model = clf.predict(X_test)
    k = ''
    if scal:
        k = ' (DATA SCALED)'
    model_metrics(model, y_test, 'NEURAL NETWORKS MODEL WITH MULTI-LAYER PERCEPTRON METHOD' + k)

