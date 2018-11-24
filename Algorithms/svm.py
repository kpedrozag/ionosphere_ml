from functions import data_scaler, model_metrics
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def SVM_kernels(X, y, kernel):
    # Both X_train, X_test are scaled
    X_train, X_test, y_train, y_test = data_scaler(X, y)
    param = []
    clf_list = []
    if kernel == 'linear':
        param = [10**i for i in range(-3, 4)]
        clf_list = [SVC(C=i,
                        kernel='linear',
                        max_iter=1000,
                        random_state=0).fit(X_train, y_train) for i in param]
    elif kernel == 'poly':
        param = [2, 5, 9, 13, 16]
        clf_list = [SVC(kernel='poly',
                        random_state=0,
                        max_iter=1000,
                        degree=i).fit(X_train, y_train) for i in param]
    elif kernel == 'rbf':
        param = [10**i for i in range(-3, 3)]
        clf_list = [SVC(kernel='rbf',
                        random_state=0,
                        max_iter=1000,
                        gamma=i).fit(X_train, y_train) for i in param]

    score = [cross_val_score(st, X_train, y_train, cv=30, n_jobs=-1).mean() for st in clf_list]
    pos_max_score = np.argmax(score)

    best_param = param[pos_max_score]
    name = ''
    if kernel == 'linear':
        clf = SVC(C=best_param,
                  kernel='linear',
                  max_iter=1000,
                  random_state=0).fit(X_train, y_train)
    elif kernel == 'poly':
        clf = SVC(kernel='poly',
                  random_state=0,
                  max_iter=1000,
                  degree=best_param).fit(X_train, y_train)
    elif kernel == 'rbf':
        clf = SVC(kernel='rbf',
                  random_state=0,
                  max_iter=1000,
                  gamma=best_param).fit(X_train, y_train)

    model = clf.predict(X_test)
    model_metrics(model, y_test, 'SVM MODEL WITH ' + kernel.upper() + ' KERNEL')
