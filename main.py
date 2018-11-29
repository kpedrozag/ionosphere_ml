from Algorithms import *
from functions import load_dataset


if __name__ == '__main__':
    X, y = load_dataset()
    cc = 0
    # Normalizado
    #LogRegCv(X, y, True)
    cc += 1
    print(cc)
    # No Normalizado
    #LogRegCv(X, y, False)
    cc += 1
    print(cc)
    # Normalizado
    #SVM_kernels(X, y, 'linear', True)
    cc += 1
    print(cc)
    # No Normalizado
    #SVM_kernels(X, y, 'linear', False)
    cc += 1
    print(cc)
    # Normalizado
    SVM_kernels(X, y, 'poly', True)
    cc += 1
    print(cc)
    # No Normalizado
    #SVM_kernels(X, y, 'poly', False)
    cc += 1
    print(cc)
    # Normalizado
    #SVM_kernels(X, y, 'rbf', True)
    cc += 1
    print(cc)
    # No Normalizado
    #SVM_kernels(X, y, 'rbf', False)
    cc += 1
    print(cc)
    # Normalizado
    #MultiLayerP(X, y, True)
    cc += 1
    print(cc)
    # No Normalizado
    #MultiLayerP(X, y, False)
    cc += 1
    print(cc)
