'''
    Class: INF552 at USC
    HW6: SVM using sklearn
    Minh Tran
    python neuralNetworkMT.py 100 1000 downgesture_train.list downgesture_test.list
'''


"""
To run code
python svm_MT_SK.py nonlinear
python svm_MT_SK.py linear
"""

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import *

def main():
    if (sys.argv[1] == "linear"):
        data = np.loadtxt('linsep.txt', dtype='float', delimiter=',')
    else:
        data = np.loadtxt('nonlinsep.txt', dtype='float', delimiter=',')

    # prepare training and test data
    X = data[:, 0:2]
    y = data[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, seed=7)

    # Build svm engine using sklearn
    """
    penalty: Specifies the norm used in the penalization
    loss: specify the loss function
    dual: Select the algorithm to either solve the dual or primal optimization problem. 
    Prefer dual=False when n_samples > n_features
    tol: tolerance for stopping criteria
    multi_class: Determines the multi-class strategy if y contains more than two classes.
    ovr: n_classes one vs rest classifiers 
    fit_intercept: calculate the intercept for the model
    class_weight: Set the parameter C of class i to class_weight[i]*C for SVC
    verbose: Enable verbose output
    C: Regularization parameter
    random_state: e seed of the pseudo random number generator to use when shuffling the data for the dual coordinate descent
    """
    if (sys.argv[1] == "linear"):
        clf = svm.SVC(C=10.0, kernel='linear', degree=3, gamma='scale',
                      coef0=0.0, shrinking=True, probability=False, tol=0.00001,
                      cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                      decision_function_shape='ovr', random_state=None)
    else:
        clf = svm.SVC(C=1000.0, kernel='poly', degree=2, gamma='scale',
                      coef0=0.0, shrinking=True, probability=False, tol=0.00001,
                      cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                      decision_function_shape='ovr', random_state=None)

    """
    if using linear svc
    """
    # clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=False,
    #                     tol=0.01, C=1, multi_class='ovr',
    #                     fit_intercept=True, intercept_scaling=1,
    #                     class_weight=None, verbose=0, random_state=None, max_iter=1000)


    # fit the training data
    clf.fit(X_train, y_train)

    # get output
    support_vectors = clf.support_vectors_
    support_vector_indices = clf.support_

    """
    if using linearsvc
    """
    # get support vectors
    # decision_function = clf.decision_function(X_train)
    # decision_function = np.dot(X_train, clf.coef_[0]) + clf.intercept_[0]
    #
    # support_vector_indices = np.where((2 * y_train - 1) * decision_function <= 1)[0]
    # support_vectors = X[support_vector_indices]

    # make prediction
    y_test_pred = clf.predict(X_test)

    print("support vectors: ", support_vectors)
    print("support vectors indices: ", support_vector_indices)

    if (sys.argv[1] == "linear"):
        print("weights: ", clf.coef_)
        print("intercept: ", clf.intercept_)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=1, s=10, edgecolors='k')
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', s=100, edgecolors='k')

    # plot the svm decision plane
    if (sys.argv[1] == "linear"):
        inputX = np.linspace(0, 1, 11)
        outputY = []
        for x in inputX:
            outputY.append(-(clf.coef_[0][0] * x + clf.intercept_[0]) / clf.coef_[0][1])

        plt.plot(inputX, outputY)
        plt.show(block=False)
        plt.pause(1)
        plt.savefig('linearSVM.png')

    else:
        h = 0.1
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        inputX = np.array([xx.ravel(), yy.ravel()]).T
        Z = clf.predict(inputX)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        plt.show(block=False)
        plt.pause(1)
        plt.savefig('nonlinearSVM.png')

    plt.close()
if __name__ == "__main__":
    main()