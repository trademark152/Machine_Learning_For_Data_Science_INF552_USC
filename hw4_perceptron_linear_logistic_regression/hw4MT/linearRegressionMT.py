'''
    Class: INF552 at USC
    HW4
    Minh Tran
    A python implementation of linear Regression.
    Run by: python linearRegressionMT.py linear-regression.txt
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting

class LinearRegression():
    def __init__(self, weights=[]):
        self.weights = weights

    def train(self, X, Y):
        # W = (X.T * X)^-1 * X.T * Y
        self.weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)

if __name__ == '__main__':
    import sys

    USAGE = 'linearRegressionMT_SK.py <input data file>'
    if len(sys.argv) != 2:
        print(USAGE)
    else:
        dataFile = sys.argv[1]

    # load trainData, use only the first 4 columns as instructed
    trainData = np.loadtxt(dataFile, delimiter=',', dtype='float', usecols=(0, 1, 2))

    # import training trainData and labels
    X = trainData[:, :-1]
    y_train = trainData[:, -1]

    model = LinearRegression()
    model.train(X, y_train)

    print('Final Weights: {0}'.format(model.weights))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y_train, c=y_train)
    ax2 = plt.gca()
    xx = np.linspace(0, 1, 10)
    yy = np.linspace(0, 1, 10)

    XX, YY = np.meshgrid(xx, yy)
    ZZ = model.weights[0] * XX + model.weights[1] * YY
    surf = ax2.plot_surface(XX, YY, ZZ, antialiased=True)

    plt.show(block=False)
    plt.pause(1)
    plt.savefig('LinearRegressionModelMT.png')
    plt.close()