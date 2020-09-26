'''
    Class: INF552 at USC
    HW4
    Minh Tran
    A python sklearn implementation of linear Regression.
    Run by: python linearRegressionMT_SK.py linear-regression.txt
'''

if __name__ == '__main__':
    # Imports
    import sys
    import numpy as np
    from numpy import loadtxt
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting

    # End imports

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
    model.fit(X, y_train)

    print('Final Weights: {0}'.format(model.coef_))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y_train, c=y_train)
    ax2 = plt.gca()
    xx = np.linspace(0, 1, 10)
    yy = np.linspace(0, 1, 10)

    XX, YY = np.meshgrid(xx, yy)
    ZZ = model.coef_[0] * XX + model.coef_[1] * YY
    surf = ax2.plot_surface(XX, YY, ZZ, antialiased=True)

    plt.show(block=False)
    plt.pause(1)
    plt.savefig('LinearRegressionModelMT_SK.png')
    plt.close()