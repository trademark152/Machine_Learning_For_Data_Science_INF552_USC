
'''
    Class: INF552 at USC
    HW4
    Minh Tran tranmt@usc.edu
    A sklearn implementation of the Perceptron Learning Algorithm.
    Run code by: python PerceptronMT_SK.py 0.01 1000 classification.txt
'''

if __name__ == '__main__':
    import sys
    import numpy as np
    from numpy import loadtxt, where
    from sklearn.linear_model import Perceptron
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting

    USAGE = 'perceptronMT_SK.py <learning rate> <max number of Iterations>  <input data file>'
    if len(sys.argv) != 4:
        print(USAGE)
    else:
        lr = float(sys.argv[1])
        num_iteration = int(sys.argv[2])
        dataFile = sys.argv[3]

    # load trainData, use only the first 4 columns as instructed
    trainData = loadtxt(dataFile, delimiter=',', dtype='float', usecols=(0, 1, 2, 3))

    # import training trainData and labels
    X = trainData[:, :-1]
    y_train = trainData[:, -1]

    # build the perceptron model
    model = Perceptron(alpha=lr, max_iter=num_iteration, verbose=False)
    model.fit(X, y_train)

    # predict
    y_predicted = model.predict(X)

    # evaluate the result
    accuracyRate = np.where(y_predicted == y_train)[0].shape[0] / y_predicted.shape[0]

    # print out results
    print("FINAL RESULT")
    print('Number of Iterations: {0}'.format(model.n_iter_))
    print('Accuracy on the train trainDataset: {0}'.format(model.score(X,y_train)))
    print('Final Perceptron Weights [W0, W1, W2,...]: {0}'.format(model.coef_))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_train)
    ax2 = plt.gca()

    xx = np.linspace(0, 1, 10)
    yy = np.linspace(0, 1, 10)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = (-1 - model.coef_[0][0] * XX -
          model.coef_[0][1] * YY) / model.coef_[0][2]
    surf = ax2.plot_surface(XX, YY, ZZ)

    plt.show(block=False)
    plt.pause(1)
    plt.savefig('PerceptronModel_SK.png')
    plt.close()