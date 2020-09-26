
'''
    Class: INF552 at USC
    HW4
    Minh Tran tranmt@usc.edu
    A python implementation of the Pocket Algorithm.
    Run code by: python pocketMT.py 0.01 7000 classification.txt
'''

# Imports
import copy
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D  #<--- This is important for 3d plotting
np.random.seed(7)
# End imports

class Pocket():
    def __init__(self, weights=[], lr=0.01, num_iteration=1000):
        self.weights = weights
        self.lr = lr
        self.num_iteration = num_iteration
        self.numError = []
        self.bestWeights = []
        self.bestErrorCount = float('inf')

    def train(self, X, Y_train, verbose=False):
        d = X.shape[1]  # trainData dimension
        n = X.shape[0]  # number of data points

        # insert column of 1s to the 1st column of input trainData
        X = np.insert(X, 0, 1, axis=1)

        # initialize random weights
        self.weights = np.random.random(d + 1)
        iter = 0
        while iter < self.num_iteration:
            self.numError.append(0)
            for x, y in zip(X, Y_train):
                # if misclassified, update weight accordingly
                # because y is +1 or -1
                if np.dot(x, self.weights) * y < 0:
                    self.weights += self.lr * y * x
                    # break # consider this to avoid looping around all misclassifications

                # prod = np.dot(x, self.weights)
                # if prod > 0 and y < 0:
                #     self.weights -= self.lr * x
                # elif prod < 0 and y > 0:
                #     self.weights += self.lr * x

            # after a round (epoch) of weight update, check prediction to update error count
            Y_temp = np.sign(np.dot(X, self.weights))

            # update error counts
            nCorrect = np.where(Y_temp == Y_train)[0].shape[0]
            self.numError[-1] = n - nCorrect
            
            if self.bestErrorCount > self.numError[-1]:
                self.bestErrorCount = self.numError[-1]
                self.bestWeights = copy.deepcopy(self.weights)
                self.best_iteration_number = iter
                
            # track progress
            if verbose:
                if iter % 1000 == 0:
                    print('Iteration in Progress: {0}'.format(iter))
            iter += 1

            # if no more misclassification, stop
            if self.numError[-1] == 0:
                break

        return iter
    
    # function to predict an instance
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.sign(np.dot(X, self.bestWeights))


if __name__ == '__main__':
    USAGE = 'pocketMT.py <learning rate> <max number of Iterations>  <input data file>'
    if len(sys.argv) != 4:
        print(USAGE)
    else:
        lr = float(sys.argv[1])
        num_iteration = int(sys.argv[2])
        dataFile = sys.argv[3]

    # load trainData, use only the first 4 columns as instructed
    trainData = np.loadtxt(dataFile, delimiter=',', dtype='float', usecols=(0, 1, 2, 4))

    # import training trainData and labels
    X = trainData[:, :-1]
    y_train = trainData[:, -1]

    # build the perceptron_model
    pocket_model = Pocket(lr=lr, num_iteration=num_iteration)
    num_iter_final = pocket_model.train(X, y_train, verbose=True)

    # predict
    y_predicted = pocket_model.predict(X)

    # evaluate the result
    accuracyRate = np.where(y_predicted==y_train)[0].shape[0] / y_predicted.shape[0]

    # print out results
    print("FINAL RESULT")
    print('Iterations Number (best): {0}'.format(pocket_model.best_iteration_number))
    print('Accuracy on the train dataset (best): {0}'.format(accuracyRate))
    print('Pocket Weights [W0, W1, W2,...]: {0}'.format(pocket_model.bestWeights))

    plt.ylabel('Error count')
    plt.xlabel('Number of Iterations')
    plt.plot(pocket_model.numError, color='green', marker='*', linestyle='dashed',linewidth=2, markersize=2)
    plt.show(block=False)
    plt.pause(1)
    plt.savefig('pocketError.png')
    plt.close()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_train)
    ax2 = plt.gca()

    xx = np.linspace(0, 1, 10)
    yy = np.linspace(0, 1, 10)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = (-pocket_model.weights[0] - pocket_model.weights[1] * XX -
          pocket_model.weights[2] * YY) / pocket_model.weights[3]
    surf = ax2.plot_surface(XX, YY, ZZ)
    plt.show(block=False)
    plt.pause(1)
    plt.savefig('PocketModel.png')
    plt.close()