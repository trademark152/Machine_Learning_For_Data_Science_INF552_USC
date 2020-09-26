'''
    Class: INF552 at USC
    HW4
    Minh Tran
    A python implementation of Logistic Regression.
    Run by: python logisticRegressionMT.py 0.01 7000 classification.txt
'''

# Imports
import numpy as np
import sys
np.random.seed(7)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #<--- This is important for 3d plotting

# function to calculate sigmoid function
def calc_sigmoid(z):
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid

# logistic regression
class LogisticRegression():
    def __init__(self, weights=[], lr=0.001, num_iteration=7000):
        self.weights = weights
        self.lr = lr
        self.num_iteration = num_iteration
        self.numError= []

    # a static method can be called without an object for that class
    @staticmethod
    # function to calculate incremental gradient descent for each instance
    def calc_gradient(weights, x, y):
        # calculate P(-yn * w * xn):
        prob = calc_sigmoid(-y * np.dot(weights, x))

        # calculate incremental gradient
        grad = (x * y) * prob
        return grad

    def train(self, X, Y, verbose=False):
        # number of data and data dimensions
        N, d = X.shape

        # insert column of 1s to the 1st column of input trainData
        X = np.insert(X, 0, 1, axis=1)

        # initialize random weights
        self.weights = np.random.random(d + 1)
        iter = 0
        while iter < self.num_iteration:
            self.numError.append(0)
            gradient = np.zeros(d + 1)
            for x, y in zip(X, Y):
                descent = LogisticRegression.calc_gradient(self.weights, x, y)

                # cumulatively add descent based on all instances
                gradient = np.add(gradient, descent)

            # get the average gradient to update weights
            gradient = gradient/N
            self.weights += self.lr * gradient

            # after a round (epoch) of weight update, check prediction to update error count
            Y_temp = np.sign(np.dot(X, self.weights))

            # update error counts
            nCorrect = np.where(Y_temp == Y)[0].shape[0]
            self.numError[-1] = N - nCorrect

            iter += 1

            if verbose:
                if iter % 500 == 0:
                    print('Completed iterations: {0}'.format(iter))

    # predict function
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        Y_pred = [None for _ in range(X.shape[0])]
        for idx, x in enumerate(X):
            # calculate P(1|x,w) = sigmoid(y*wT*x)
            prob_one = calc_sigmoid(1 * np.dot(self.weights, x))

            # determine label
            if prob_one <= 0.5:
                Y_pred[idx] = -1
            else:
                Y_pred[idx] = 1
        return np.asarray(Y_pred)

"""
MAIN
"""
if __name__ == '__main__':
    USAGE = 'logisticRegressionMT.py <learning rate> <max number of Iterations>  <input data file>'
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

    # build the logistic regression_model
    LogisticRegression_model = LogisticRegression(lr=lr, num_iteration=num_iteration)
    num_iter_final = LogisticRegression_model.train(X, y_train, verbose=True)

    # predict
    y_predicted = LogisticRegression_model.predict(X)

    # evaluate the result
    accuracyRate = np.where(y_predicted == y_train)[0].shape[0] / y_predicted.shape[0]

    # print out results
    print("FINAL RESULT")
    print('Accuracy on the train dataset (best): {0}'.format(accuracyRate))
    print('Logistic Regression Weights [W0, W1, W2,...]: {0}'.format(LogisticRegression_model.weights))

    plt.ylabel('Error count')
    plt.xlabel('Number of Iterations')
    plt.plot(LogisticRegression_model.numError, color='green', marker='*', linestyle='dashed', linewidth=2, markersize=2)
    plt.show(block=False)
    plt.pause(1)
    plt.savefig('LogisticRegressionError.png')
    plt.close()


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_train)

    # Ensure that the next plot doesn't overwrite the first plot
    ax2 = plt.gca()

    xx = np.linspace(0, 1, 10)
    yy = np.linspace(0, 1, 10)

    XX, YY = np.meshgrid(xx, yy)
    ZZ = (-LogisticRegression_model.weights[0] - LogisticRegression_model.weights[1] * XX -
          LogisticRegression_model.weights[2] * YY) / LogisticRegression_model.weights[3]
    surf = ax2.plot_surface(XX, YY, ZZ)

    plt.show(block=False)
    plt.pause(1)
    plt.savefig('LogisticRegressionModel.png')
    plt.close()