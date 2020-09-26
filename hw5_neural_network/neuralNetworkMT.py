'''
    Class: INF552 at USC
    HW5: Neural network
    Minh Tran
    A python implementation of neural network.
    python neuralNetworkMT.py 100 1000 downgesture_train.list downgesture_test.list
'''

# import libraries
import re
import numpy as np
import cv2
import sys

"""
Read directory list file to extract files with one and zero label 
input: list file directory
output:2 list: ['file1_label_1','file2_label_1'...] and ['file1_label_0','file2_label_0'...]   
"""
def parse_file_list(list_dir):
    files_label1 = []
    files_label0 = []
    with open(list_dir) as f:
        for line in f.readlines():
            if 'down' not in line:
                files_label0.append(line.strip())
            else:
                files_label1.append(line.strip())
    return files_label1, files_label0

"""
read pgm file into 1 np list with label
input: file lists and corresponding label
output:1 list: [[data_pic1_label1, 1],...]    
"""
def read_pgm_image(file_list, common_label):
    output = []  # these files share the same label due to parsing of previous step
    for file in file_list:
        # use imread to read the file
        # If the image cannot be read (because of missing file, improper permissions,
        # unsupported or invalid format) then this method returns an empty matrix.
        # output is a list of lists based on dimension
        img = cv2.imread(file, 0) # 0 for greyscale;

        # reshape the image into 1 list with size = numRow * numCol
        dim = len(img) * len(img[0])
        pgm_image = list(img.reshape(dim))
        
        # append label to the end of the image
        pgm_image.append(common_label)
        
        # add the final image to data
        output.append(pgm_image)
    return output


"""
A class of neural network
"""
class NN(object):
    W = []  # weight
    X = []  # output of perceptron
    S = []  # input of perceptron
    Theta = []  # ?

    # initialization: layers is a list consisting of number of neurons at each layer
    # prompt: layers = [length(input), 100, 1]
    def __init__(self, layers, X, Y, lr):
        self.layers = layers
        self.num_layers = len(layers)  # number of layers including both input and output layers
        self.num_data, self.dim_data = np.shape(X)  # get number of data points and dimension of data
        self.lr = lr

        # initialization
        for i in range(self.num_layers):
            # input layer
            if (i == 0):
                self.X.append(X)
                self.Y = Y[:, np.newaxis] # increase the  dimension of existing array by 1 dimension
                self.S.append([])
                self.Theta.append([])
            # other layers
            else:
                # creates an array of specified shape and fills it with random values
                # as per standard normal distribution.
                self.X.append(np.random.randn(self.num_data, layers[i]))
                self.S.append(np.random.randn(self.num_data, layers[i]))
                self.Theta.append(np.random.randn(self.num_data, layers[i]))

            # if not the last layer, initialize the weight w_ij for layer i+1
            if (i != self.num_layers - 1):
                self.W.append(self.random(layers[i], layers[i + 1]))

    # weight matrix via randomized initiation a*b dimension
    # a and b are the number of neurons in layer i and i+1 respectively
    # value specified between -0.01 and 0.01 as guided by the prompt
    def random(self, a, b):
        return 0.02 * np.random.rand(a, b) - 0.01

    def set_data(self, X, Y):
        self.X[0] = X
        self.Y = Y[:, np.newaxis]

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def predict(self):
        for i in range(1, self.num_layers):
            self.S[i] = np.dot(self.X[i - 1], self.W[i - 1])
            np.clip(self.S[i], -700, None, out=self.S[i])
            self.X[i] = self.sigmoid(self.S[i])

    def backword(self):
        for i in range(self.num_layers - 1, 0, -1):
            if (i == self.num_layers - 1):
                self.Theta[i] = np.multiply(np.multiply(self.X[i] - self.Y, 2), np.multiply(self.X[i], (1 - self.X[i])))
            else:
                self.Theta[i] = np.multiply(np.dot(self.Theta[i + 1], self.W[i].T), np.multiply(self.X[i], (1 - self.X[i])))

        for i in range(self.num_layers - 1):
            self.W[i] = self.W[i] - np.multiply(self.lr,np.dot(self.X[i].T, self.Theta[i + 1]) / self.num_data)

    def display(self):
        print(self.S)

# evaluate the error
def eval(data, nn, Y):
    # initialize the output to be all 1s
    predict = np.ones((len(data), 1))

    # replace the output from 1 to 0 in location where prediction is <0.5
    predict[nn.X[2] < 0.5] = 0

    # measure accuracy
    acc = 0
    for i in range(len(data)):
        if (int(predict[i, 0]) == Y[i]):
            acc = acc + 1
    print(acc * 1.0 / len(data))

if __name__ == '__main__':
    USAGE = 'neuralNetworkMT.py <number of neurons in hidden layer>' \
            ' <max number of Iterations>  <train data> <test data>'
    # if len(sys.argv) != 5:
    #     print(USAGE)
    # else:
    #     num_neurons = int(sys.argv[1])
    #     num_iteration = int(sys.argv[2])
    #     train_data = sys.argv[3]
    #     test_data = sys.argv[4]

    train_data = 'downgesture_train.list'
    test_data = 'downgesture_test.list'
    num_neurons = 100
    num_iteration = 1000  #number of epoch training
    lr = 0.1

    # read input data and separate to 'down' and 'not_down' gestures files
    print("start reading input text files")
    train_files_label1, train_files_label0 = parse_file_list(train_data)
    test_files_label1, test_files_label0 = parse_file_list(test_data)

    print("start reading input pictures files")
    data_one_train = read_pgm_image(train_files_label1, 1)
    data_zero_train = read_pgm_image(train_files_label0, 0)
    data_one_test = read_pgm_image(test_files_label1, 1)
    data_zero_test = read_pgm_image(test_files_label0, 0)

    # combine pgm data with corresponding labels in train and test
    data_train = np.concatenate((data_one_train, data_zero_train), axis=0)
    data_test = np.concatenate((data_one_test, data_zero_test), axis=0)

    # shuffle and prepare data
    print("start shuffling and splitting data")
    np.random.seed(7)
    np.random.shuffle(data_train)

    # prepare data
    X_train = data_train[:, :-1]
    Y_train = data_train[:, -1]
    X_test = data_test[:, :-1]
    Y_test = data_test[:, -1]

    # create the mlp neural network
    nn = NN([np.size(X_train, 1), num_neurons, 1], X_train, Y_train, lr)
    for i in range(num_iteration):
        nn.predict()
        nn.backword()

    # fit?
    eval(data_train, nn, Y_train)

    # perform similar steps to the test data
    nn.set_data(X_test, Y_test)
    nn.predict()

    eval(data_test, nn, Y_train)
    print(np.ones((len(data_test), 1))[:, 0])

