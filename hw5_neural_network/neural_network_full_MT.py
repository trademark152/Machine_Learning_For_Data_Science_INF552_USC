'''
    Class: INF552 at USC
    HW5: Neural network
    Minh Tran
    A python implementation of neural network.
    python neuralNetworkMT.py 100 1000 downgesture_train.list downgesture_test.list
'''
# import libraries
import argparse
import cv2
import numpy as np
import json

# Softmax loss and Softmax gradient
### Loss functions ###
class softmax_cross_entropy:
    def __init__(self):
        self.expand_Y = None
        self.calib_logit = None
        self.sum_exp_calib_logit = None
        self.prob = None

    def forward(self, X, Y):
        self.expand_Y = np.zeros(X.shape).reshape(-1)
        self.expand_Y[Y.astype(int).reshape(-1) + np.arange(X.shape[0]) * X.shape[1]] = 1.0
        self.expand_Y = self.expand_Y.reshape(X.shape)

        self.calib_logit = X - np.amax(X, axis = 1, keepdims = True)
        self.sum_exp_calib_logit = np.sum(np.exp(self.calib_logit), axis = 1, keepdims = True)
        self.prob = np.exp(self.calib_logit) / self.sum_exp_calib_logit

        forward_output = - np.sum(np.multiply(self.expand_Y, self.calib_logit - np.log(self.sum_exp_calib_logit))) / X.shape[0]
        return forward_output

    def backward(self, X, Y):
        backward_output = - (self.expand_Y - self.prob) / X.shape[0]
        return backward_output


### Momentum ###
def add_momentum(model):
    # initialize momentum as a dictionary
    momentum = dict()

    # loop through model's keys: params, gradient...
    for module_name, module in model.items():
        # within model's parameters, pick 'params'
        if hasattr(module, 'params'):
            # loop through each keys of params of a model
            for key, _ in module.params.items():
                # add key to momentum as params+ValueOfW/b+W/b
                # add value to momentum as gradient of that W/b
                momentum[module_name + '_' + key] = np.zeros(module.gradient[key].shape)
    return momentum


def data_loader_mnist(dataset):
    # This function reads the MNIST data and separate it into train, val, and test set
    with open(dataset, 'r') as f:
        data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = np.array(train_set[0])
    Ytrain = np.array(train_set[1])
    Xvalid = np.array(valid_set[0])
    Yvalid = np.array(valid_set[1])
    Xtest = np.array(test_set[0])
    Ytest = np.array(test_set[1])

    return Xtrain, Ytrain, Xvalid, Yvalid, Xtest, Ytest


def predict_label(f):
    # This is a function to determine the predicted label given scores
    if f.shape[1] == 1:
        return (f > 0).astype(float)
    else:
        return np.argmax(f, axis=1).astype(float).reshape((f.shape[0], -1))

"""
Class to separate dataset to subset/batches
"""
class DataSplit:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N, self.d = self.X.shape

    def get_example(self, idx):
        batchX = np.zeros((len(idx), self.d))
        batchY = np.zeros((len(idx), 1))
        for i in range(len(idx)):
            batchX[i] = self.X[idx[i]]
            batchY[i, :] = self.Y[idx[i]]
        return batchX, batchY

# 1. One sigmoid Neural Network layer with forward and backward steps
class sigmoid:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size)
        self.gradient = np.zeros((output_size, input_size))
        self.output = np.zeros((output_size, 1))

    def forward(self, X):
        s = np.inner(self.W, X)
        e = np.exp(-s)
        e = 1 + e
        self.output = 1 / e
        return 1 / e

    def backward(self, incoming_gradient, X):
        local_gradient = ((1.0 - self.output) * self.output)
        self.gradient = local_gradient * np.sum(incoming_gradient, axis=0)
        self.gradient = np.outer(self.gradient, X)
        return self.gradient

# 1. One linear Neural Network layer with forward and backward steps
### Modules ###
class linear_layer:
    """
        The linear (affine/fully-connected) module.

        It is built up with two arguments:
        - input_D: the dimensionality of the input example/instance of the forward pass
        - output_D: the dimensionality of the output example/instance of the forward pass

        It has two learnable parameters:
        - self.params['W']: the W matrix (numpy array) of shape input_D-by-output_D
        - self.params['b']: the b vector (numpy array) of shape 1-by-output_D

        It will record the partial derivatives of loss w.r.t. self.params['W'] and self.params['b'] in:
        - self.gradient['W']: input_D-by-output_D numpy array
        - self.gradient['b']: 1-by-output_D numpy array
    """

    def __init__(self, input_D, output_D):
        self.params = dict()
        ###############################################################################################
        # TODO: Use np.random.normal() with mean as 0 and standard deviation as 0.1
        # W Shape (input_D, output_D), b shape (1, output_D)
        ###############################################################################################
        # raise NotImplementedError("Not Implemented function: __init__, class: linear_layer")
        # weights: dimension input_D-by-output_D, 'W' is the key
        self.params['W'] = np.random.normal(0, 0.1, (input_D, output_D))

        # bias: shape 1-by-output_D, 'b' is the key
        self.params['b'] = np.random.normal(0, 0.1, (1, output_D))

        self.gradient = dict()
        ###############################################################################################
        # TODO: Initialize gradients with zeros
        # Note: Shape of gradient is same as the respective variables
        ###############################################################################################
        # raise NotImplementedError("Not Implemented function: __init__, class: linear_layer")
        # weight gradients
        self.gradient['W'] = np.zeros((input_D, output_D))

        # bias gradient
        self.gradient['b'] = np.zeros((1, output_D))

    def forward(self, X):
        """
            The forward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, where each 'row' is an input example/instance (i.e., X[i], where                   i = 1,...,N).
                The mini-batch size is N.

            Return:
            - forward_output: A N-by-output_D numpy array, where each 'row' is an output example/instance.
        """

        ################################################################################
        # TODO: Implement the linear forward pass. Store the result in forward_output  #
        ################################################################################
        # raise NotImplementedError("Not Implemented function: forward, class: linear_layer")
        # u = X*w+b
        forward_output = np.matmul(X, self.params['W']) + self.params['b']
        # dimension check: N*Dout = N*Din * Din*Dout  + N*Dout
        return forward_output

    def backward(self, X, grad):
        """
            The backward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, the input to the forward pass.
            - grad: A N-by-output_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss
                 w.r.t. forward_output[i].

            Operation:
            - Compute the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['W'], self.params['b'].

            Return:
            - backward_output: A N-by-input_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss w.r.t. X[i].
        """

        #################################################################################################
        # TODO: Implement the backward pass (i.e., compute the following three terms)
        # self.gradient['W'] = ? (input_D-by-output_D numpy array, the gradient of the mini-batch loss w.r.t. self.params['W'])
        # self.gradient['b'] = ? (1-by-output_D numpy array, the gradient of the mini-batch loss w.r.t. self.params['b'])
        # backward_output = ? (N-by-input_D numpy array, the gradient of the mini-batch loss w.r.t. X)
        # only return backward_output, but need to compute self.gradient['W'] and self.gradient['b']
        #################################################################################################
        # raise NotImplementedError("Not Implemented function: backward, class: linear_layer")
        # Use matrix element-wise operation to avoid FOR loop

        # Store the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['W'] in self.gradient['W']:
        # dL/dW = dL/dU * dU/dW = dL/dU * X
        self.gradient['W'] = np.matmul(X.T, grad)

        # Store the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['b'] in self.gradient['b'].
        # dL/db = dL/dU * dU/db = dL/dU
        # keepdims is to maintain the array structure intact
        # axis = 0 is summmation along columns
        # here gradb is 1*D_out while grad is N*D_out so need to take sum of all data in batch as loss
        self.gradient['b'] = np.sum(grad, axis=0, keepdims=True)

        # Store the partial derivatives (gradients) of the mini-batch loss w.r.t. X in backward_output.
        # dL/dX = dL/dU * dU/dX = dL/dU * W
        backward_output = np.matmul(grad, self.params['W'].T)

        return backward_output


# 2. ReLU Activation
class relu:
    """
        The relu (rectified linear unit) module.

        It is built up with NO arguments.
        It has no parameters to learn.
        self.mask is an attribute of relu. You can use it to store things (computed in the forward pass) for the use in the backward pass.
    """

    def __init__(self):
        self.mask = None

    def forward(self, X):
        """
            The forward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape.

            Return:
            - forward_output: A numpy array of the same shape of X
        """

        ################################################################################
        # TODO: Implement the relu forward pass. Store the result in forward_output    #
        ################################################################################
        # raise NotImplementedError("Not Implemented function: forward, class: relu")
        # store a corresponding matrix of boolean that signifies if each element of X is positive or not, this is actually the derivative of relu function w.r.t X
        self.mask = np.array(X > 0.0).astype(float)

        # RELU definition: compare each element of X to 0, take the greater value of the two
        forward_output = np.maximum(X, 0.0)

        return forward_output

    def backward(self, X, grad):
        """
            The backward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in forward_output.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in  X.
        """

        ####################################################################################################
        # TODO: Implement the backward pass
        # You can use the mask created in the forward step.
        ####################################################################################################
        # raise NotImplementedError("Not Implemented function: backward, class: relu")
        # backward_output = dL/dX = dL/d(forward_output)*d(forward_output)/dX = grad*mask
        # d(forward_output)/dX is 1 for element greater than 0 and 0 else
        backward_output = np.multiply(self.mask, grad)

        return backward_output


# 3. tanh Activation

class tanh:
    def forward(self, X):
        """
            Input:
            - X: A numpy array of arbitrary shape.

            Return:
            - forward_output: A numpy array of the same shape of X
        """
        forward_output = np.tanh(X)
        return forward_output

    # Derivative of tanh is (1 - tanh^2)
    def backward(self, X, grad):
        """
            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in forward_output.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in  X.
        """
       # backward_output = dL/dX = dL/d(forward_output) * d(forward_output)/dX = grad * d(tanh(X))/dX
        forward_output = np.tanh(X)
        # use element wise matrix multiplication Hadaman product
        backward_output = np.multiply(grad, 1 - np.square(forward_output))
        return backward_output

# 4. Dropout
class dropout:
    """
        It is built up with one arguments:
        - r: the dropout rate

        It has no parameters to learn.
        self.mask is an attribute of dropout. You can use it to store things (computed in the forward pass) for the use in the backward pass.
    """

    def __init__(self, r):
        self.r = r
        self.mask = None

    def forward(self, X, is_train):

        """
            Input:
            - X: A numpy array of arbitrary shape.
            - is_train: A boolean value. If False, no dropout is performed.

            Operation:
            - If p >= self.r, output that element multiplied by (1.0 / (1 - self.r)); otherwise, output 0 for that element

            Return:
            - forward_output: A numpy array of the same shape of X (the output of dropout)
        """

        ################################################################################
        #  TODO: We provide the forward pass to you. You only need to understand it.   #
        ################################################################################

        # boolean to determine if dropout is performed
        # I(pi>=r) = np.random.uniform(0.0, 1.0, X.shape)>=self.r
        # calculate [1/(1-r)*I(pi>=r)]
        if is_train:
            self.mask = (1.0 / (1.0 - self.r)) * (np.random.uniform(0.0, 1.0, X.shape) >= self.r).astype(float)
        else:  # no dropout is performed so maintain a matrix with value ones to multiply with X (stay constant)
            self.mask = np.ones(X.shape)

        # element wise: [1/(1-r)*I(pi>=r)] * q(i)
        forward_output = np.multiply(X, self.mask)
        return forward_output

    def backward(self, X, grad):

        """
            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in forward_output.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in X.
        """

        ####################################################################################################
        # TODO: Implement the backward pass
        # You can use the mask created in the forward step
        ####################################################################################################
        # backward_output = dL/dX = dL/d(forward_output) * d(forward_output)/dX
        #  d(forward_output)/dX = [1/(1-r)*I(pi>=r)] because all these terms are constant when taking derivative w.r.t x or q
        backward_output = np.multiply(grad, self.mask)
        # raise NotImplementedError("Not Implemented function: backward, class: dropout")
        return backward_output


# 5. Mini-batch Gradient Descent Optimization
def miniBatchGradientDescent(model, momentum, _lambda, _alpha, _learning_rate):
    '''
        Input:
            model: Dictionary containing all parameters of the model
            momentum: Check add_momentum() function in utils.py to understand this parameter
            _lambda: Regularization constant
            _alpha: Momentum hyperparameter
            _learning_rate: Learning rate for the update

        Note: You can learn more about momentum here: https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/

        Returns: Updated model
    '''

    for module_name, module in model.items():

        # check if a module has learnable parameters
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                g = module.gradient[key] + _lambda * module.params[key]

                if _alpha > 0.0:

                    #################################################################################
                    # TODO: Update momentun using the formula:
                    # m = alpha * m - learning_rate * g (Check add_momentum() function in utils file)
                    # And update model parameter
                    #################################################################################

                    # raise NotImplementedError("Not Implemented function: miniBatchGradientDescent")
                    momentum[module_name + '_' + key] = _alpha * momentum[module_name + '_' + key] - _learning_rate * g

                    # update weights using momentum
                    module.params[key] += momentum[module_name + '_' + key]


                else:  # _alpha = 0

                    #################################################################################
                    # TODO: update model parameter without momentum
                    #################################################################################

                    # raise NotImplementedError("Not Implemented function: miniBatchGradientDescent")
                    module.params[key] -= _learning_rate * g
    return model
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
        img = cv2.imread(file, 0)  # 0 for greyscale;

        # reshape the image into 1 list with size = numRow * numCol
        dim = len(img) * len(img[0])
        pgm_image = list(img.reshape(dim))

        # append label to the end of the image
        pgm_image.append(common_label)

        # add the final image to data
        output.append(pgm_image)
    return output


def main(main_params, optimization_type="minibatch_sgd"):

    num_neurons = 100
    num_iteration = 1000

    # read input data and separate to 'down' and 'not_down' gestures files
    print("start reading input text files")
    train_files_label1, train_files_label0 = parse_file_list(main_params['input_train_file'])
    test_files_label1, test_files_label0 = parse_file_list(main_params['input_test_file'])

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
    np.random.seed(int(main_params['random_seed']))
    np.random.shuffle(data_train)

    # prepare data
    Xtrain = data_train[:, :-1]
    Ytrain = data_train[:, -1]
    Xval = data_test[:, :-1]
    Yval = data_test[:, -1]

    # data processing
    N_train, d = Xtrain.shape
    N_val, _ = Xval.shape

    index = np.arange(10)

    # get unique values of training data
    unique, counts = np.unique(Ytrain, return_counts=True)
    counts = dict(zip(unique, counts)).values()

    # create a train and test set to be divided into batches later
    trainSet = DataSplit(Xtrain, Ytrain)
    valSet = DataSplit(Xval, Yval)

    ### building/defining MLP ###
    """
    The network structure is input --> linear --> relu --> dropout --> linear --> softmax_cross_entropy loss
    the hidden_layer size (num_L1) is 100
    the output_layer size (num_L2) is 1
    """
    model = dict()
    num_L1 = main_params['num_neurons_hidden']
    num_L2 = main_params['num_neurons_output']

    # experimental setup
    num_epoch = int(main_params['num_epoch'])
    minibatch_size = int(main_params['minibatch_size'])

    # optimization setting: _alpha for momentum, _lambda for weight decay
    _learning_rate = float(main_params['learning_rate'])
    _step = 10 # ??
    _alpha = float(main_params['alpha'])
    _lambda = float(main_params['lambda'])
    _dropout_rate = float(main_params['dropout_rate'])
    _activation = main_params['activation']

    if _activation == 'relu':
        act = relu
    elif _activation == 'sigmoid':
        act = sigmoid
    else:
        act = tanh

    # create objects (modules) from the module classes
    model['L1'] = linear_layer(input_D=d, output_D=num_L1)
    model['nonlinear1'] = act()
    model['drop1'] = dropout(r=_dropout_rate)
    model['L2'] = linear_layer(input_D=num_L1, output_D=num_L2)
    model['loss'] = softmax_cross_entropy()

    # Momentum
    if _alpha > 0.0:
        momentum = add_momentum(model)
    else:
        momentum = None

    train_acc_record = []
    val_acc_record = []

    train_loss_record = []
    val_loss_record = []

    ### run training and validation ###
    for t in range(num_epoch):
        print('At epoch ' + str(t + 1))
        if (t % _step == 0) and (t != 0):
            _learning_rate = _learning_rate * 0.1

        idx_order = np.random.permutation(N_train)

        train_acc = 0.0
        train_loss = 0.0
        train_count = 0

        val_acc = 0.0
        val_count = 0
        val_loss = 0.0

        for i in range(int(np.floor(N_train / minibatch_size))):
            # get a mini-batch of data
            x, y = trainSet.get_example(idx_order[i * minibatch_size: (i + 1) * minibatch_size])

            ### forward ###  x -L1> a1 -NL> h1 -Dr> d1 -L2> a2 -loss> y
            a1 = model['L1'].forward(x)  # a1 or u
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train=True)  # d1 or h after dropout
            a2 = model['L2'].forward(d1)
            loss = model['loss'].forward(a2, y)

            ### backward ###
            grad_a2 = model['loss'].backward(a2, y)
            ######################################################################################
            # TODO: Call the backward methods of every layer in the model in reverse order
            # We have given the first and last backward calls
            # Do not modify them.
            ######################################################################################

            # raise NotImplementedError("Not Implemented BACKWARD PASS in main()")
            grad_a2 = model['loss'].backward(a2, y)
            grad_d1 = model['L2'].backward(d1, grad_a2)
            grad_h1 = model['drop1'].backward(h1, grad_d1)
            grad_a1 = model['nonlinear1'].backward(a1, grad_h1)

            ######################################################################################
            # NOTE: DO NOT MODIFY CODE BELOW THIS, until next TODO
            ######################################################################################
            grad_x = model['L1'].backward(x, grad_a1)

            ### gradient_update ###
            model = miniBatchGradientDescent(model, momentum, _lambda, _alpha, _learning_rate)

        ### Computing training accuracy and obj ###
        for i in range(int(np.floor(N_train / minibatch_size))):
            x, y = trainSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))

            ### forward ###
            ######################################################################################
            # TODO: Call the forward methods of every layer in the model in order
            # Check above forward code
            ######################################################################################

            # raise NotImplementedError("Not Implemented COMPUTING TRAINING ACCURACY in main()")
            a1 = model['L1'].forward(x)  # a1 or u
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train=True)  # d1 or h after dropout
            a2 = model['L2'].forward(d1)

            ######################################################################################
            # NOTE: DO NOT MODIFY CODE BELOW THIS, until next TODO
            ######################################################################################

            loss = model['loss'].forward(a2, y)
            train_loss += loss
            train_acc += np.sum(predict_label(a2) == y)
            train_count += len(y)

        train_loss = train_loss
        train_acc = train_acc / train_count
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)

        print('Training loss at epoch ' + str(t + 1) + ' is ' + str(train_loss))
        print('Training accuracy at epoch ' + str(t + 1) + ' is ' + str(train_acc))

        ### Computing validation accuracy ###
        for i in range(int(np.floor(N_val / minibatch_size))):
            x, y = valSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))

            ### forward ###
            ######################################################################################
            # TODO: Call the forward methods of every layer in the model in order
            # Check above forward code
            ######################################################################################
            a1 = model['L1'].forward(x)  # a1 or u
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train=True)  # d1 or h after dropout
            a2 = model['L2'].forward(d1)

            # raise NotImplementedError("Not Implemented COMPUTING VALIDATION ACCURACY in main()")

            ######################################################################################
            # NOTE: DO NOT MODIFY CODE BELOW THIS, until next TODO
            ######################################################################################

            loss = model['loss'].forward(a2, y)
            val_loss += loss
            val_acc += np.sum(predict_label(a2) == y)
            val_count += len(y)

        val_loss_record.append(val_loss)
        val_acc = val_acc / val_count
        val_acc_record.append(val_acc)

        print('Validation accuracy at epoch ' + str(t + 1) + ' is ' + str(val_acc))

    # save file
    json.dump({'train': train_acc_record, 'val': val_acc_record},
              open('MLP_lr' + str(main_params['learning_rate']) +
                   '_m' + str(main_params['alpha']) +
                   '_w' + str(main_params['lambda']) +
                   '_d' + str(main_params['dropout_rate']) +
                   '_a' + str(main_params['activation']) +
                   '.json', 'w'))

    print('Finish running!')
    return train_loss_record, val_loss_record


if __name__ == "__main__":
    # parsing argument
    # train_data = 'downgesture_train.list'
    # test_data = 'downgesture_test.list'
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=7)
    parser.add_argument('--learning_rate', default=0.1)
    parser.add_argument('--alpha', default=0.0) # L2 penalty term for regularization
    parser.add_argument('--lambda', default=0.0) # regurlarization?
    parser.add_argument('--dropout_rate', default=0.0)
    parser.add_argument('--num_epoch', default=100) # number of scans around dataset
    parser.add_argument('--minibatch_size', default=10) # size for minibatches for stochastic optimizer
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--input_train_file', default='downgesture_train.list')
    parser.add_argument('--input_test_file', default='downgesture_test.list')
    parser.add_argument('--num_neurons_output', default=2)
    parser.add_argument('--num_neurons_hidden', default=100)

    # combine arguments and parse them to parameters
    args = parser.parse_args()
    main_params = vars(args)

    # run main
    main(main_params)