import numpy as np
import mlnn
import cv2
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


# evaluate the error
def eval(Y_pred, Y_true):
    # measure accuracy
    acc = 0
    for i in range(len(Y_pred)):
        if (int(Y_pred[i]) == Y_true[i]):
            acc = acc + 1
    print(acc * 1.0 / len(Y_true))

if __name__ == '__main__':
    train_data = 'downgesture_train.list'
    test_data = 'downgesture_test.list'
    num_neurons = 100
    num_iteration = 1000  #number of epoch training
    lr = 0.01

    # read input data and separate to 'down' and 'not_down' gestures files
    print("start reading input text files")
    train_files_label1, train_files_label0 = parse_file_list(train_data)
    test_files_label1, test_files_label0 = parse_file_list(test_data)

    # data_one_train = [960 px, 1]; data_zero_train = [960 px, 0] with labels at the end
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
    # np.random.seed(7)
    np.random.shuffle(data_train)

    # prepare data by separating input and response
    X_train = data_train[:, :-1]
    Y_train = data_train[:, -1]
    X_test = data_test[:, :-1]
    Y_test = data_test[:, -1]

    # last dimension is 1 for lse, 2 or more for softmax
    layers_dim = [np.shape(X_train)[1], num_neurons, 1]

    # create ANN: choose tanh or sigmoid, lse or softmax
    model = mlnn.Model(layers_dim, activation_func='sigmoid', output_func ='lse')
    model.train(X_train, Y_train, num_passes=num_iteration, lr=lr, regularization=0.01, to_print=True)

    # predict train data
    y_train_pred = model.predict(X_train)

    # predict test data
    y_pred = model.predict(X_test)

    # evaluate error
    print("training accuracy:")
    eval(y_train_pred, Y_train)

    print("testing accuracy:")
    eval(y_pred, Y_test)
