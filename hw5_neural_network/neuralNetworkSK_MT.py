'''
    Class: INF552 at USC
    HW5: Neural network
    Minh Tran
    A python sklearn implementation of neural network.
'''
from sklearn.neural_network import MLPClassifier

"""
Load pgm images: 
input is the picture directory
output is the image: list of pixel values
"""
def read_PGM_img(pic):
    with open(pic, 'rb') as file:
        # skip the first two lines
        file.readline() # "P5"
        file.readline()  # Comment line

        # image dimensions 32*30
        dimList = file.readline().split()
        dimX = int(dimList[0])
        dimY = int(dimList[1])
        size = dimX*dimY

        # grey scale
        max_greyscale = int(file.readline().strip())

        # image content, appending a pixel value between 0 and 1
        image = []
        for i in range(size):
            pixel = file.read(1)[0]
            image.append(pixel/max_greyscale)
        # print(image)
        return image

"""
MAIN
input are directories of train data and test data
"""
def main(train_List_Dir, test_List_Dir):
    # initialize lists of images and labels
    images = []
    labels = []

    # import training data
    with open(train_List_Dir) as f:
        for line in f.readlines():
            # get file directory
            train_img_dir = line.strip()

            # import the images
            images.append(read_PGM_img(train_img_dir))

            # assign label based on file name: 1 for down gesture, 0 for otherwise
            if 'down' not in train_img_dir:
                labels.append(0)
            else:
                labels.append(1)
    # print(images[0])
    # print(labels[0])

    # create the mlp
    # hidden_layer_sizes: tuple The ith element represents the number of neurons in the ith hidden layer.
    # solver: solver for weight optimization (sgd, lbfgs - quasiNewton, adam - sgd)
    # alpha: float: L2 penalty (regularization)
    # batch_size: size of minibatches for stochastic optimizers
    # learning_rate: constant, adaptive, invscaling
    # max_iter: int: iterate until converge or this number of iterations
    # warm_start: bool, reuse the solution of previous call to fit as initialization
    # tol: optimization tolerance
    # early_stopping: bool: terminate training when validation (10% training data set aside) is not improving
    nn = MLPClassifier(solver='sgd', tol=1e-3, alpha=0.1, learning_rate='adaptive',
                      hidden_layer_sizes=(100,), activation='logistic', learning_rate_init=0.1,
                      max_iter=1000, verbose=False, warm_start=True, early_stopping=False, validation_fraction=0.1)

    # fit training data
    nn.fit(images, labels)

    f.close()

    # predict testing data
    total_count = 0
    correct_count = 0
    with open(test_List_Dir) as f:
        for line in f.readlines():
            total_count += 1
            test_image = line.strip()

            # make prediction
            pred = nn.predict([read_PGM_img(test_image), ])[0]
            # print('{}: {}'.format(test_image, p))

            # update correct_count count if prediction is 1 and the name
            # of the image has "down"
            if (pred == 1) == ('down' in test_image):
                correct_count += 1

    print('correct_count rate on test data: {}'.format(correct_count / total_count))

if __name__ == '__main__':
    main('downgesture_train.list', 'downgesture_test.list')