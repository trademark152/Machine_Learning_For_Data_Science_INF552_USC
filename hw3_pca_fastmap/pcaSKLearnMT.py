"""
INF 552 - Homework 3 - Minh Tran
PCA using Sklearn
"""

'''
python pcaSKLearnMT.py pca-data.txt 2 pca-sklearn-mt.txt
'''

# Imports
import sys
import numpy as np
from numpy import genfromtxt, set_printoptions
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #<--- This is important for 3d plotting

if __name__ == '__main__':
    USAGE = 'USAGE: python pcaSKLearnMT.py pca-data.txt 2 pca-sklearn-mt.txt'
    if len(sys.argv) != 4:
        print(USAGE)
        sys.exit(1)
    else:
        inFile = sys.argv[1] # input file
        numDims = int(sys.argv[2])  # desired number of dimension after pca
        outFile = sys.argv[3] # output file

    # import data
    inputData = genfromtxt(inFile, delimiter='\t')
    # print("inputData: ", np.size(inputData,1))

    # plot data
    if np.size(inputData, 1) == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(inputData[:,0], inputData[:,1], inputData[:,2])
        plt.show(block=False)
        plt.pause(1)
        plt.savefig('3dData.png')
        plt.close()

    if np.size(inputData, 1) < numDims:
        raise ValueError('Cannot perform PCA because desire dimension is greater than original dimension')

    # create model
    pca = PCA(n_components=numDims)

    # fit data X = 6000*3
    pca.fit(inputData)

    # print out the vectors representing the PC: each with dimension 3*1
    for idxPC, pcVec in enumerate(pca.components_):
        print('SKLearn Principal Component {0}: {1}'.format(idxPC, pcVec))

    # transform the data
    transformedData = pca.transform(inputData)
    # print("transformed data: ", transformedData)

    if np.size(transformedData, 1) == 2:
        fig, ax = plt.subplots()
        ax.scatter(transformedData[:, 0], transformedData[:, 1])
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.show(block=False)
        plt.pause(1)
        plt.savefig('2dData.png')
        plt.close()

    # write output
    # set_printoptions(suppress=True)
    with open(outFile, 'w') as file:
        for old, new in zip(inputData, transformedData):
            file.write('PCA: {0} -> {1}'.format(old, new))
            file.write('\n')