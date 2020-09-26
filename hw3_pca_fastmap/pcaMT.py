"""
INF 552 - Homework 3 - Minh Tran
PCA
"""

'''
python pcaMT.py pca-data.txt 2 pca-mt.txt
'''

# Imports
from collections import namedtuple
from numpy import genfromtxt, linalg, dot, sqrt
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #<--- This is important for 3d plotting

"""
PCA algorithm:
Performs principal component on x, a matrix with observations in the rows.
"""

"""
The transpose trick says that if v is an eigenvector of M^T M, then M^T v is an eigenvector of MM^T.
We arbitrarily select "100" as the switching threshold. Another approach is to switch by comparing numPts and numDims.
"""

PC = namedtuple('PC', ['idx', 'eigVal', 'eigVec'])
def pca(x, targetDims):
    # Subtract the mean of dimension i from column i, in order to center the matrix.
    x = (x - x.mean(axis=0))

    # extract the number of data points and number of dimensions
    numPts, numDims = x.shape

    """
    Returns the projection matrix (the eigenvectors of x^T x, ordered with largest eigenvectors first) and the eigenvalues (ordered from largest to smallest).
    """
    if numDims >= 100:
        # get eigenvalues and eigenvectors of x*x.T, here eigenvalues are in ascending order
        eigenvalues, eigenvectors = linalg.eigh(dot(x, x.T))

        # print("eigenvalues: ", eigenvalues)

        # obtain the actual eigenvectors of x.T*x using transpose trick
        v = (dot(x.T, eigenvectors).T)[::-1]  # Unscaled and reversing order, but the relative order is still correct.

        # obtain the eigenvalues in descending order:
        s = sqrt(eigenvalues)[::-1]  # Unscaled, but the relative order is still correct.
        # print("s: ", s)
    else:
        u, s, v = linalg.svd(x, full_matrices=False)

    # print("s: ", s)
    # print("v: ", v)

    # extract the needed components:enumerate to maintain the count
    principalComps = sorted(enumerate(s), key=lambda x: x[1], reverse=True)[:targetDims]
    # print("principalComps: ", principalComps)

    return [PC(idx, eigVal, v[idx]) for (idx, eigVal) in principalComps]
    # return v, s

def PCAtransform(old, PCs):
    # extract matrix consisting of eigenvectors
    eigVecMat = np.asarray([pc.eigVec for pc in PCs])

    return eigVecMat.dot(old)

if __name__ == '__main__':
    USAGE = 'USAGE: python pcaSKLearnMT.py pca-data.txt 2 pca-sklearn-mt.txt'
    if len(sys.argv) != 4:
        print(USAGE)
        sys.exit(1)
    else:
        inFile = sys.argv[1]  # input file
        numDims = int(sys.argv[2])  # desired number of dimension after pca
        outFile = sys.argv[3]  # output file

    # import data
    inputData = genfromtxt(inFile, delimiter='\t')
    # print("inputData: ", np.size(inputData,1))

    # plot data
    if np.size(inputData, 1) == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(inputData[:, 0], inputData[:, 1], inputData[:, 2])
        plt.show(block=False)
        plt.pause(1)
        plt.savefig('3dData.png')
        plt.close()

    if np.size(inputData, 1) < numDims:
        raise ValueError('Cannot perform PCA because desired dimension is greater than original dimension')

    # principalComps = decomposeComponents(inputData, numDims)
    principalComps = pca(inputData, numDims)

    for count, eigVal, eigVec in principalComps:
        print('Principal Component Vector {0}: {1} with eigenvalue: {2}'.format(count, eigVec, eigVal))

    # np.set_printoptions(suppress=True)
    with open(outFile, 'w') as file:
        for old in inputData:
            transformedData = PCAtransform(old, principalComps)
            file.write('PCA: {0} -> {1}'.format(old, transformedData))
            file.write('\n')