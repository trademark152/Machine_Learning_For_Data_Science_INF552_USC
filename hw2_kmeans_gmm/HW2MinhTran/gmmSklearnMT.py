"""
INF 552 - Homework 2 - Minh Tran
Gaussian Mixture Model using Scikit-learn
"""
# Import library
import sys
from numpy import genfromtxt
from sklearn.mixture import GaussianMixture

"""
MAIN
"""
if __name__ == '__main__':
    # import input
    USAGE = 'run command: python gmmSklearnMT.py clusters.txt 3'
    if len(sys.argv) != 3:
        print(USAGE.format(sys.argv[0]))
        sys.exit(1)
    else:
        inFile = sys.argv[1]
        numClusters = int(sys.argv[2])

    # import data
    data = genfromtxt(inFile, delimiter = ',')
    
    # create gmm model
    gmm = GaussianMixture(n_components=numClusters, tol=0.001,max_iter=500,init_params='kmeans')
    
    # fit to data
    gmm.fit(data)
    
    # output 
    for idx, (means, covariances, weights) in enumerate(zip(gmm.means_, gmm.covariances_, gmm.weights_)):
        print('Gaussian Mixture {0}:\nMean:\t\t{1}\nCovariance:\t{2}\nWeight:\t\t{3}'.format(idx, means, covariances, weights))
        print('-' * 40)