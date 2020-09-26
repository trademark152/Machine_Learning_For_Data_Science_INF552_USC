"""
INF 552 - Homework 2 - Minh Tran
K-means using Scikit-learn
"""
import sys
import numpy as np
from sklearn.cluster import KMeans

"""
MAIN
"""
if __name__ == '__main__':
    USAGE = 'run command: python kmeansSklearnMT.py clusters.txt 3'
    if len(sys.argv) != 3:
        print(USAGE.format(sys.argv[0]))
        sys.exit(1)
    else:
        inFile = sys.argv[1]
        numClusters = int(sys.argv[2])

    # import data
    data = np.genfromtxt(inFile, delimiter = ',')

    # create model and fir to data
    kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(data)

    # print output
    for idx, centroid in enumerate(kmeans.cluster_centers_):
        print('Cluster {0} - centroid {1}'.format(idx, centroid))
