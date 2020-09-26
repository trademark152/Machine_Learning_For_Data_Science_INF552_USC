"""
INF 552 - Homework 2 - Minh Tran
K-means algorithm
"""
from collections import defaultdict
import random
import numpy as np
from numpy import genfromtxt

"""
Function to compute euclidean distance between 2 points: sqrt of sum squared
"""
def calcDistance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

"""
Function to update centroids
Input is the dataset: list of lists and centroids: list of lists
"""
def updateCentroids(data, centroids):
    # initialize dict of clusters: cluster_index: {data point}
    clustersDict = defaultdict(list)

    # loop through each data points
    for i in range(0, len(data)):
        distance = []
        # loop through each centroids to calculate distance to that centroid
        for x in centroids:
            distance.append(calcDistance(data[i], x))
        # print("distance: ", distance)

        # find the index of the centroid with min distance to the point
        clusterIdx = distance.index(min(distance))

        # val, clusterIdx = min((val, idx) for (idx, val) in enumerate([compute_distance(data[i], x) for x in centroids]))

        # update the dict with cluster and associated data point
        clustersDict[clusterIdx].append(data[i])

    # print("clusters: ", clustersDict)

    newCentroids = []
    for cluster in clustersDict:
        # print("cluster: ", cluster)
        # update the centroid for each cluster by taking the mean of all associated points
        newCentroid = np.array(clustersDict[cluster]).mean(axis=0)
        newCentroids.append(newCentroid)

    return newCentroids

"""
MAIN
"""
if __name__ == "__main__":
    # initialization
    # finalCentroids = None
    maxIter = 100
    error = 0.01
    numCluster = 3
    numIter = 1

    # generate input data
    data = np.ndarray.tolist(genfromtxt('clusters.txt', delimiter=','))
    # print(len(data))

    # randomly select nodes as the original centroids
    finalCentroids = np.array(random.sample(data, numCluster))

    while True:
        newCentroids = np.array(updateCentroids(data, finalCentroids))
        # print("current centroids: ", newCentroids)

        # if max number of iteration exceeds, or new centroids completely equals old ones, or error exceeds
        if numIter >= maxIter or np.array_equal(finalCentroids, newCentroids) or (np.abs(newCentroids - finalCentroids) < error).all():
            print("number of interations:",numIter)
            print("Cluster Centroids: ", newCentroids)
            break

        numIter += 1
        finalCentroids = newCentroids