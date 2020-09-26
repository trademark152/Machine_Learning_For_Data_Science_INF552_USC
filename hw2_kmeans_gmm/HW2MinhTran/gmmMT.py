"""
INF 552 - Homework 2 - Minh Tran
Gaussian Mixture Model
"""
from collections import defaultdict
import numpy as np
from numpy import genfromtxt
import random

amplitude =[]
means =[]
covariance =[]
"""
Expectation Step: update the soft assignment" (fixing parameters)
"""
def expectationAlgo(data, numClusters):
    N = len(data)
    probs = np.empty([N, numClusters]) # density function
    dimension = len(data[0])

    # calculate probs/probability of each point in a particular cluster
    # loop through each cluster
    for i in range(0, numClusters):
        mean = means[i] # means
        cov = covariances[i] # covariance

        # compute the inverse of the covariance matrix
        invCov = np.mat(np.linalg.inv(cov))

        # multivariate gaussian/normal distribution:
        # f(x1,...xk) = 1/(sqrt(2*pi)^k * det(cov)) * exp (-1/2*((x-mean)*invcov*(x-mean))
        coeff = 1 / np.sqrt((2 * np.pi) ** dimension * np.linalg.det(cov))

        # loop through each data point
        for n in range(0, N):
            temp1 = np.mat(data[n, :] - mean)
            temp2 = coeff * np.exp(-0.5 * temp1 * invCov * temp1.T)
            # print("val:", val)
            probs[n][i] = temp2[0][0]
    # print("probs: ", probs)

    # updating membership weight: gamma_nk = weight * prob / sum(weight*prob)
    gammaNK = np.empty([N, numClusters])
    for i in range(0, N):
        deno = np.sum(np.multiply(amps, probs[i]))
        for j in range(0, numClusters):
            gammaNK[i][j] = amps[j] * probs[i][j] / deno
    # print("weights: ", weights)
    return gammaNK

"""
Maximization Step: Update the model parameter (fixing assignment)
"""
def maximizationAlgo(data, gammaNK, numClusters):
    global means, covariances, amps

    # reset parameters
    means = []
    covariances = []
    amps = []

    N = len(data) # number of data points
    dimension = len(data[0]) # number of dimensions of each data point

    # at starting point
    if gammaNK is None:
        # initialize the weights to be equal at first
        amps = np.ones(numClusters) / numClusters
        amps = amps.tolist()

        # initialize clusters based on k-means approach
        clusters = pseudokmeans(numClusters, data)
        for i in range(0, numClusters):
            # update parameters
            means.append(np.mean(clusters[i], axis=0))

            # for each cluster, there are k points in 2 dimensions --> np.cov generate a 2*2 cov matrix for each cluster
            covariances.append(np.cov(clusters[i].T))

        print("Initial weights: ", amps)
        print("Initial Means: ", np.array(means))
        print("Initial Covariances: ", np.array(covariances))
    # if membership weight is initiated as a 2D np array N rows (# data pts) and numClusters columns (clusters)
    else:
        # Lecture 10.pdf: wk = sum(gamma_nk)/N
        amps = np.sum(gammaNK, axis=0) / N

        for i in range(0, numClusters):
            # update means: muy_k = sum(gamma_nk*xn)/sum(gamma_nk)
            # print("length of data: ", len(data))
            # print("membership weight: ", len(gammaNK[:,i]))

            # np.multiply ensures component wise multiplication between each data point Xi (xi1, xi2) and its membership weight (gamma_ik) to each cluster
            numeMeans = np.sum(np.multiply(data, gammaNK[:, i].reshape(len(data), 1)), axis=0)
            denoMeans = sum(gammaNK[:, i])
            means.append(numeMeans/denoMeans)

            # update covariance sigma_k = 1/sum(gamma_nk)*sum(gamma_nk*(x_n-muy_k)*(x_n-muy_k)
            numeCov = np.zeros((dimension, dimension)) # size of covariance matrix for each

            # numerator sum(gamma_nk*x_n): loop through each data point
            for j in range(0, N):
                # subtract the mean of cluster from each data point: (x_n - muy_k)
                temp = data[j] - means[i]

                # gamma_nk * dot product
                numeCov += gammaNK[j][i] * np.dot(temp.T.reshape(dimension, 1), temp.reshape(1, dimension))

            # denominator: sum(gamma_nk)
            denoCov = np.sum(gammaNK[:, i])
            covariances.append(numeCov/denoCov)
    return 
"""
k means
"""
def pseudokmeans(numClusters, data):
    # centroids = np.array(random.sample(data.tolist(), numClusters))
    # print(centroids)

    centroids = np.array(data.tolist())
    centroids = centroids[:3]
    print(centroids)

    # initialize dict of clusters: cluster_index: {data point}
    clustersDict = defaultdict(list)
    for i in range(0, len(data)):
        distance = []
        # loop through each centroids to calculate distance to that centroid
        for x in centroids:
            distance.append(calcDistance(data[i], x))
        # print("distance: ", distance)

        # find the index of the centroid with min distance to the point
        clusterIdx = distance.index(min(distance))

        # update the dict with cluster and associated data point
        clustersDict[clusterIdx].append(data[i].tolist())

    clustersDict = [np.array(clustersDict[i]) for i in clustersDict]
    return clustersDict

"""
Calculate distance
"""
def calcDistance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

"""
MAIN
"""
if __name__ == "__main__":
    # generate input data
    data = genfromtxt('clusters.txt', delimiter=',')

    # set parameters to converge
    error = 0.001
    maxIter = 500
    numClusters = 3
    count = 1
    current_weights = None

    while True:
        # maximization step with current weights
        maximizationAlgo(data, current_weights, numClusters)
        # print(means)
        # print(covariance)
        
        # update new weights with expectation step
        new_weights = expectationAlgo(data, numClusters)

        # print('current_membership_weights', current_weights)

        # if exceed max iterations then stop
        if count >= maxIter:
            print("...number of iteration reached error")
            break

        # if weight change does not exceeed error , also stop
        if current_weights is not None and new_weights is not None and (np.abs(new_weights - current_weights) < error).all():
            print("...weights converged")
            break

        count += 1
        current_weights = new_weights

    print("Number of iterations: ", count)
    print("Current weights: ", amps)
    print("Means: ", np.array(means))
    print("Covariances: ", np.array(covariances))

