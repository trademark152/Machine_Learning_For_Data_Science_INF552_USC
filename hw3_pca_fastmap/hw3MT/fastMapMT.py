"""
INF 552 - Homework 3 - Minh Tran
FastMap Algorithm
"""

'''
python fastMapMT.py fastmap-data.txt 10 2 fastmap-wordlist.txt
'''

import copy
import numpy as np
import matplotlib.pyplot as plt
import sys

print(sys.version)

"""
Get 2 pivot points that are furthest from each other
"""
def getFarthestPoints(distancesMat):
    numPts = len(distancesMat)

    # initialize the first pivot
    firstPivot = np.random.randint(0, numPts)
    while True:
        # get the furthest point from the first pivot
        secondPivot = np.argmax(distancesMat[firstPivot])

        # take the second pt as pivot, get the furthest point from it
        tempPivot = np.argmax(distancesMat[secondPivot])
        if tempPivot == firstPivot:
            break
        else: # if not converged, recursive
            firstPivot = secondPivot

    # return result by ordering the min index first
    farthestPts = (min(firstPivot, secondPivot), max(firstPivot, secondPivot))
    return farthestPts

"""
get coordinate when projecting a point onto a line
Input: distancesMat Matrix, line [pt1 pt2] and the outer point
"""
def getProjectCoord(distancesMat, projectLine, outerPt):
    # extract the two endpoints
    firstEndPt, secondEndPt = projectLine
    
    # calculate the projecting coordinate of the outer point: eq 3 in reference fastmapAlgo.pdf
    # xi = (Dai^2 + Dab^2 - Dbi^2)/(2*Dab)
    coord = (distancesMat[firstEndPt][outerPt]**2 + distancesMat[firstEndPt][secondEndPt]**2 - distancesMat[secondEndPt][outerPt]**2)/(2 * distancesMat[firstEndPt][secondEndPt])
    return coord

"""
redefine distances matrix
Input: old distances matrix,
"""
def recalcDistanceMat(oldDistancesMat, latestCoords):
    # store the distance matrix separately for later use
    newDistancesMat = copy.deepcopy(oldDistancesMat)
    
    numPts = len(oldDistancesMat)
    
    # update the new distance matrix
    for i in range(numPts):
        for j in range(numPts):
            # formula 4 from ref: D'(Oi, Oj) = sqrt(D(Oi, Oj)^2 - (xi-xj)^2)
            newDistancesMat[i][j] = np.sqrt(oldDistancesMat[i][j]**2 - (latestCoords[i] - latestCoords[j])**2)
    return newDistancesMat

"""
FASTMAP algorithm
Input: distancesMat: 2d array and numDims
Output: embedding result
"""
def fastMap(distancesMat, numDims):
    # initialize result as list of coordinates of each point
    embedCoords = [[] for k in distancesMat]

    # looping through each dimension to embed
    for k in range(numDims):
        # get 2 points that are furthest apart and the corresponding distance
        firstPivot, secondPivot = getFarthestPoints(distancesMat)

        # for each point
        for pt in range(len(distancesMat)):
            if pt == firstPivot:
                dist = 0
            elif pt == secondPivot:
                dist = distancesMat[firstPivot][secondPivot]
            else:
                dist = getProjectCoord(distancesMat, (firstPivot, secondPivot), pt)
            # print("dist: ", dist)
            embedCoords[pt].append(dist)

        # extract the most recent coordinate of all data points
        latestCoords = list(map(lambda dat: dat[-1], embedCoords))
        # print("latestCoords: ", latestCoords)

        # recompute distancesMat matrix
        distancesMat = recalcDistanceMat(distancesMat, latestCoords)

    return embedCoords

"""
MAIN
"""
if __name__ == '__main__':
    USAGE = 'python FastMap.py fastmap-data.txt 10 2 fastmap-wordlist.txt'
    if len(sys.argv) != 5:
        print(USAGE)
        sys.exit(1)
    else:
        fileInput = sys.argv[1] # input file with distancesMat
        numPts = int(sys.argv[2]) # number of data points
        numDims = int(sys.argv[3]) # number of desired dimensions to embed
        fileWord = sys.argv[4] # input file with words

    # import data as list of lists
    # create a 2d array distance matrix
    distancesMat = np.zeros((numPts, numPts))
    with open(fileInput) as file:
        for line in file:
            firstObj, secondObj, dist = line.split()
            distancesMat[int(firstObj)- 1][int(secondObj) - 1], distancesMat[int(secondObj) - 1][int(firstObj) - 1] = float(dist), float(dist)
    # print("distancesMat matrix: ", distancesMat)

    # run fastmap algorithm
    embedCoords = fastMap(distancesMat, numDims)
    # print("embedding coordinates: ", embedCoords)

    # import word list
    wordList = []
    with open(fileWord) as file2:
        wordList = list(map(str.strip, file2.readlines()))

    # print results of embedding each word: because sequence of words is aligned with its index in the coordinate
    for word, coOrd in zip(wordList, embedCoords):
        print('Embedding word "{0}" to 2-D target space: {1}'.format(word, coOrd))

    # plotting with 2d dimensions
    if numDims == 2:
        embedCoords = np.asarray(embedCoords)

        fig, ax = plt.subplots()
        ax.scatter(embedCoords[:, 0], embedCoords[:, 1])
        for idx, word in enumerate(wordList):
            ax.annotate(word, (embedCoords[idx]))

        plt.show(block=False)
        plt.pause(1)
        plt.savefig('wordEmbedding.png')
        plt.close()
    else:
        print('This script only valid for 2d plotting')
