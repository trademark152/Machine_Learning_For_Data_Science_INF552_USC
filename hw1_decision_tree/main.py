# import DecisionTree
# import Node
import math
"""
Create a node in a decision tree
2 properties are the root attribute and children: branches' values
"""


class Node:
    attribute = ""
    children = []

    # initialization
    def __init__(self, att, branchValDict):
        self.setValue(att)
        self.makeChildren(branchValDict)

    # print node
    def __str__(self):
        return str(self.attribute)

    # set root attribute
    def setValue(self, att):
        self.attribute = att

    # generate branches
    def makeChildren(self, branchValDict):
        if (isinstance(branchValDict, dict)):
            self.children = branchValDict.keys()


"""
FIND most common value for the target attribute:
INPUT: training data and index of the target attribute
"""


def majority(data, idx):
    # find target attribute:
    valCount = {}

    # find target in data
    # idx = attributes.index(target)

    # calculate frequency of values in target attr
    for tuple in data:
        if tuple[idx] in valCount:
            valCount[tuple[idx]] += 1
        else:
            valCount[tuple[idx]] = 1

    max = 0
    majorityVal = ""
    for key in valCount.keys():
        if valCount[key] > max:
            max = valCount[key]
            majorityVal = key

    return majorityVal


"""
Calculates the entropy of the given data set for the target attr
"""


def entropy(attributes, data, targetAttr):
    valCount = {}
    dataEntropy = 0.0

    # find index of the target
    i = attributes.index(targetAttr)
    print("i:", i)

    # Calculate the frequency of each of the values in the target attr
    print("data: ", data)
    for entry in data:
        if entry[i] in valCount:
            valCount[entry[i]] += 1.0
        else:
            valCount[entry[i]] = 1.0
    print("valCount2: ", valCount)

    # Calculate the weighted entropy of the data for the target attr
    for count in valCount.values():
        dataEntropy += (-count / len(data)) * math.log(count / len(data), 2)

    return dataEntropy


"""
Calculates the information gain (reduction in entropy) that would
result by splitting the data on the chosen attribute (attr).
"""


def gain(attributes, data, attr, targetAttr):
    valCount = {}
    subsetEntropy = 0.0

    # find index of the chosen attribute to split
    i = attributes.index(attr)

    # Calculate the frequency of each of the values in the target attribute
    for entry in data:
        if entry[i] in valCount:
            valCount[entry[i]] += 1.0
        else:
            valCount[entry[i]] = 1.0
    print("valCount:", valCount)

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    # weighted entropy = sum(w*entropyEachBranch)
    # entropyEachBranch = -sum(P*logP)
    # example valCount = {overcast: 4, rainy: 5, sunny: 5} for attribute "outlook"
    for val in valCount.keys():
        # calculate probability of each occurrence (weight w)
        valProb = valCount[val] / sum(valCount.values())

        # extract subset data of each branch
        dataSubset = [entry for entry in data if entry[i] == val]

        # calculate weighted entropy of this split
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)

    # To calculate the information gain, subtract the entropy of the chosen attribute from the entropy of the whole data set with respect to the target attribute (and return it)
    ig = entropy(attributes, data, targetAttr) - subsetEntropy
    return ig


# choose best attribute to split base on information gain
def chooseAttr(data, attributes, target):
    # extract list of attribute excluding the target attribute
    attributesTemp = [n for n in attributes if n != target]

    # initialize the best attribute and max ig
    best = attributesTemp[0]
    maxGain = 0

    # loop through all attributes
    for attr in attributesTemp:
        # calculate ig
        newGain = gain(attributes, data, attr, target)
        if newGain > maxGain:
            maxGain = newGain
            best = attr
    return best


"""
get unique values in the column of the given attribute 
"""


def getValues(data, attributes, attr):
    # get the index of chosen attribute to split
    index = attributes.index(attr)

    values = []
    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])
    return values


"""
Get all entries that fall under the "best" attribute and a particular "val"
"""


def getSubdata(data, attributes, best, val):
    # initialize
    subdata = []

    # get the index of the best attribute
    index = attributes.index(best)
    for entry in data:
        # find entries with the given value
        if (entry[index] == val):
            subEntry = []
            # add value if it is not in best column
            for i in range(0, len(entry)):
                if (i != index):
                    subEntry.append(entry[i])
            subdata.append(subEntry)
    print("subdata: ", subdata)
    return subdata


"""
MAIN function to make the tree
"""


def makeTree(data, attributes, target, recursion):
    recursion += 1

    # Returns a new decision tree based on the examples given.
    data = data[:]

    # find index of target attribute:
    idx = attributes.index(target)

    # collect all values of target attribute for each data entry
    vals = [entry[idx] for entry in data]
    print("vals: ", vals)

    # find the majority of target attribute
    default = majority(data, idx)
    print("default: ", default)

    # If the dataset is empty or the attributes list (excluding target attribute by "-1") is empty, return the default value.
    if not data or (len(attributes) - 1) <= 0:
        return default

    # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = chooseAttr(data, attributes, target)
        print("best: ", best)

        # Create a new decision tree/node with the best attribute and an empty dictionary object--we'll fill that up next.
        tree = {best: {}}
        print("updated tree: ", tree)

        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in getValues(data, attributes, best):
            # Create a subtree for the current value under the "best" field
            newData = getSubdata(data, attributes, best, val)
            newAttr = attributes[:]

            # remove the best attribute to recursively make tree
            newAttr.remove(best)

            # recursively making new tree after removing the attribute
            subtree = makeTree(newData, newAttr, target, recursion)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree
            print("updated tree: ", tree)
    return tree


def main():
    """
    USER: Change this file path to change training data
    """
    trainFile = open('dt_data.csv')

    """
    USER: Change this variable to change target attribute 
    """
    target = "Enjoy"

    # import data
    trainData = []
    for line in trainFile:
        line = line.strip("\r\n")
        trainData.append(line.split(','))
    print("train Data: ", trainData)

    # import data and extract attributes
    attributes = trainData[0]
    trainData.remove(attributes)

    #Run ID3
    tree = makeTree(trainData, attributes, target, 0)

    print("Decision Tree generated")
    print("FINAL TREE: ", tree)

    """
    IMPORTANT USER: Change this file path to change testing data 
    """
    testData = []
    testFile = open('Weather1.csv')

    # gather data
    for line in testFile:
        line = line.strip("\r\n")
        testData.append(line.split(','))
    print("test Data: ", testData) # last column is blank: need prediction

    count = 0
    for testDataEntry in testData:
        # print("testDataEntry:", testDataEntry)
        count += 1
        # create a copy to truncate the tree gradually
        # tree is structured in a sequence: {attribute: {value1:{attribute1,...}, value2:...}}}
        tempDict = tree.copy()
        result = ""

        # loop until the temp dictionary is not empty
        while (isinstance(tempDict, dict)):
            firstUpAttribute = list(tempDict.keys())[0]

            # 1st Node is from the script, 2nd Node is from the class within the script
            root = Node(firstUpAttribute, tempDict[firstUpAttribute])
            print("root", root.attribute, root.children)

            # traverse down the tree
            tempDict = tempDict[firstUpAttribute]

            # find the index of that attribute in the original attribute list
            idx = attributes.index(root.attribute)
            # print("idx:", idx)

            # find the corresponding value in test data
            value = testDataEntry[idx]
            print("value: ", value)
            print("temp dict keys: ", tempDict.keys())

            # check if attribute value exist in the decision branches
            if value in tempDict.keys():
                # create a child node if needed
                # child = Node.Node(value, tempDict[value])

                result = tempDict[value]
                # print("result: ", result)

                tempDict = tempDict[value]

            else:
                print("Decision Tree can't process input %s" % count)
                result = "N/A"
                break

        print("test Data Entry %s = %s" % (count, result))
    
if __name__ == '__main__':
    main()