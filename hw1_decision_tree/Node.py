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
        if(isinstance(branchValDict, dict)):
            self.children = branchValDict.keys()
    
    