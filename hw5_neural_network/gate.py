import numpy as np

"""
Class to represent computational unit
General: 
Forward: given input x and y, the neuron emits output z
Backward: given x, y and backward loss dL/dz, the neuron calculates dL/dx, dL/dy
dL/dx = dL/dz * dz/dx
dL/dy = dL/dz * dz/dy
"""

"""
Multiply: 
Multiply forward: Z = X * W
Multiply backward: given W X and dZ
"""
class MultiplyGate:
    # Z = X*W with X = numPt * numDim and W =  numDim*
    def forward(self, W, X):
        return np.dot(X, W)

    def backward(self, W, X, dLdZ):
        # here dW represents dL/dW
        # dL/dW = dL/dZ * dZ/dW = dL/dZ * X
        dLdW = np.dot(np.transpose(X), dLdZ)
        dLdX = np.dot(dLdZ, np.transpose(W))
        return dLdW, dLdX

"""
Add: 
Add forward: sum = S =  Z+b
Add backward: given Z b and dL/dS
dL/dZ = dL/dS * dS/dZ = dL/dS * 1
"""
class AddGate:
    def forward(self, Z, b):
        return Z + b

    def backward(self, Z, b, dLdS):
        # Return an array of ones with the same shape and type as a given array.
        dLdZ = dLdS * np.ones_like(Z)
        dLdb = np.dot(np.ones((1, dLdS.shape[0]), dtype=np.float64), dLdS)
        return dLdb, dLdZ