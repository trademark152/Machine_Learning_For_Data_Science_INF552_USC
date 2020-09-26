import numpy as np

"""
Sigmoid:
Forward: sigmoid(z) = s = 1/(1+exp(-z))
Backward: given dL/ds, calculate dL/dz
dL/dz = dL/ds * ds/dz = dL/ds * s * (1-s)
"""
class Sigmoid:
    def forward(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def backward(self, X, dLds):
        s = self.forward(X)
        return (1.0 - s) * s * dLds


"""
Tanh:
Forward: t = tanh(z)
Backward: given dL/dt, calculate dL/dz
dL/dz = dL/dt * dt/dz = dL/dt * (1-t^2)
"""
class Tanh:
    def forward(self, X):
        return np.tanh(X)

    def backward(self, X, dLdt):
        t = self.forward(X)
        return (1.0 - np.square(t)) * dLdt