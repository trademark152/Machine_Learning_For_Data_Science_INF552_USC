"""
Test utils functions
"""
import numpy as np
from utils import *
import cvxopt

X = np.random.rand(10,3)

print(X)
Xnorm = normalize(X, -1,2)
print("normalized X:", Xnorm)
Xstandard = standardize(X)
print("standardized X: ", Xstandard)

Xint = np.random.randint(10, size=(10, 3))
print(Xint)
linear_kernel = linear_kernel()
print("linear kernel:", linear_kernel(Xint[0],Xint[1]))

poly_kernel = polynomial_kernel(power=2, coef=1.)
print("2nd order polynomia kernel:", poly_kernel(Xint[0],Xint[1]))

rbf_kernel = rbf_kernel(gamma=0.1)
print("rbf kernel:", rbf_kernel(Xint[0],Xint[1]))

Xint2 = np.random.randint(5, size=(2, 2))
print(Xint2)
print("covariance matrix: ", calculate_covariance_matrix(Xint2))

"""
CVXOPT
solving a linear program:
minimize c*X subject to A*x <= b
Here: minimize y = 2*x1 + x2
subject to:
-x1 + x2 <= 1
-x1 - x2 <= -2
-x2 <= 0
x1 - 2*x2 <= 4
"""
from cvxopt import matrix, solvers
# made up by 2 columns of coefficient with x1, x2....
A = matrix([[-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0]])
print("A: ", A)

b = matrix([ 1.0, -2.0, 0.0, 4.0 ])
print("b: ", b)

c = matrix([2.0, 1.0])
lin_sol = solvers.lp(c, A, b)
print("optimal solution of x: ", lin_sol['x'])
# print(lin_sol)
print("optimal solution of y: ", lin_sol['primal objective'])


"""
CVXOPT
solving a nonlinear/quadratic program:
minimize 1/2* xT * P * x + qT * x subject to G*x <= h and A*x = b
convex if and only if P is PSD
 P and q are required, the others are optional
Here: minimize y = 1/2*x1^2 + 3*x1 + 4*x2
subject to:
-x1 <= 0
-x2 <= 0
-x1-3*x2 <=-15
2*x1+5*y2 <= 100
3*x1 + 4*x2 <= 80
x1+x2 = 5

"""
# Define QP parameters (directly)
Q = matrix([[1.0, 0.0], [0.0, 0.0]])
p = matrix([3.0, 4.0])

G = matrix([[-1.0, 0.0, -1.0, 2.0, 3.0], [0.0, -1.0, -3.0, 5.0, 4.0]])
h = matrix([0.0, 0.0, -15.0, 100.0, 80.0])
A = matrix([1.0, 1.0], (1, 2))
b = matrix(5.0)

# Define QP parameters (with NumPy)
# P = matrix(np.diag([1,0]), tc='d')
# q = matrix(np.array([3.0, 4.0]), tc='d')
# G = matrix(np.array([[-1,0],[0,-1],[-1,-3],[2,5],[3,4]]), tc='d')
# h = matrix(np.array([0,0,-15,100,80]), tc='d')

quad_sol = solvers.qp(Q, p, G, h, A, b)

"""
minimize 1/2* xT * Q * x + pT * x subject to G*x <= h and A*x = b
Here: minimize y =2*x1^2 + x2^2 + x1*x2 + x1 + x2 
subject to:
-x1 <= 0
-x2 <= 0
x1+x2 = 1
"""
Q = 2*matrix([[2, .5], [.5, 1]])
p = matrix([1.0, 1.0])
G = matrix([[-1.0, 0.0], [0.0, -1.0]])
h = matrix([0.0, 0.0])
A = matrix([1.0, 1.0], (1, 2))
b = matrix(1.0)
quad_sol2 = solvers.qp(Q, p, G, h, A, b)

print("optimal solution of x: ", quad_sol2['x'])
# print(quad_sol2)
print("optimal solution of y: ", quad_sol2['primal objective'])