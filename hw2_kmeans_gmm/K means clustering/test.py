import numpy as np
gamma_nk = np.array([[0.5, 0.2, 0.3],[0.4, 0.3, 0.3],[0.5, 0.2, 0.3]])

x = [[1,1],[1,4],[3,2]]

for i in range(0, 3):
    print(gamma_nk[:, i])
    print(np.multiply(x, gamma_nk[:, i].reshape(len(x), 1)))
    num = np.sum(np.multiply(x, gamma_nk[:, i].reshape(len(x), 1)), axis=0)
    print(num)