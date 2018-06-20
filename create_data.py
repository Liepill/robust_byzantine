import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)
#
row = 1000
col = 50
A = np.random.normal(0, 1.0, size=(row, col))
y = np.random.uniform(-2.0, 2.0, size=(col,))
noise = np.random.normal(0, 0.1, size=(row,))
b = np.dot(A, y) + noise
print b.shape
np.save('./data/A2.npy', A)
np.save('./data/y2.npy', y)
np.save('./data/noise2.npy', noise)
np.save('./data/b2.npy', b)

# b = np.lo
# A = np.load('./data/A.npy')
# b = np.load('./data/b.npy')
# x_star = np.load('./result/SGD/x_star.npy')
# grad = []
# for i in range(len(b)):
#     tmp = (np.dot(A[i], x_star) - b[i]) * A[i]
#     grad.append(tmp)
# grad = np.array(np.fabs(grad))
# print np.max(grad)

