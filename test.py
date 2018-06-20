import numpy as np
import random
import matplotlib.pyplot as plt

# dict = [i for i in range(100)]
# list = [0 for _ in range(100)]
# for i in range(100000):
#     # id = random.choice(dict)
#     id = random.randint(0, 99)
#     list[id] = list[id] + 1
#
# plt.plot(np.arange(100), list)
# plt.show()

# for i in range(5):
#     print i
# 182.285800872

A = np.load('./data/A.npy')
b = np.load('./data/b.npy')
# x = np.arange(50).reshape((50, ))
# print "x shape:", x.shape
# t = np.dot(A, x) - b
# t = t.reshape((100, 1))
# print t.shape
# t1 = np.zeros_like(A)
# tmp = t * A
# print tmp.shape
#
# grad = tmp.sum(axis=0) / A.shape[0]
# print grad.shape
# print np.linalg.norm(grad)

a = A[19*5:20*5]
print a.shape