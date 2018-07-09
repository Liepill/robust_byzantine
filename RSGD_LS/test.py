import numpy as np
import random
import matplotlib.pyplot as plt

A = np.load('./data/A.npy')
b = np.load('./data/b.npy')
x_star = np.load('./result/GD/correct/machine5/alpha0.002/x_star.npy')
#
#
# s1 = 'lam0.5_alpha0.1_sqrt'
# x0 = np.load('./result/RDSGD/fault/5/' + s1 + '/x0_li.npy')
# print x0.shape
# m = x0.shape[0]
# x_star_norm = []
# for i in range(m):
#     x_star_norm.append(np.linalg.norm(x0[i] - x_star))
#
# np.save('./result/RDSGD/fault/5/' + s1 + '/x_star_norm.npy', np.array(x_star_norm))
# np.savetxt('./result/RDSGD/fault/5/' + s1 + '/x_star_norm.txt', x_star_norm)
#
# plt.semilogy(np.arange(m), x_star_norm)
# plt.xlabel('iter')
# plt.ylabel('log||x0 - x*||')
# plt.title(s1)
# plt.savefig('./result/RDSGD/fault/5/' + s1 + '/x_star_norm.jpg')
# plt.show()

grad_all = np.zeros((80, 50))
for i in range(80):
    grad = (np.dot(A[i], x_star) - b[i]) * A[i]
    grad_all[i] = grad

print np.amax(grad_all)


