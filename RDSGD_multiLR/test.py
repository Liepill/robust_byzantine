import numpy as np
# import random
import matplotlib.pyplot as plt

file1 = './result/RSGD/fault/q8/lam0.5_wei0.01_alpha0.0001_sqrt_2/theta0_li_diff.npy'
file2 = './result/RSGD/fault/q8/lam0.5_wei0.01_alpha0.0001_sqrt_2/theta_li_diff.npy'
theta0 = np.load(file1)
theta_li = np.load(file2)
num_iter = theta0.shape[0]

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.semilogy(np.arange(num_iter), theta_li[:, 1], color='green', label='theta2_diff')
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.2), fontsize=14)
plt.xlabel('iter', fontsize=14)
plt.ylabel('log||theta2(k)-theta2(k-1)||', fontsize=14)
plt.show()

# print theta_li.shape
# print theta0[4000]
