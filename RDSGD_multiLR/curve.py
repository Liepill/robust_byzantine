import numpy as np
# import random
import matplotlib.pyplot as plt

def file2list(filename):
    """read data from txt file"""
    fr = open(filename)
    arrayOLines = fr.readlines()
    returnMat = []
    for line in arrayOLines:
        line = line.strip()
        returnMat.append(float(line))
    return returnMat

# file = './result/SGD/wei0.01_alpha0.00001/grad_norm.npy'
# # x_star_norm = file2list(file)
# x_star_norm = np.load(file)
# num_iter = x_star_norm.shape[0]
# plt.plot(np.arange(num_iter), x_star_norm)
# plt.xlabel('iter')
# plt.ylabel('||grad||')
# plt.title('wei0.01_alpha0.00001')
# plt.savefig('./result/SGD/wei0.01_alpha0.00001/grad_norm2.jpg')
# plt.show()

file_no = './result/RSGD/no_fault/lam1.0_wei0.01_alpha0.0001_sqrt/acc.npy'
file1 = './result/RSGD/fault/q1/lam0.5_wei0.01_alpha0.0001_sqrt/acc.npy'
file2 = './result/RSGD/fault/q3/lam0.5_wei0.01_alpha0.0001_sqrt/acc.npy'
file3 = './result/RSGD/fault/q8/lam0.5_wei0.01_alpha0.0001_sqrt/acc.npy'
# file4 = './result/RSGD/fault/q10/lam0.5_wei0.01_alpha0.0001_sqrt/acc.npy'
# file5 = './result/RSGD/fault/q11/lam0.5_wei0.01_alpha0.0001_sqrt/acc.npy'
# file6 = './result/RSGD/fault/q12/lam0.5_wei0.01_alpha0.0001_sqrt/acc.npy'
file4 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/q1/alpha0.0001_sqrt_wei0.01/acc_li.txt'
file5 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/q3/alpha0.001_sqrt_wei0.01/acc_li.txt'
file6 = 'G:\python_code/byzantine/Krum/result/mnist/machine20/fault/q8/alpha0.001_sqrt_wei0.01/acc_li.txt'

no = np.load(file_no)
q1 = np.load(file1)
q3 = np.load(file2)
q8 = np.load(file3)
k1 = file2list(file4)
k3 = file2list(file5)
k8 = file2list(file6)
# q10 = np.load(file4)
# q11 = np.load(file5)
# q12 = np.load(file6)

num_iter = q1.shape[0]

# font_legend = {'weight': 'normal',
#                'size': '20'}
# font_axis = {'weight': 'normal',
#              'size': '20'}
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.plot(np.arange(num_iter)*10, no, color='green', label='no')
plt.plot(np.arange(num_iter)*10, q1, color='red', label='q1', marker='+', markevery=50)
plt.plot(np.arange(num_iter)*10, q3, color='blue', label='q3', marker='*', markevery=50)
plt.plot(np.arange(num_iter)*10, q8, color='green', label='q8', marker='d', markevery=50)
plt.plot(np.arange(num_iter)*10, k1, color='red', linestyle='-.', label='krum_q1', marker='h', markevery=50)
plt.plot(np.arange(num_iter)*10, k3, color='blue', linestyle='-.', label='krum_q3', marker='s', markevery=50)
plt.plot(np.arange(num_iter)*10, k8, color='green', linestyle='-.', label='krum_q8', marker='o', markevery=50)

# plt.plot(np.arange(num_iter)*10, q10, color='purple', label='q10')
# plt.plot(np.arange(num_iter)*10, q11, color='black', linestyle='-.', label='q11')
# plt.plot(np.arange(num_iter)*10, q12, color='yellow', linestyle='-.', label='q12')
# # plt.semilogy(np.arange(num_iter), lambda1_0_9, color='yellow', label='lambda_0.9')
plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.2), fontsize=20)
plt.xlabel('iter', fontsize=20)
plt.ylabel('accuracy', fontsize=20)
plt.show()
# num_iter = len(batch5)
# plt.semilogy(np.arange(num_iter), batch1, color='green', label='batch1')
# # plt.semilogy(np.arange(num_iter), batch3, color='red', label='batch3')
# plt.semilogy(np.arange(num_iter), batch5, color='blue', label='batch5')
# plt.legend()
# plt.xlabel('iter')
# plt.ylabel('log||x_k - x*||')
# plt.show()
# # file4 = './result/RDSGD/20/alpha1.0_lam0.01_sqrt/x_star_norm.txt'
# file1_1 = './result/RDSGD/fault/20/alpha0.5_lam0.01_sqrt/x_star_norm.txt'
# file2_1 = './result/RDSGD/fault/20/alpha0.3_lam0.2_sqrt/x_star_norm.txt'
# file3_1 =  './result/RDSGD/fault/20/alpha0.3_lam0.5_sqrt/x_star_norm.txt'
#
# file_no = './result/DSGD_param/fault/20/alpha0.1_sqrt/x_star_norm.txt'
#
# lambda1_0_01 = file2list(file1)
# lambda1_0_1 = file2list(file2)
# lambda1_0_2 = file2list(file3)
# # lambda1_0_9 = file2list(file4)
# lambda1_0_01a = file2list(file1_1)
# lambda1_0_1a = file2list(file2_1)
# lambda1_0_2a = file2list(file3_1)
#
# no_l1_star = file2list(file_no)
#
# grad1 = './result/RDSGD/no_fault/20/alpha0.5_lam0.01_sqrt/grad_norm.txt'
# grad2 = './result/RDSGD/no_fault/20/alpha0.3_lam0.2_sqrt/grad_norm.txt'
# grad3 = './result/RDSGD/no_fault/20/alpha0.3_lam0.5_sqrt/grad_norm.txt'
# # grad4 = './result/RDSGD/20/alpha1.0_la0.9_step/grad_norm.txt'
# grad1_1 = './result/RDSGD/fault/20/alpha0.5_lam0.01_sqrt/grad_norm.txt'
# grad2_1 = './result/RDSGD/fault/20/alpha0.3_lam0.2_sqrt/grad_norm.txt'
# grad3_1 = './result/RDSGD/fault/20/alpha0.3_lam0.5_sqrt/grad_norm.txt'
#
# grad_no = './result/DSGD_param/fault/20/alpha0.1_sqrt/grad_norm.txt'
#
# grad_0_01 = file2list(grad1)
# grad_0_1 = file2list(grad2)
# grad_0_2 = file2list(grad3)
# # grad_0_9 = file2list(grad4)
# grad_0_01a = file2list(grad1_1)
# grad_0_1a= file2list(grad2_1)
# grad_0_2a = file2list(grad3_1)
#
# no_l1_grad = file2list(grad_no)
#
#
#
# print len(lambda1_0_01)
# num_iter = len(lambda1_0_01)
#
# fig1 = plt.figure(1)

#
# fig2 = plt.figure(2)
# # plt.semilogy(np.arange(num_iter), grad_0_01, color='green', label='lambda_0.01_NoFault')
# # plt.semilogy(np.arange(num_iter), grad_0_1, color='red', label='lambda_0.2_NoFault')
# # plt.semilogy(np.arange(num_iter), grad_0_2, color='blue', label='lambda_0.5_NoFault')
# plt.semilogy(np.arange(num_iter), grad_0_01a, color='green', linestyle='-', label='lambda_0.01')
# plt.semilogy(np.arange(num_iter), grad_0_1a, color='red', linestyle='-', label='lambda_0.2')
# plt.semilogy(np.arange(num_iter), grad_0_2a, color='blue', linestyle='-', label='lambda_0.5')
# plt.semilogy(np.arange(num_iter), no_l1_grad, color='purple', linestyle='-', label='no_l1_alpha0.4')
# # plt.semilogy(np.arange(num_iter), grad_0_9, color='yellow', label='lambda_0.9')
# plt.legend()
# plt.xlabel('iter')
# plt.ylabel('log||grad||')
# plt.show()


