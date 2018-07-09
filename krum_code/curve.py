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

file = './result/bgd/dist_data/machine20/no_fault/batch3_alpha0.1_sqrt/x_star_norm.txt'
x_star_norm = file2list(file)
num_iter = len(x_star_norm)
plt.plot(np.arange(num_iter), x_star_norm)
plt.xlabel('iter')
plt.ylabel('||x_k - x*||')
plt.title('alpha0.01_batch1')
plt.savefig('./result/bgd/dist_data/machine20/no_fault/batch3_alpha0.1_sqrt/star_norm.jpg')
plt.show()

# file1 = './result/bgd/dist_data/machine5/alpha0.04_batch1/x_star_norm.txt'
# # file2 = './result/bgd/dist_data/machine5/alpha0.01_batch5/x_star_norm.txt'
# file3 = './result/bgd/dist_data/machine5/alpha0.03_batch5//x_star_norm.txt'
# batch1 = file2list(file1)
# # batch3 = file2list(file2)
# batch5 = file2list(file3)
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
# num_iter = len(batch5)
# plt.semilogy(np.arange(num_iter), batch1, color='green', label='batch1')
# plt.semilogy(np.arange(num_iter), batch3, color='red', label='batch3')
# plt.semilogy(np.arange(num_iter), batch5, color='blue', label='batch5')
# plt.semilogy(np.arange(num_iter), lambda1_0_01a, color='green', linestyle='-', label='lambda_0.01')
# plt.semilogy(np.arange(num_iter), lambda1_0_1a, color='red', linestyle='-', label='lambda_0.2')
# plt.semilogy(np.arange(num_iter), lambda1_0_2a, color='blue', linestyle='-', label='lambda_0.5')
# plt.semilogy(np.arange(num_iter), no_l1_star, color='purple', linestyle='-', label='no_l1_alpha0.4')
# # plt.semilogy(np.arange(num_iter), lambda1_0_9, color='yellow', label='lambda_0.9')
# plt.legend()
# plt.xlabel('iter')
# plt.ylabel('log||x_k - x*||')
# plt.show()
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


