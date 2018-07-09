import numpy as np
import matplotlib.pyplot as plt
import random

A = np.load('./data/A.npy')
b = np.load('./data/b.npy')
# x_star = np.load('./data/y.npy')
x_star = np.load('./result/GD/x_star.npy')
# x_star = np.load('./result/GD/2/alpha0.001/x_star.npy')


def CalGrad(A, b, x):

    grad = np.zeros_like(x)
    grad = grad + (np.dot(A, x) - b) * A
    return grad

def CalTotalGrad(A, b, x):
    grad = np.zeros_like(x)
    for i in range(len(b)):
        grad = grad + (np.dot(A[i], x) - b[i]) * A[i]
    return grad

def sgd():
    num_iter = 1000
    alpha = 0.01
    # d = 0.001
    m = A.shape[0]
    x = np.zeros((A.shape[1],))
    x_li = []
    diff_x_star = []
    grad_norm = []
    grad_li = []
    x_li.append(x)
    k = 0
    for step in range(num_iter):
        # permutation = np.random.permutation(m)
        # shu_A = A[permutation, :]
        # shu_b = b[permutation]
        for id in range(m):
            # if (step * m + id + 1) % 5000 == 0:
            # # #      k += 1
            #      alpha = alpha / np.sqrt(2)
            # alpha = d / (step + 1)
            # alpha = d / np.sqrt(step * m + id + 5)
            # id = random.randint(0, m-1)
            grad = CalGrad(A[id], b[id], x_li[-1])
            # grad_li.append(grad)
            x = x - alpha * grad
            x_li.append(x)
            total_grad = CalTotalGrad(A, b, x_li[-1])
            grad_li.append(total_grad)
            diff_x_star.append(np.linalg.norm(x_li[-1] - x_star))
            grad_norm.append(np.linalg.norm(total_grad))
            print "step:", step * m + id, "grad_norm:", grad_norm[-1]

    s1 = 'alpha0.01'
    np.save('./result/SGD/' + s1 + '/grad.npy', grad_li)
    np.savetxt('./result/SGD/' + s1 + '/grad.txt', grad_li)
    np.savetxt('./result/SGD/' + s1 + '/grad_norm.txt', grad_norm)
    np.save('./result/SGD/' + s1 + '/x.npy', x_li)
    np.savetxt('./result/SGD/' + s1 + '/x.txt', x_li)
    np.savetxt('./result/SGD/' + s1 + '/diff_x_y.txt', diff_x_star)

    print x_li[-1]
    plt.semilogy(np.arange(num_iter * m), grad_norm)
    plt.xlabel('iter')
    plt.ylabel('log||grad||')
    plt.title(s1)
    plt.savefig('./result/SGD/' + s1 + '/grad_norm.jpg')
    plt.show()

    plt.semilogy(np.arange(num_iter * m), diff_x_star)
    plt.xlabel('iter')
    plt.ylabel('log||x - x*||')
    plt.title(s1)
    plt.savefig('./result/SGD/' + s1 + '/x_star.jpg')
    plt.show()

sgd()