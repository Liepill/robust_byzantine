import numpy as np
import matplotlib.pyplot as plt

A1 = np.load('./data/A.npy')
b1 = np.load('./data/b.npy')
x_star = np.load('./data/y.npy')

A = A1[0:85]
b = b1[0:85]

print A.shape
print b.shape
# print A[90]


def CalGrad(A, b, x):

    grad = np.zeros_like(x)
    for i in range(len(b)):
        grad = grad + (np.dot(A[i], x) - b[i]) * A[i]
    return grad


def GD():
    num_iter = 10000
    alpha = 0.004
    x_li = []
    grad_li = []
    grad_norm = []
    diff_x_star = []

    x = np.zeros((A.shape[1],))
    x_li.append(x)

    for step in range(num_iter):
        grad = CalGrad(A, b, x_li[-1])
        x = x - alpha * grad
        x_li.append(x)
        diff_x_star.append(np.linalg.norm(x - x_star))
        grad_li.append(grad)
        grad_norm.append(np.linalg.norm(grad))
        print "step:", step, "grad_norm:", grad_norm[-1]
        # print "step:", step, "x_star_norm:", diff_x_star[-1]

    s1 = 'alpha0.004'
    np.save('./result/GD/correct/machine20/q3/' + s1 + '/grad.npy', grad_li)
    np.savetxt('./result/GD/correct/machine20/q3/' + s1 + '/grad.txt', grad_li)
    np.savetxt('./result/GD/correct/machine20/q3/' + s1 + '/grad_norm.txt', grad_norm)
    np.save('./result/GD/correct/machine20/q3/' + s1 + '/x.npy', x_li)
    np.savetxt('./result/GD/correct/machine20/q3/' + s1 + '/x.txt', x_li)
    # np.savetxt('./result/GD/correct/machine20/q5/' + s1 + '/diff_x_y.txt', diff_x_star)
    np.save('./result/GD/correct/machine20/q3/' + s1 + '/x_star.npy', x_li[-1])
    np.savetxt('./result/GD/correct/machine20/q3/' + s1 + '/x_star.txt', x_li[-1])

    print x_li[-1]
    plt.semilogy(np.arange(num_iter), grad_norm)
    plt.xlabel('iter')
    plt.ylabel('log||grad||')
    plt.title(s1)
    plt.savefig('./result/GD/correct/machine20/q3/' + s1 + '/grad_norm.jpg')
    plt.show()

    plt.semilogy(np.arange(num_iter), diff_x_star)
    plt.xlabel('iter')
    plt.ylabel('log||x - x*||')
    plt.title(s1)
    plt.savefig('./result/GD/correct/machine20/q3/' + s1 + '/diff_x_star.jpg')
    plt.show()


GD()
