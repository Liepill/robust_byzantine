import numpy as np
import matplotlib.pyplot as plt
import random

num_samples = 100
num_machines = 20
num_iter = 100000
dimension = 50
exit_byzantine = True
batch_size = 1
num_byz = 3

def CalTotalGrad(A, b, x):
    grad = np.zeros_like(x)
    for i in range(len(b)):
        grad = grad + (np.dot(A[i], x) - b[i]) * A[i]
    return grad


class Machine:
    def __init__(self, A, b, machine_id):
        self.A = A
        self.b = b
        self.machine_id = machine_id

    def update(self, x0, x, alpha, lambda1, id):

            if (exit_byzantine == True and self.machine_id == num_machines - 1):
                new_x = np.ones_like(x0) * 100
            elif(exit_byzantine == True and self.machine_id == num_machines - 2):
                new_x = np.ones_like(x0) * 100
            elif (exit_byzantine == True and self.machine_id == num_machines - 3):
                new_x = np.ones_like(x0) * 100
            # elif (exit_byzantine == True and self.machine_id == num_machines - 4):
            #     new_x = np.ones_like(x0) * 100
            # elif (exit_byzantine == True and self.machine_id == num_machines - 5):
            #     new_x = np.ones_like(x0) * 100
            else:
                m = len(self.b)
                grad_f = np.zeros_like(x0)
                id = random.randint(0, m - batch_size)
                for i in range(batch_size):
                    grad_f += (np.dot(self.A[id + i], x) - self.b[id + i]) * self.A[id + i]
                grad = (grad_f + lambda1 * np.sign(x - x0))
                new_x = x - alpha * grad
            return new_x


class Parameter_server:
    def __init__(self):

        self.x0_li = []
        self.x_li = []
        self.grad_li = []
        self.grad_norm = []
        self.x_star_norm = []

        A = np.load('./data/A.npy')
        b = np.load('./data/b.npy')
        self.A = A
        self.b = b

        sample_per_machine = num_samples / num_machines
        self.machines = []
        for i in range(num_machines):
            new_machine = Machine(A[i * sample_per_machine:(i + 1) * sample_per_machine],
                                  b[i * sample_per_machine:(i + 1) * sample_per_machine], i)
            self.machines.append(new_machine)

    def broadcast(self, x0, x_li, alpha, lambda1, id):

        new_x_li = []
        for i, mac in enumerate(self.machines):
            new_x_li.append(mac.update(x0, x_li[i], alpha, lambda1, id))
        tmp = np.zeros_like(x0)
        for i in range(len(x_li)):
            tmp += np.sign(x0 - x_li[i])
        new_x0 = x0 - alpha * lambda1 * tmp
        return new_x0, new_x_li

    def train(self, init_x0, init_x, alpha, lambda1):

        # x_star = np.load('./data/y.npy')
        x_star = np.load('./result/GD/correct/machine20/q3/alpha0.004/x_star.npy')
        self.x0_li.append(init_x0)
        self.x_li.append(init_x)
        lambda1 = 0.1
        alpha = 0.001
        d = 0.1
        k = 0
        sample_per_machine = num_samples / num_machines
        for i in range(num_iter):
            # if (i + 1) % 1000 == 0:
            #     k += 1
            #     alpha = alpha / 10
            alpha = d / np.sqrt(i + 1)
            # alpha = d / (i + 1)
            id = i % sample_per_machine
            rec_x0, rec_x = self.broadcast(self.x0_li[-1], self.x_li[-1], alpha, lambda1, id)
            self.x0_li.append(rec_x0)
            self.x_li.append(rec_x)
            self.x_star_norm.append(np.linalg.norm(rec_x0 - x_star))
            total_grad = CalTotalGrad(self.A[:85], self.b[:85], self.x0_li[-1])
            # total_grad = CalTotalGrad(self.A, self.b, self.x0_li[-1])
            self.grad_li.append(total_grad)
            self.grad_norm.append(np.linalg.norm(total_grad))
            # print "step:", i, "grad_norm:", self.grad_norm[-1]
            # print "step:", i, "x0:", self.x0_li[-1]
            print "step:", i, "x_star_norm:", self.x_star_norm[-1]
        print "train end!"

    def plot_curve(self):

        s1 = 'lam0.1_alpha0.1_sqrt'

        np.save('./result/RDSGD/fault/20_q3/' + s1 + '/x_li.npy', self.x_li)
        # np.savetxt('./result/RDSGD/no_fault/x_li.txt', self.x_li)
        np.save('./result/RDSGD/fault/20_q3/' + s1 + '/x0_li.npy', self.x0_li)
        np.savetxt('./result/RDSGD/fault/20_q3/' + s1 + '/x0_li.txt', self.x0_li)
        np.save('./result/RDSGD/fault/20_q3/' + s1 + '/x_star_norm.npy', np.array(self.x_star_norm))
        np.savetxt('./result/RDSGD/fault/20_q3/' + s1 + '/x_star_norm.txt', self.x_star_norm)
        np.savetxt('./result/RDSGD/fault/20_q3/' + s1 + '/grad.txt', self.grad_li)
        np.savetxt('./result/RDSGD/fault/20_q3/' + s1 + '/grad_norm.txt', self.grad_norm)

        fig = plt.figure(1)
        plt.semilogy(np.arange(num_iter), self.grad_norm)
        plt.xlabel('iter')
        plt.ylabel('log(||grad||)')
        plt.title(s1)
        plt.savefig('./result/RDSGD/fault/20_q3/' + s1 + '/grad_norm.jpg')
        plt.show()

        plt.semilogy(np.arange(num_iter), self.x_star_norm)
        plt.xlabel('iter')
        plt.ylabel('log||x0 - x*||')
        plt.title(s1)
        plt.savefig('./result/RDSGD/fault/20_q3/' + s1 + '/x_star_norm.jpg')
        plt.show()


def init():
    server = Parameter_server()
    return server

def main():
    server = init()
    init_x0 = np.zeros((dimension, ))
    init_x = []
    for i in range(num_machines):
        init_x.append(np.zeros((dimension, )))
    alpha = 0.01
    lambda1 = 0.2
    server.train(init_x0, init_x, alpha, lambda1)
    server.plot_curve()

main()