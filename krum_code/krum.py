import numpy as np
import matplotlib.pyplot as plt
import random

num_samples = 100
num_batches = 5
num_machines = 20
num_iter = 100000
dimension = 50
num_byz = 1
batch_size = 3
exist_byzantine = False


def CalTotalGrad(A, b, x):

    grad = np.zeros_like(x)
    for i in range(len(b)):
        grad = grad + (np.dot(A[i], x) - b[i]) * A[i]
    return grad


def krum(grad_li):

        score = []
        num_near = num_machines - num_byz - 2
        for i, g_i in enumerate(grad_li):
            dist_li = []
            for j, g_j in enumerate(grad_li):
                if i != j:
                    dist_li.append(np.linalg.norm(g_i - g_j) ** 2)
            dist_li.sort(reverse=False)
            score.append(sum(dist_li[0:num_near]))
        i_star = score.index(min(score))
        # i_star = random.randint(0, 18)
        # i_star = 2
        # return i_star
        return grad_li[i_star], i_star


class Machine:

    def __init__(self, A, b, machine_id):

        self.A = A
        self.b = b
        self.machine_id = machine_id

    def calc_gradient(self, x, id):

        # id = random.randint(0, 4)
        per_samples = num_samples / num_machines
        id = random.randint(0, per_samples - batch_size)
        # print "machine id:", self.machine_id, "sample id:", id
        grad = np.zeros_like(x)
        if(exist_byzantine == True and self.machine_id == num_machines - 1):
            tmp = [100 for _ in range(dimension)]
            grad = np.array(tmp)
        else:
            for i in range(batch_size):
                grad += (np.dot(self.A[id + i], x) - self.b[id + i]) * self.A[id + i]
            # grad = (np.dot(self.A[id], x) - self.b[id]) * self.A[id]
            # grad = grad / batch_size
        return grad


class Parameter_server:

    def __init__(self):
        A = np.load('./data/A.npy')
        b = np.load('./data/b.npy')
        self.A = A
        self.b = b
        self.x_li = []
        self.x_star_norm = []
        self.total_grad = []
        self.index_li = []
        # self.grad_norm = []

        sample_per_machine = num_samples / num_machines
        self.machines = []
        for i in range(num_machines):
            new_machine = Machine(A[i * sample_per_machine:(i + 1) * sample_per_machine],
                                  b[i * sample_per_machine:(i + 1) * sample_per_machine], i)
            # new_machine = Machine(A, b, i)
            self.machines.append(new_machine)

    def broadcast(self, x, id):

        grad_li = []
        for mac in self.machines:
            grad_li.append(mac.calc_gradient(x, id))
        return grad_li

    # def krum(self, grad_li):
    #     dist_li = []
    #     score = []
    #     num_near = num_machines - num_byz - 2
    #     for i in range(num_machines):
    #         for j in range(num_machines):
    #             if i != j:
    #                 dist_li.append(np.linalg.norm(grad_li[i] - grad_li[j]) ** 2)
    #         dist_li.sort(reverse=False)
    #         tmp = 0.0
    #         for k in range(num_near):
    #             tmp += dist_li[k]
    #         score.append(tmp)
    #         dist_li = []
    #     i_star = score.index(min(score))
    #     # i_star = 2
    #     return i_star

    def train(self, init_x, alpha):

        self.x_li.append(init_x)
        x_star = np.load('./data/correct/alpha0.002/x_star.npy')

        sample_per_machine = num_samples / num_machines

        alpha = 0.1
        d = 0.1
        # self.x_star_norm.append(np.linalg.norm(init_x - x_star))
        for i in range(num_iter):
            # if (i + 1) % 30000 == 0:
            #     alpha = alpha / 10
            alpha = d / np.sqrt(i + 1)
            # id = i % sample_per_machine
            id = i % num_samples
            # id = random.randint(0, 4)
            grad_li = self.broadcast(self.x_li[-1], id)
            # print len(grad_li)
            grad, i_star = krum(grad_li)
            self.index_li.append(int(i_star))
            # print "step:", i, "grad_norm:", np.linalg.norm(grad_li[i_star])
            # new_x = self.x_li[-1] - alpha * grad_li[i_star]
            new_x = self.x_li[-1] - alpha * grad
            total = CalTotalGrad(self.A[:95], self.b[:95], new_x)
            self.total_grad.append(np.linalg.norm(total))
            # print"step:", i, "x_k:", np.linalg.norm(new_x)
            self.x_li.append(new_x)
            self.x_star_norm.append(np.linalg.norm(new_x - x_star))
            print "step:", i, "x_star_norm:", self.x_star_norm[-1]
            # print "step:", i, "total_grad_norm:", self.total_grad[-1]

    def plot(self):

        s1 = 'batch3_alpha0.1_sqrt'
        np.save('./result/bgd/dist_data/machine20/no_fault/' + s1 + '/x_li.npy', self.x_li)
        np.savetxt('./result/bgd/dist_data/machine20/no_fault/' + s1 + '/x_star_norm.txt', self.x_star_norm)
        np.savetxt('./result/bgd/dist_data/machine20/no_fault/' + s1 + '/index_li.txt', self.index_li)

        plt.semilogy(np.arange(num_iter), self.x_star_norm)
        # plt.plot(np.arange(num_iter), self.x_star_norm)
        plt.xlabel('iter')
        plt.ylabel('log||x - x*||')
        plt.title(s1)
        plt.savefig('./result/bgd/dist_data/machine20/no_fault/' + s1 + '/x_star_norm.jpg')
        plt.show()


def init():
    server = Parameter_server()
    return server


def main():
    server = init()
    init_x = np.zeros((dimension,))
    alpha = 0.1
    server.train(init_x, alpha)
    server.plot()


main()












