import numpy as np
import matplotlib.pyplot as plt
import random

num_class = 10
num_feature = 28 * 28
num_train = 60000
num_test = 10000
num_machines = 20
batch_size = 32

num_iter = 5000
exit_byzantine = True
num_byz = 1


def cal_total_grad(X, Y, theta, weight_lambda):

    """
    :param X: shape(num_samples, features + 1)
    :param Y: labels' one_hot array, shape(num_samples, features + 1)
    :param theta: shape (num_classes, feature+1)
    :param weight_lambda: scalar
    :return: grad, shape(num_classes, feature+1)
    """
    m = X.shape[0]
    t = np.dot(theta, X.T)
    t = t - np.max(t, axis=0)
    pro = np.exp(t) / np.sum(np.exp(t), axis=0)
    total_grad = -np.dot((Y.T - pro), X) / m + weight_lambda * theta
    return total_grad


def cal_loss(X, Y, theta, weight_lambda):

    m = X.shape[0]
    t1 = np.dot(theta, X.T)
    t1 = t1 - np.max(t1, axis=0)
    t = np.exp(t1)
    tmp = t / np.sum(t, axis=0)
    loss = -np.sum(Y.T * np.log(tmp)) / m + weight_lambda * np.sum(theta ** 2) / 2
    return loss


def cal_acc(test_x, test_y, theta):

    pred = []
    num = 0
    m = test_x.shape[0]
    for i in range(m):
        pro = np.exp(np.dot(theta, test_x[i]))
        index = np.argmax(pro)
        if index == test_y[i]:
            num += 1
    acc = float(num) / m
    return acc, pred


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

    def __init__(self, data_x, data_y, machine_id):

        self.data_x = data_x
        self.data_y = data_y
        self.machine_id = machine_id

    def calc_gradient(self, theta, weight_lambda, id):

        per_samples = num_train / num_machines
        id = random.randint(0, per_samples - batch_size)
        grad = np.zeros_like(theta)
        if(exit_byzantine == True and self.machine_id == num_machines - 1):
            grad = np.ones_like(theta) * 100
        elif(exit_byzantine == True and self.machine_id == num_machines - 2):
            grad = np.ones_like(theta) * 100
        elif (exit_byzantine == True and self.machine_id == num_machines - 3):
            grad = np.ones_like(theta) * 100
        elif (exit_byzantine == True and self.machine_id == num_machines - 4):
            grad = np.ones_like(theta) * 100
        elif (exit_byzantine == True and self.machine_id == num_machines - 5):
            grad = np.ones_like(theta) * 100
        elif (exit_byzantine == True and self.machine_id == num_machines - 6):
            grad = np.ones_like(theta) * 100
        elif (exit_byzantine == True and self.machine_id == num_machines - 7):
            grad = np.ones_like(theta) * 100
        elif (exit_byzantine == True and self.machine_id == num_machines - 8):
            grad = np.ones_like(theta) * 100
        else:
            grad = cal_total_grad(self.data_x[id:(id + batch_size)], self.data_y[id:(id + batch_size)], theta, weight_lambda)
        return grad


class Parameter_server:

    def __init__(self):
        self.x_li = []
        self.x_star_norm = []
        self.total_grad = []
        self.index_li = []
        self.acc_li = []
        # self.grad_norm = []

        train_img = np.load('G:\python_code/byzantine/RSGD_multiLR/data/mnist/train_img.npy')  # shape(60000, 784)
        train_lbl = np.load('G:\python_code/byzantine/RSGD_multiLR/data/mnist/train_lbl.npy')  # shape(60000,)
        one_train_lbl = np.load('G:\python_code/byzantine/RSGD_multiLR/data/mnist/one_train_lbl.npy')  # shape(10, 60000)
        test_img = np.load('G:\python_code/byzantine/RSGD_multiLR/data/mnist/test_img.npy')  # shape(10000, 784)
        test_lbl = np.load('G:\python_code/byzantine/RSGD_multiLR/data/mnist/test_lbl.npy')  # shape(10000,)

        bias_train = np.ones(num_train)
        train_img_bias = np.column_stack((train_img, bias_train))

        bias_test = np.ones(num_test)
        test_img_bias = np.column_stack((test_img, bias_test))

        self.test_img_bias = test_img_bias
        self.test_lbl = test_lbl
        self.train_img_bias = train_img_bias
        self.one_train_lbl = one_train_lbl

        samples_per_machine = num_train / num_machines

        self.machines = []
        for i in range(num_machines):
            new_machine = Machine(train_img_bias[i * samples_per_machine:(i + 1) * samples_per_machine, :],
                                  one_train_lbl[i * samples_per_machine:(i + 1) * samples_per_machine], i)
            self.machines.append(new_machine)

    def broadcast(self, x, wei_lambda, id):

        grad_li = []
        for mac in self.machines:
            grad_li.append(mac.calc_gradient(x, wei_lambda, id))
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

    def train(self, init_x, alpha, wei_lambda):

        self.x_li.append(init_x)
        tmp = np.load('./result/mnist/gd/theta_li.npy')
        tmp = list(tmp)
        x_star = tmp[-1]

        sample_per_machine = num_train/ num_machines

        alpha = 0.0001
        d = 0.001
        wei_lambda = 0.01
        # self.x_star_norm.append(np.linalg.norm(init_x - x_star))
        for i in range(num_iter):
            # if (i + 1) % 30000 == 0:
            #     alpha = alpha / 10
            alpha = d / np.sqrt(i + 1)
            id = i % sample_per_machine
            grad_li = self.broadcast(self.x_li[-1], wei_lambda, id)
            # print len(grad_li)
            grad, i_star = krum(grad_li)
            self.index_li.append(int(i_star))
            new_x = self.x_li[-1] - alpha * grad
            total = cal_total_grad(self.train_img_bias, self.one_train_lbl, new_x, wei_lambda)
            self.total_grad.append(np.linalg.norm(total))
            if (i + 1) % 10 == 0:
                acc, _ = cal_acc(self.test_img_bias, self.test_lbl, new_x)
                self.acc_li.append(acc)
            # print"step:", i, "x_k:", np.linalg.norm(new_x)
            self.x_li.append(new_x)
            self.x_star_norm.append(np.linalg.norm(new_x - x_star))
            # print "step:", i, "x_star_norm:", self.x_star_norm[-1]
            print "step:", i, "total_grad_norm:", self.total_grad[-1]

    def plot(self):

        s1 = 'alpha0.001_sqrt_wei0.01'
        # np.save('./result/mnist/machine20/no_fault/' + s1 + '/x_li.npy', self.x_li)
        np.savetxt('./result/mnist/machine20/fault/q8/' + s1 + '/x_star_norm.txt', self.x_star_norm)
        np.savetxt('./result/mnist/machine20/fault/q8/' + s1 + '/index_li.txt', self.index_li)
        np.savetxt('./result/mnist/machine20/fault/q8/' + s1 + '/acc_li.txt', self.acc_li)

        plt.plot(np.arange(len(self.acc_li)) * 10, self.acc_li)
        plt.xlabel('iter')
        plt.ylabel('accuracy')
        # plt.title(s1)
        plt.savefig('./result/mnist/machine20/fault/q8/' + s1 + '/acc.jpg')
        plt.show()

        plt.semilogy(np.arange(num_iter), self.total_grad)
        # plt.plot(np.arange(num_iter), self.x_star_norm)
        plt.xlabel('iter')
        plt.ylabel('log||grad||')
        # plt.title(s1)
        plt.savefig('./result/mnist/machine20/fault/q8/' + s1 + '/grad_norm.jpg')
        plt.show()


def init():
    server = Parameter_server()
    return server


def main():
    server = init()
    init_x = np.zeros((num_class, num_feature + 1))
    alpha = 0.1
    wei_lam = 0.01
    server.train(init_x, alpha, wei_lam)
    server.plot()


main()












