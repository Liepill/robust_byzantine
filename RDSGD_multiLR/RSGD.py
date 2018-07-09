import numpy as np
import random
import matplotlib.pyplot as plt

num_class = 10
num_feature = 28 * 28
num_train = 60000
num_test = 10000
num_machines = 20
batch_size = 32

num_iter = 20000
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
    # loss = 0.0
    t = np.dot(theta, X.T)
    t = t - np.max(t, axis=0)
    pro = np.exp(t) / np.sum(np.exp(t), axis=0)
    total_grad = -np.dot((Y.T - pro), X) / m #+ weight_lambda * theta
    # loss = -np.sum(Y.T * np.log(pro)) / m + weight_lambda / 2 * np.sum(theta ** 2)
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


class Machine:
    def __init__(self, data_x, data_y, machine_id):
        """Initializes the machine with the data
        Accepts data, a numpy array of shape :(num_samples/num_machines, dimension)
        data_x : a numpy array has shape :num_samples/num_machines, dimension)
        data_y: a list of length 'num_samples/num_machine', the label of the data_x"""

        self.data_x = data_x
        self.data_y = data_y
        self.machine_id = machine_id

    def update(self, theta0, theta, alpha, l1_lambda, weight_lambda):
        """Calculates gradient with a randomly selected sample, given the current theta
         Accepts theta, a np array with shape of (dimension,)
         Returns the calculated gradient"""
        if (exit_byzantine == True and self.machine_id == num_machines - 1):
                new_theta = np.ones((num_class, num_feature + 1)) * 100
        elif(exit_byzantine == True and self.machine_id == num_machines - 2):
            new_theta = np.ones((num_class, num_feature + 1)) * 100
        elif (exit_byzantine == True and self.machine_id == num_machines - 3):
            new_theta = np.ones((num_class, num_feature + 1)) * 100
        elif (exit_byzantine == True and self.machine_id == num_machines - 4):
            new_theta = np.ones((num_class, num_feature + 1)) * 100
        elif (exit_byzantine == True and self.machine_id == num_machines - 5):
            new_theta = np.ones((num_class, num_feature + 1)) * 100
        elif (exit_byzantine == True and self.machine_id == num_machines - 6):
            new_theta = np.ones((num_class, num_feature + 1)) * 100
        elif (exit_byzantine == True and self.machine_id == num_machines - 7):
            new_theta = np.ones((num_class, num_feature + 1)) * 100
        elif (exit_byzantine == True and self.machine_id == num_machines - 8):
            new_theta = np.ones((num_class, num_feature + 1)) * 100
        # elif (exit_byzantine == True and self.machine_id == num_machines - 9):
        #     new_theta = np.ones((num_class, num_feature + 1)) * 100
        # elif (exit_byzantine == True and self.machine_id == num_machines - 10):
        #     new_theta = np.ones((num_class, num_feature + 1)) * 100
        # elif (exit_byzantine == True and self.machine_id == num_machines - 11):
        #     new_theta = np.ones((num_class, num_feature + 1)) * 100
        # elif (exit_byzantine == True and self.machine_id == num_machines - 12):
        #     new_theta = np.ones((num_class, num_feature + 1)) * 100
        else:
            m = self.data_x.shape[0]
            id = random.randint(0, m - batch_size)
            grad_f = cal_total_grad(self.data_x[id:(id + batch_size)], self.data_y[id:(id + batch_size)], theta, weight_lambda)
            grad = grad_f / num_machines + l1_lambda * np.sign(theta - theta0)
            new_theta = theta - alpha * grad
        return new_theta


class Parameter_server:
    def __init__(self):
        """Initializes all machines"""
        self.theta0_li = []
        self.theta_li = [] #list that stores each theta, grows by one iteration
        self.acc_li = []
        self.grad_li = []
        self.grad_norm = []
        self.theta0_star_norm = []
        self.acc_li = []
        self.loss_li = []
        self.theta_li_diff = []
        self.theta0_li_diff = []

        train_img = np.load('./data/mnist/train_img.npy')  # shape(60000, 784)
        train_lbl = np.load('./data/mnist/train_lbl.npy')  # shape(60000,)
        one_train_lbl = np.load('./data/mnist/one_train_lbl.npy')  # shape(10, 60000)
        test_img = np.load('./data/mnist/test_img.npy')  # shape(10000, 784)
        test_lbl = np.load('./data/mnist/test_lbl.npy')  # shape(10000,)

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

    def broadcast(self, theta0, theta_li, alpha, l1_lambda, weight_lambda):
        """Broadcast theta
        Accepts theta, a numpy array of shape:(dimension,)
        Return a list of length 'num_machines' containing the updated theta of each machine"""

        new_theta_li = []
        for i, mac in enumerate(self.machines):
            new_theta_li.append(mac.update(theta0, theta_li[i], alpha, l1_lambda, weight_lambda))
        tmp = np.zeros_like(theta0)
        for i in range(len(theta_li)):
            tmp += np.sign(theta0 - new_theta_li[i])
        new_theta0 = theta0 - alpha * (l1_lambda * tmp + weight_lambda * theta0)
        return new_theta0, new_theta_li

    def train(self, init_theta0, init_theta, alpha, l1_lambda, weight_lambda):
        """Peforms num_iter rounds of update, appends each new theta to theta_li
        Accepts the initialed theta, a numpy array has shape:(dimension,)"""

        tmp = np.load('./result/GD/wei0.01_alpha0.00001/theta_li.npy')
        tmp = list(tmp)
        theta_star = tmp[-1]
        self.theta0_li.append(init_theta0)
        self.theta_li.append(init_theta)
        k = 0
        d = 0.0001
        for i in range(num_iter):
            # if (i + 1) == 1000:
            #     alpha = alpha / 10
            alpha = d / np.sqrt(i + 1)
            rec_theta0, rec_theta = self.broadcast(self.theta0_li[-1], self.theta_li[-1], alpha, l1_lambda, weight_lambda)
            self.theta0_li.append(rec_theta0)
            self.theta_li.append(rec_theta)
            tmp = []
            for j in range(20):
                tmp.append(np.linalg.norm(self.theta_li[-1][j] - self.theta_li[-2][j]))
            self.theta_li_diff.append(tmp)
            tmp0 = self.theta0_li[-1] - self.theta0_li[-2]
            self.theta0_li_diff.append(np.linalg.norm(tmp0))
            # self.theta0_star_norm.append(np.linalg.norm(rec_theta0 - theta_star))
            # total_grad = CalTotalGrad(self.train_x[:1280], self.train_y[:1280], self.theta0_li[-1])
            # self.grad_li.append(total_grad)
            # self.grad_norm.append(np.linalg.norm(total_grad))
            # print "step: ", i, "||theta0 - thata*||:", self.theta0_star_norm[-1]
            # loss = cal_loss(self.train_img_bias, self.one_train_lbl, rec_theta0, weight_lambda)
            total_grad = cal_total_grad(self.train_img_bias, self.one_train_lbl, rec_theta0, weight_lambda) + weight_lambda * rec_theta0
            self.grad_norm.append(np.linalg.norm(total_grad))
            # self.loss_li.append(loss)
            if (i + 1) % 10 == 0:
                acc, _ = cal_acc(self.test_img_bias, self.test_lbl, rec_theta0)
                self.acc_li.append(acc)
                print "step:", i, " acc:", acc
            # print "step: ", i, " loss: ", self.loss_li[-1]
            print "step: ", i, " grad_norm: ", self.grad_norm[-1]
        print("train end!")

    def plot_curve(self):
        """plot the loss curve and the acc curve
        save the learned theta to a numpy array and a txt file"""

        s1 = 'lam0.5_wei0.01_alpha0.0001_sqrt_3'
        np.save('./result/RSGD/fault/q8/' + s1 + '/acc.npy', self.acc_li)
        # np.save('./result/RSGD/fault/q8/' + s1 + '/theta_li.npy', self.theta_li)
        # np.save('./result/RSGD/fault/q8/' + s1 + '/theta0_li.npy', self.theta0_li)
        np.save('./result/RSGD/fault/q8/' + s1 + '/theta0_li_diff.npy', self.theta0_li_diff)
        np.save('./result/RSGD/fault/q8/' + s1 + '/theta_li_diff.npy', self.theta_li_diff)

        plt.plot(np.arange(len(self.acc_li)) * 10, self.acc_li)
        plt.xlabel('iter')
        plt.ylabel('accuracy')
        # plt.title(s1)
        plt.savefig('./result/RSGD/fault/q8/' + s1 + '/acc.jpg')
        plt.show()

        # plt.plot(np.arange(num_iter), self.loss_li)
        # plt.xlabel('iter')
        # plt.ylabel('loss')
        # # plt.title(s1)
        # plt.savefig('./result/RSGD/fault/q8/' + s1 + '/loss.jpg')
        # plt.show()

        plt.semilogy(np.arange(num_iter), self.grad_norm)
        plt.xlabel('iter')
        plt.ylabel('log||grad||')
        # plt.title(s1)
        plt.savefig('./result/RSGD/fault/q8/' + s1 + '/grad_norm.jpg')
        plt.show()


def init():
    server = Parameter_server()
    return server


def main():
    server = init()
    init_theta0 = np.zeros((num_class, num_feature + 1))
    init_theta = []
    for i in range(num_machines):
        init_theta.append(np.zeros((num_class, num_feature + 1)))
    alpha = 0.0001
    l1_lambda = 0.5
    weight_lambda = 0.01
    server.train(init_theta0, init_theta, alpha, l1_lambda, weight_lambda)
    server.plot_curve()


main()
