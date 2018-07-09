import numpy as np
import matplotlib.pyplot as plt
import random

num_samples = 100
num_machines = 20
num_iter = 100000
dimension = 50
exit_byzantine = True
num_byz = 1

def CalGard(A, b, x):
    # grad = np.zeros_like(x)
    grad = (np.dot(A, x) - b) * A
    return grad

def CalTotalGrad(A, b, x):
    grad = np.zeros_like(x)
    for i in range(len(b)):
        grad = grad + (np.dot(A[i], x) - b[i]) * A[i]
    return grad

class Machine:
    def __init__(self, A, b, machine_id):
        """Initializes the machine with the data
        Accepts data, a numpy array of shape :(num_samples/num_machines, dimension)
        data_x : a numpy array has shape :num_samples/num_machines, dimension)
        data_y: a list of length 'num_samples/num_machine', the label of the data_x"""

        self.A = A
        self.b = b
        self.machine_id = machine_id

    def update(self, x, alpha):

        if (exit_byzantine == True and self.machine_id == num_machines - 1):
            tmp = [100 for _ in range(dimension)]
            new_x = np.array(tmp)
        else:
            m = len(self.b)
            id = random.randint(0, m - 1)
            grad = (np.dot(self.A[id], x) - self.b[id]) * self.A[id]
            new_x = x - alpha * grad
        return new_x

class Parameter_server:
        def __init__(self):
            """Initializes all machines"""
            self.x_li = []  # list that stores each theta, grows by one iteration
            self.grad_li = []
            self.grad_norm = []
            self.x_star_norm = []
            A = np.load('./data/A.npy')
            b = np.load('./data/b.npy')
            # x_star = np.load('./data/y.npy')

            self.A = A
            self.b = b

            sample_per_machine = num_samples / num_machines
            self.machines = []
            for i in range(num_machines):
                new_machine = Machine(A[i*sample_per_machine:(i + 1)*sample_per_machine], b[i * sample_per_machine:(i + 1) * sample_per_machine], i)
                self.machines.append(new_machine)

        def broadcast(self, x, alpha):
                """Broadcast theta
                Accepts theta, a numpy array of shape:(dimension,)
                Return a list of length 'num_machines' containing the updated theta of each machine"""

                new_x_li = []
                for i, mac in enumerate(self.machines):
                    new_x_li.append(mac.update(x, alpha))
                return new_x_li

        def calc_mean(self, x_li):
                """Calculates a list of mean of theta per batch, given the theta list
                Accepts theta_list, a list of length 'num_machines' containing theta
                Returns a list of length """
                sum_val = np.zeros_like(x_li[-1])
                for item in x_li:
                    sum_val = sum_val + item
                mean = sum_val / len(x_li)
                return mean

        def train(self, init_x, alpha):
                """Peforms num_iter rounds of update, appends each new x to x_li
                Accepts the initialed x, a numpy array has shape:(dimension,)"""

                # x_star = np.load('./data/y.npy')
                x_star = np.load('./result/GD/x_star.npy')
                self.x_li.append(init_x)
                lambda1 = 0.2
                d = 0.1
                for i in range(num_iter):
                    alpha = d / np.sqrt(i + 1)
                    # if (i + 1) % 10000 == 0:
                    #     alpha = alpha / np.sqrt(8)
                    rec_x = self.broadcast(self.x_li[-1], alpha)
                    mean_x = self.calc_mean(rec_x)
                    self.x_li.append(mean_x)
                    self.x_star_norm.append(np.linalg.norm(mean_x - x_star))
                    total_grad = CalTotalGrad(self.A, self.b, self.x_li[-1])
                    self.grad_li.append(total_grad)
                    self.grad_norm.append(np.linalg.norm(total_grad))
                    # print "step:", i, "grad_norm:", self.grad_norm[-1]
                    print "step:", i, "x_star_norm:", self.x_star_norm[-1]

                print("train end!")

        def plot_curve(self):
                """plot the loss curve and the acc curve
                save the learned theta to a numpy array and a txt file"""
                s1 = 'alpha0.1_sqrt'
                np.save('./result/DSGD_param/fault/20/' + s1 + '/x_li.npy', self.x_li)
                np.savetxt('./result/DSGD_param/fault/20/' + s1 + '/x_star_norm.txt', self.x_star_norm)
                np.savetxt('./result/DSGD_param/fault/20/' + s1 + '/grad.txt', self.grad_li)
                np.savetxt('./result/DSGD_param/fault/20/' + s1 + '/grad_norm.txt', self.grad_norm)

                fig = plt.figure(1)
                plt.semilogy(np.arange(num_iter), self.grad_norm)
                plt.xlabel('iter')
                plt.ylabel('log(||grad||)')
                plt.title('log(||grad||)')
                plt.savefig('./result/DSGD_param/fault/20/' + s1 + '/grad_norm.jpg')
                plt.show()

                plt.semilogy(np.arange(num_iter), self.x_star_norm)
                plt.xlabel('iter')
                plt.ylabel('log')
                plt.title('log||x0 - x*||')
                plt.savefig('./result/DSGD_param/fault/20/' + s1 + '/x_star_norm.jpg')
                plt.show()


def init():
    server = Parameter_server()
    return server

def main():
    server = init()
    init_x = np.zeros((dimension,))
    alpha = 0.01
    server.train(init_x, alpha)
    server.plot_curve()

main()