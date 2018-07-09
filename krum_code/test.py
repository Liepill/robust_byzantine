import numpy as np
from collections import Counter

#
# A = np.load('./data/A.npy')
# b = np.load('./data/b.npy')
#
# num_machines = 20
# grad_li = []
# x = np.zeros((50,))
#
# per_sample = 5
# for i in range(20):
#     tmp = (np.dot(A[i * 5], x) - b[i * 5]) * A[i * 5]
#     grad_li.append(tmp)
#
# dist = np.zeros((20, 20))
# dist_li = []
# score = []
# for i in range(20):
#     for j in range(20):
#         # dist[i, j] = np.linalg.norm(grad_li[i] - grad_li[j]) ** 2
#         if i != j:
#             dist_li.append(np.linalg.norm(grad_li[i] - grad_li[j]) ** 2)
#     dist_li.sort(reverse=False)
#     tmp = 0.0
#     for k in range(17):
#         tmp += dist_li[k]
#     score.append(tmp)
#     print dist_li
#     dist_li = []
#     break
# print score
# print len(score)
# print score.index(min(score))
# i_star = score.index(min(score))
# print np.linalg.norm(grad_li[i_star])

# a = [1, 1, 1, 2, 2, 3]
# print a
# a.sort(reverse=False)
# print a
# print a.count(1)
# val_count = Counter(a)
# print type(val_count)
# print val_count

file = open('./result/bgd/dist_data/machine20/no_fault/batch3_alpha0.1_sqrt/index_li.txt')
all_lines = file.readlines()
index_li = []
for line in all_lines:
    tmp = line.strip()
    index_li.append(float(tmp))

print len(index_li)
count_li = []
num_machine = 20
for i in range(num_machine):
    count_li.append(index_li.count(i))
print len(count_li)
print count_li

# correct = np.load('./result/bgd/dist_data/machine20/x_star.npy')
# all = np.load('./data/x_star.npy')
# print np.linalg.norm(correct - all)
# print correct
# print all


