# from sklearn.metrics import pairwise_distances
import numpy as np


def minimum_vector(vec1, vec2):
    if not len(vec1) == len(vec2):
        return 0
    vec = np.zeros(len(vec1))
    for ii in range(len(vec1)):
        if vec1[ii] <= vec2[ii]:
            vec[ii] = vec1[ii]
        else:
            vec[ii] = vec2[ii]
    return vec


dataset = np.array([[0.1, 0.0, 0.2, 0.8, 0.3, 0.0, 0.5, 0.6, 0.0, 0.1, 0.3, 0.1, 0.2, 0.2, 0.1, 0.2],
                    [0.7, 0.5, 0.2, 0.1, 0.0, 0.4, 0.0, 0.3, 0.5, 0.6, 0.2, 0.5, 0.0, 0.6, 0.7, 0.4],
                    [0.2, 0.5, 0.2, 0.0, 0.4, 0.0, 0.4, 0.0, 0.1, 0.0, 0.1, 0.4, 0.2, 0.1, 0.1, 0.2],
                    [0.0, 0.0, 0.4, 0.1, 0.3, 0.6, 0.1, 0.1, 0.4, 0.3, 0.4, 0.0, 0.6, 0.1, 0.1, 0.2]])
dataset = dataset.T
# calculate cosine similarity to generate fuzzy tolerance relation
R1 = np.zeros((16, 16))
for i in range(16):
    for j in range(16):
        R1[i, j] = np.dot(dataset[i, :], dataset[j, :].T) / \
                   np.sqrt(np.sum(dataset[i, :] ** 2) * np.sum(dataset[j, :] ** 2))
# R1 = pairwise_distances(dataset.T, metric='cosine')

# transform R1 into a fuzzy equivalence relation R
R_temp = np.zeros((16, 16))

R = np.zeros((16, 16))
while 1:
    for i in range(16):
        for j in range(16):
            R_temp[i, j] = np.max(minimum_vector(R1[i, :], R1[:, j].T))
    if (R_temp == R1).all():
        break
    else:
        # R1 = R_temp
        R1 = R_temp.copy()
R = R_temp
print(R)
print((R == R1).all())
# alpha-cut
R_04 = np.zeros((16, 16))
R_08 = np.zeros((16, 16))
R_04[R >= 0.4] = 1
R_08[R >= 0.8] = 1

# iterate alpha-cut
alpha = 1
temp = []
while len(temp) != 3:
    R_alpha = np.zeros((16, 16))
    R_alpha[R >= alpha] = 1
    temp = [R_alpha[0, :]]
    for i in range(16):
        flag = False
        for j in range(len(temp)):
            if (temp[j] == R_alpha[i, :]).all():
                flag = True
                break
        if not flag:
            temp.append(R_alpha[i, :])
    alpha -= 0.01


### test
R_alpha = np.zeros((16, 16))
R_alpha[R >= alpha] = 1
print(alpha)
print(R_alpha)