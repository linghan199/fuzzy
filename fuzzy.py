import numpy as np


# util
def get_min(a, b):
    c = np.zeros(len(a))
    index = a < b
    c[index] = a[index]
    c[~index] = b[~index]
    return c


data = np.array([[0.1, 0.0, 0.2, 0.8, 0.3, 0.0, 0.5, 0.6, 0.0, 0.1, 0.3, 0.1, 0.2, 0.2, 0.1, 0.2],
                 [0.7, 0.5, 0.2, 0.1, 0.0, 0.4, 0.0, 0.3, 0.5, 0.6, 0.2, 0.5, 0.0, 0.6, 0.7, 0.4],
                 [0.2, 0.5, 0.2, 0.0, 0.4, 0.0, 0.4, 0.0, 0.1, 0.0, 0.1, 0.4, 0.2, 0.1, 0.1, 0.2],
                 [0.0, 0.0, 0.4, 0.1, 0.3, 0.6, 0.1, 0.1, 0.4, 0.3, 0.4, 0.0, 0.6, 0.1, 0.1, 0.2]])

# a. find the fuzzy tolerance relation R1
R1 = np.zeros((16, 16))

for i in range(16):
    for j in range(16):
        numerator = np.dot(data[:, i].T, data[:, j])
        dominator = np.sqrt(np.sum(data[:, i] ** 2) * np.sum(data[:, j] ** 2))
        R1[i, j] = numerator / dominator

print(R1)

# b. transform R1 into a fuzzy equivalence relation R

R = np.zeros((16, 16))

R_current = R1.copy()

while 1:
    for i in range(16):
        for j in range(16):
            # find the minimum of associated relations
            v_min = get_min(R1[:, i], R1[j, :].T)
            # and get the maximum among these relations
            R_current[i, j] = np.max(v_min)

    if (R_current == R1).all():
        break
    else:
        R1 = R_current.copy()

R = R_current
print(R)

# c. Generate the alpha-cut R_a for alpha = 0.4 and alpha = 0.8

R_a_1 = np.zeros((16, 16))
R_a_2 = np.zeros((16, 16))

R_a_1[R >= 0.4] = 1
R_a_2[R >= 0.8] = 1

print(R_a_1)
print(R_a_2)

# d. Classify X into 3 classes
alpha = 0.8
step = 0.001
r = 0
R_a = np.zeros((16, 16))

# lower bound first, just start with alpha = 0.8
while r != 3:
    R_a = np.zeros((16, 16))
    R_a[R >= alpha] = 1
    # the number of classes are equal to the number of different col. vectors
    # here, different cols are linearly independent, so just calculate the rank of R_a
    r = np.linalg.matrix_rank(R_a)
    alpha += step

alpha -= step
print("lower bound: ", alpha)
print(R_a)

# upper bound, start with alpha = 1
alpha = 1
r = 0

while r != 3:
    R_a = np.zeros((16, 16))
    R_a[R >= alpha] = 1
    r = np.linalg.matrix_rank(R_a)
    alpha -= step

alpha += step
print("upper bound: ", alpha)
print(R_a)



print("#################test#####################")
print(R)
alpha = 1
R_a = np.zeros((16, 16))
R_a[R >= alpha] = 1
r = np.linalg.matrix_rank(R_a)
print(R_a)
print(r)