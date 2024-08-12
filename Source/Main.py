import pandas as pd
import numpy as np
import math
import random
import sys

# Rectified Linear Unit (ReLU)
def ReLU(x1):
    return x1 * (x1 > 0)

# Multiplication of two matrices
def multiply(w1, inp):
    result = np.array([[sum(a * b for a, b in zip(A_row, B_col))
                        for B_col in zip(*inp)]
                       for A_row in w1])
    return result

# Softmax Activation Function
def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        dist = e / np.sum(e, axis=0)
    else:
        dist = e / np.array([np.sum(e, axis=1)]).T

    for i in range(5):
        if np.isnan(dist[0][i]):
            dist[0][i] = np.mean(dist)
    return dist

# Using the One-Hot method to encode rating value
def onehot(x):
    l = np.array([[0, 0, 0, 0, 0]])
    for i in range(0, 5):
        if i == x - 1:
            l[0][i] = 1
    return l

# Calculating cross entropy (difference between the prediction result y and the supervised value Y)
def error(y, Y):
    x = 0
    for i in range(5):
        if y[0][i] <= 0:
            y[0][i] = 0.00001
        elif y[0][i] >= 1:
            y[0][i] = 0.99999
        x += ((Y[0][i] * (math.log(y[0][i]))) + ((1 - Y[0][i]) * (math.log(1 - y[0][i]))))

    return -1 * x

# Reading ratings, movies, and users data
ratings_list = [i.strip().split("::") for i in open('ml/ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open('ml/users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('ml/movies.dat', 'r').readlines()]

# Convert ratings_list to DataFrame
ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype=int)
movies_df = pd.DataFrame(movies_list, columns=['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

# Forming user-item matrix with values of ratings
R_df = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
R = R_df.values
r = R

# Print user-item ratings matrix
print(r)

# n - number of users, m - number of items
n = R.shape[0]
m = R.shape[1]
print(n)
print(m)

# a - user's latent features, b - item's latent features
a = 20
b = 20
# Total no. of latent features
l = a + b

# Reproducibility
random.seed(10000)

# U - Users latent feature matrix, V - Items latent feature matrix
U = np.random.rand(n, a)
V = np.random.rand(m, b)

# W - Second-order coefficients matrix
W = np.random.rand(l, l)

# Delta matrix
D = np.random.rand(n, m)

# Initialize random user-ratings matrix
R1 = np.random.rand(n, m)

# Constant-term
z = random.random()

# Learning-rate
eta = 0.01

# w - first-order coefficients matrix
w = []
for i in range(l):
    w.append(random.random())

def summation(u, v):
    k = 0
    for i in range(0, a):
        for j in range(0, b):
            k += W[i][j] * U[u][i] * V[v][j]
    return k

def value_u(u, i):
    sumd = 0
    for v in range(0, m):
        sumw = 0
        for j in range(0, b):
            sumw += W[i][j] * V[v][j]
        sumd += D[u][v] * sumw
    return sumd

def value_v(v, j):
    sumd = 0
    for u in range(0, n):
        sumw = 0
        for i in range(0, a):
            sumw += W[i][j] * U[u][i]
        sumd += D[u][v] * sumw
    return sumd

def value_p(u):
    k = 0
    for i in range(a):
        k += w[i] * U[u][i]
    k1 = 0
    for i in range(a - 1):
        for j in range(i + 1, a):
            k1 += W[i][j] * U[u][i] * U[u][j]
    return k + k1

def value_q(v):
    k = 0
    for j in range(b):
        k += w[a + j] * V[v][j]
    k1 = 0
    for i in range(b - 1):
        for j in range(i + 1, b):
            k1 += W[i + a][j + a] * V[v][i] * V[v][j]
    return k + k1

L = sys.maxsize
p = []
q = []

# Calculate pu
for u in range(n):
    p.append(value_p(u))

# Calculate qu
for v in range(m):
    q.append(value_q(v))

# Calculate R1
for u in range(0, n):
    for v in range(0, m):
        R1[u][v] = z + p[u] + q[v] + summation(u, v)

while True:
    # Calculate Loss
    L1 = 0
    for u in range(0, n):
        for v in range(0, m):
            if r[u][v] != 0:
                L1 += ((r[u][v] - R1[u][v]) * (r[u][v] - R1[u][v]))
    L1 = L1 * 0.5
    if L1 >= L:
        print("Breaks")
        break
    print(L1)
    L = L1
    for u in range(0, n):
        for v in range(0, m):
            if r[u][v] != 0:
                D[u][v] = (R1[u][v] - r[u][v])
            else:
                D[u][v] = 0
    d = 0
    # Update z
    for i in range(n):
        for j in range(m):
            d += D[i][j]
    z = z - (eta * d)
    # Start loop for U
    for u in range(0, n):
        sumv = 0
        for x in range(0, m):
            sumv += D[u][x]
        p[u] = p[u] - (eta * sumv)
        for i in range(0, a):
            U[u][i] = U[u][i] - (eta * value_u(u, i))
    # Start loop for V
    for v in range(0, m):
        sumu = 0
        for x in range(0, n):
            sumu += D[x][v]
        q[v] = q[v] - (eta * sumu)
        for j in range(0, b):
            V[v][j] = V[v][j] - (eta * value_v(v, j))
    # Update Weight
    for i in range(0, a):
        for j in range(0, b):
            sum1 = 0
            for u in range(0, n):
                for v in range(0, m):
                    sum1 += (D[u][v]) * (U[u][i]) * (V[v][j])
            W[i][j] = W[i][j] - (eta * sum1)

print("User Matrix \n")
print(U)
print("\n")
print("Item Matrix \n")
print(V)
print("\n")

V = V.T

# New User Item Rating Matrix
Rc = np.random.rand(n, m)

# Learning rate
eta = 0.001
for u in range(R.shape[0]):
    for v in range(R.shape[1]):
        # Random weight matrix between input to first hidden layer
        w1 = np.random.rand(l, 27)
        # Random matrix between first hidden layer to second hidden layer
        w2 = np.random.rand(27, 12)
        # Random matrix between second hidden layer to output hidden layer
        w3 = np.random.rand(12, 5)
        # Bias matrix between input to first hidden layer
        b1 = np.random.rand(1, 27)
        # Bias matrix between first hidden layer to second hidden layer
        b2 = np.random.rand(1, 12)
        # Bias matrix between second hidden layer to output hidden layer
        b3 = np.random.rand(1, 5)
        c = 0
        for epoch in range(0, 100):
            X = np.concatenate([U[u], V[v]])
            X = X.T
            # Input to first hidden layer
            Z1 = np.dot(X, w1) + b1
            X1 = ReLU(Z1)
            # Input to second hidden layer
            Z2 = np.dot(X1, w2) + b2
            X2 = ReLU(Z2)
            # Input to output hidden layer
            Z3 = np.dot(X2, w3) + b3
            X3 = softmax(Z3)
            if r[u][v] != 0:
                L = error(X3, onehot(r[u][v]))
            else:
                L = 0
            if epoch == 99:
                Rc[u][v] = np.argmax(X3) + 1
            c += 1
            # Calculating error and updating weight and bias accordingly using backpropagation
            if r[u][v] != 0:
                dL_dz3 = X3 - onehot(r[u][v])
            else:
                dL_dz3 = 0
            dL_dw3 = np.dot(X2, dL_dz3)
            dL_db3 = dL_dz3
            dL_dx2 = np.dot(dL_dz3, w3.T)
            dL_dz2 = np.multiply(dL_dx2, np.where(Z2 > 0, 1, 0))
            dL_dw2 = np.dot(X1, dL_dz2)
            dL_db2 = dL_dz2
            dL_dx1 = np.dot(dL_dz2, w2.T)
            dL_dz1 = np.multiply(dL_dx1, np.where(Z1 > 0, 1, 0))
            dL_dw1 = np.dot(X.reshape(-1, 1), dL_dz1)
            dL_db1 = dL_dz1
            w3 -= eta * dL_dw3
            b3 -= eta * dL_db3
            w2 -= eta * dL_dw2
            b2 -= eta * dL_db2
            w1 -= eta * dL_dw1
            b1 -= eta * dL_db1

# New User Item Rating Matrix
print(Rc)
