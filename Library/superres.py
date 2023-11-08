import numpy as np
import scipy.misc
from scipy.spatial.distance import cdist

import pickle, os
from os.path import dirname
import matplotlib.pyplot as plt
from utils import logdet

#Read data
path = os.path.join(os.getcwd(), 'data\\')
f = open(path + 'lena.dat','rb')
X = np.array(pickle.load(f))
X = X[250:350,250:350].astype(np.float32)
f.close()

# Normalize
X = -0.5 + (X - X.min()) / (X.max() - X.min())

plt.imshow(X)
plt.show()

row = X.shape[0]
col = X.shape[1]
N = row * col

# Generate low-res images
K = 8
D = 2
M = N / (D**2)
Gam = 2
r = 1
A = 0.04
beta = 1 / 0.05

indh = np.array([[i // row, i % col] for i in np.arange(row * col)]).astype(np.float32)
indl = np.array([[i // row, i % col] for i in np.arange(row * col) if ((i // row) % D == 0) and ((i % col) % D == 0)])
v_h = np.array([row/2, col/2])

W = []
Xl = []
for k in range(K):
    s = np.random.uniform(-2, 2, size = 2)
    Th = np.random.uniform(-4, 4) * np.pi / 180
    R = np.array([[np.cos(Th), np.sin(Th)], [-np.sin(Th), np.cos(Th)]])
    u = (indl - v_h) @ R.T + v_h + s
    u = u.astype(np.float32)
    
    Wk = 2*u @ indh.T - np.sum(indh**2, axis = 1) - np.sum(u**2, axis = 1)[None].T
    Wk = np.exp(Wk / (Gam**2))
    W.append(Wk / Wk.sum(axis = 1)[None].T)
    
    eps = np.random.normal(scale = 1/beta, size = (W[-1].shape[0]))
    
    Xk = W[-1] @ X.flatten() + eps
    Xl.append(Xk.reshape((row//D , col//D)))

plt.imshow(Xl[0])
plt.show()
    

# Prior
Z = 2 * indh @ indh.T - np.sum(indh**2, axis = 1)[None] - np.sum(indh**2, axis = 1)[None].T
Z = A * np.exp(Z / (r**2))
Zi = np.linalg.inv(Z)


# Posterior
S = np.linalg.inv(Z)
for k in range(K):
    S += beta * W[k].T @ W[k]
S = np.linalg.inv(S)
m = beta * S @ sum([W[k].T @ Xl[k].flatten() for k in range(K)])

# Log-Likelihood
ll = sum([0.5*M*(np.log(beta) - np.log(2* np.pi)) - 0.5 *  beta * np.sum((Xl[k].flatten() - W[k] @ X.flatten())**2) for k in range(K)])

# Log Marginal
lm = - 0.5 *  (beta * np.sum((Xl[0].flatten() - W[0] @ m)**2) - beta * np.trace(W[0] @ S @ W[0].T) 
          + m @ Zi @ m + logdet(Z) - logdet(S))



Xp = m.reshape((row, col))
plt.imshow(Xp)

