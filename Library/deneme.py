

import numpy as np


D = 20
K = 3
W = np.random.multivariate_normal(np.zeros(D), np.eye(D), size = K)
A = np.random.multivariate_normal(np.zeros(K), np.eye(K), size = K)
Sigma = np.diag(np.random.multivariate_normal(np.zeros(D), np.eye(D))**2)
#Sigma = np.random.multivariate_normal(np.zeros(D), np.eye(D), size = D)

ans =np.trace(Sigma @ W.T @ A @ W)
ans2 = sum([Sigma[j][j] * W[:,j].T @ A @ W[:,j] for j in range(D)])


ans = W @ Sigma @ W.T
ans2 = sum([Sigma[j][j] * W[:,j][None].T @ W[:,j][None] for j in range(D)])