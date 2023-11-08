import numpy as np
from scipy.special import softmax
from sklearn.datasets import make_sparse_spd_matrix, make_spd_matrix

from utils import norm_cov
'''
To-do:
    
    

'''


class data_gen():
    
    def __init__(self, K, P,  M, C_max = 0, mr = 0., D = 0, static = True):
        
        self.K = K # Latent space dimension
        self.P = P # of multinomial modalities + # of gaussian modalities
        self.M = M # Observation dimensions
        self.C_max = C_max # Total number of counts - 1 (if 0 -> Categorical)
        
        self.D = D # Dimension of covariates
        self.mr = mr # Missing rate
        self.static = static
        
        # Hyperparams
        self.tau = 0.01 # Expected noise variance for Gaussian modalities
        self.lmd = 0.1 # State transition scale
        self.tr_ratio = 0.8 # Train ratio during split
        self.alpha = 0.95
        
        self.mu0 = np.random.normal(size = self.K)
        self.S0 = np.eye(self.K)
        self.Q = np.eye(self.K)
        
        # Generate params
        self.W = []
        self.b = []
        for p in range(sum(self.P)):
            self.W.append(np.random.multivariate_normal(np.zeros(self.K), np.eye(self.K), size = int(self.M[p])))
            self.b.append(np.random.multivariate_normal(np.zeros(int(self.M[p])), np.eye(int(self.M[p]))))

        
        
    def generate_data(self, N, I = 1):
        
        # Generate latent variables
        V = np.random.normal(size = self.D * self.K).reshape(self.K, self.D)
        
        
        #mu0 = np.zeros(self.K)
    

        if self.static:
            A = np.zeros((self.K, self.K))
            self.mu = self.mu0
        else:
            A = np.random.multivariate_normal(np.zeros(self.K), self.lmd * np.eye(self.K), size = self.K)
            self.mu = np.random.normal(size = self.K)
        
        y = None
        if self.D > 0:
            y = []
        z = []
        for i in range(I):
            if self.D > 0:
                y.append(np.random.multivariate_normal(np.zeros(self.D), np.eye(self.D), size = N))
            
            zin = []
            zin.append(np.random.multivariate_normal(self.mu0, self.S0))
            for n in range(1, N):
                Th = A @ zin[-1] + self.mu
                if self.D > 0:
                    Th += V @ y[i][n]
                zin.append(np.random.multivariate_normal(Th, self.Q))
            z.append(zin)
        self.z = z
                
            
        # Generate multinomial
        X = []
        mean = []
        for p in range(self.P[0]):
            Xin = []
            meanin = []
            for i in range(I):
                C = np.random.poisson(self.C_max[p],N) + 1 
                Th = z[i] @ self.W[p].T # + self.b[p]
                Th_ext = np.append(Th, np.zeros((Th.shape[0],1)), axis = 1)
                prob = softmax(Th_ext, axis = 1)
                Xtmp = np.array([np.random.multinomial(C[n], prob[n])[:-1] for n in range(N)])
                Xin.append(np.append(Xtmp, C[None].T, axis = 1))  #Last column is total number of counts
                meanin.append(prob)
            X.append(Xin)
            mean.append(meanin)
            
            
        # Generate Gauss
        self.Sigma = []
        for p in range(self.P[1]):
            self.Sigma.append(self.tau * np.random.gamma(1,1) * np.identity(int(self.M[p + self.P[0]]))) # Diagonal
            #Prec = make_sparse_spd_matrix(dim = int(self.M[p + self.P[0]]), alpha = self.alpha, norm_diag = True)
            #Cov = np.linalg.inv(Prec)
            #self.Sigma.append(norm_cov(Cov))

        for p in range(self.P[1]):
            Xin = []
            meanin = []
            for i in range(I):
                Th = z[i] @ self.W[p + self.P[0]].T
                Xin.append(np.array([np.random.multivariate_normal(Th[n], self.Sigma[p]) for n in range(N)]))
                meanin.append(Th)
            X.append(Xin)
            mean.append(meanin)
            
        return X, y, mean
        
    def generate_mask(self, X):
        
        I = len(X[0])
        N = len(X[0][0])
        
        # Generate missing mask
        mask = []
        for i in range(I):
            ind = np.random.choice(sum(self.P), size = N, p = (1- self.mr, self.mr))
            mask.append(np.array([np.eye(sum(self.P))[np.random.choice(sum(self.P))] if ind[n] == 1 else np.ones(sum(self.P)) for  n in range(N)]))
        
        return mask
    
    def train_test_split(self, X, y, mean, mask = None):
        
        N = len(X[0][0])
        I = len(X[0])

        Xte = [[X[p][i][int(N*self.tr_ratio):,:] for i in range(I)] for p in range(sum(self.P))]
        X = [[X[p][i][:int(N*self.tr_ratio),:] for i in range(I)] for p in range(sum(self.P))]
        
        meante = [[mean[p][i][int(N*self.tr_ratio):,:] for i in range(I)] for p in range(self.P[0])]
        mean = [[mean[p][i][:int(N*self.tr_ratio),:] for i in range(I)] for p in range(self.P[0])]
        
        if y is not None:
            yte = [y[i][int(N*self.tr_ratio):,:] for i in range(I)]
            y = [y[i][:int(N*self.tr_ratio),:] for i in range(I)]
        else:
            yte = y
            
        arr = [X, y, mean, Xte, yte, meante]
            
        if mask is not None:
            maskte = [mask[i][int(N*self.tr_ratio):,:] for i in range(I)]
            mask = [mask[i][:int(N*self.tr_ratio),:]  for i in range(I)]
            arr += [mask, maskte]
            
        return arr

if __name__ == "__main__":
    
    static = False # Dynamic / Static
    
    I = 1 # Number of sequences
    K = 3  # Latent space dimension
    N = 200 # Number of instances
    D = 2 # Dimension of the covariates
    mr = 0.1 # Random missing rate
    
    P = [1]  # of multinomial modalities
    M = [5] * P[0] # Observation dimension
    C_max = [10] * P[0] # Total number of counts - 1 (if 0 -> Categorical)
    
    P += [1]  # of gaussian modalities
    M = np.append(M, [120] * P[1]) # Observation dimension
    
    data = data_gen(K, P, M, C_max, mr, D, static)
    X, y, mean = data.generate_data(N, I)
    mask = data.generate_mask(X)
    X, y, mean, Xte, yte, meante, mask, maskte = data.train_test_split(X, y, mean, mask)
    
    
    
    
    
    