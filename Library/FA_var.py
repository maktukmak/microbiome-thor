import numpy as np
from scipy.linalg import norm
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.stats import invwishart, invgamma
import matplotlib.pyplot as plt
from scipy.special import loggamma, multigammaln, gamma, digamma, softmax, logsumexp
import seaborn as sns
from scipy.optimize import minimize
sns.set_theme()
import matplotlib.pyplot as plt
from sklearn.covariance import graphical_lasso
from scipy.stats import gamma
from data_gen import data_gen
from utils import logdet, gauss_entropy, gauss_loglik

'''
To-do:
    - Put prior on Prec_x, lmd

'''

class FA_var():
    
    def __init__(self, M, K = 5, D = 5):
        
        self.M = M # Observation dimension
        self.K = K # Latent space dimension
        self.D = D # Covariate dimension
        
        self.mod = 'FA' # 'FA', 'pPCA'
        
        if self.D > 0:
            self.V = np.random.normal(size = D * K).reshape(K, D)
            
        self.lmd = 1 * np.ones(self.K)
        self.Gam = np.tile(1*np.eye(K), (self.M,1,1))
        self.W = np.random.normal(size = M * (K)).reshape(M, K)
        self.b = np.zeros(M)
        self.S_b = np.eye(M)
        
        self.a = np.zeros(K)
        
        self.Prec_x = np.identity(M)
        self.Sigma_x = np.linalg.inv(self.Prec_x) 

        self.S_prior = np.eye(K)
        self.Inf_S_out = 0
        self.Inf_mu_out = 0
        
    def init_with_data(self, x, y = None):
        
        self.mu0 = np.tile(self.a, (x.shape[0], 1))
        if self.D > 0:
            self.mu0 += y @ self.V.T

        self.mu_z = self.mu0
        self.S_z = np.array([np.eye(self.K) for i in range(x.shape[0])])

    
    def elbo_multi(self, x, y = None):
        
        self.mu0 = np.tile(self.a, (x.shape[0], 1))
        if self.D > 0:
            self.mu0 += y @ self.V.T
            
        mu_Th = self.mu_z @ self.W.T + self.b
        mu_S = self.W @ self.S_z @ self.W.T
        mmT = np.einsum('ij,ik->ijk', self.mu_z , self.mu_z)
        
        
        self.ll_obs = gauss_loglik(x, mu_Th, self.Prec_x, mu_S = mu_S)
        for j in range(self.M):
            self.ll_obs += - np.sum(0.5 * self.Prec_x[j][j] * np.trace((mmT + self.S_z) @ self.Gam[j], axis1=1, axis2=2))
        self.ll_obs += - 0.5 * len(x) * np.trace(self.Prec_x @ self.S_b)
        
        self.ll_lv_loc = 0
        for j in range(self.K):
            self.ll_lv_loc += gauss_loglik(self.W[:,j][None], np.zeros(self.M), self.lmd[j] * np.eye(M))
        self.ll_lv_loc += - 0.5 * np.trace(self.lmd * sum(self.Gam))
        self.ll_lv_loc += gauss_entropy(self.Gam)
        self.ll_lv_loc += gauss_loglik(self.b[None], np.zeros(self.M), np.eye(M))
        self.ll_lv_loc += - 0.5 * np.trace(self.S_b)
        self.ll_lv_loc += gauss_entropy(self.S_b)
        
        
        self.ll_lv = gauss_loglik(self.mu_z, self.mu0, np.linalg.inv(self.S_prior), self.S_z)
        self.ll_lv += gauss_entropy(self.S_z)
        
        elbo = self.ll_obs + self.ll_lv + self.ll_lv_loc
                
        return elbo
    
    def e_step_inf(self, x, y = None):
        
        
        self.Inf_S_out = np.tile(self.W.T @ self.Prec_x @ self.W 
                                 + sum(np.diag(self.Prec_x)[:, None, None] * self.Gam)
                                 , (x.shape[0],1,1)) 
        self.Inf_mu_out = ((x - self.b) / np.diag(self.Sigma_x)[None]) @ self.W

    def e_step(self, x, y = None, Inf_mu_in = 0, Inf_S_in = 0):
        
        self.mu0 = np.tile(self.a, (x.shape[0], 1))
        if self.D > 0:
            self.mu0 += y @ self.V.T
            
        self.e_step_inf(x, y)
        Inf_S = self.Inf_S_out + np.linalg.inv(self.S_prior) + Inf_S_in
        Inf_mu = self.Inf_mu_out + self.mu0 @ np.linalg.inv(self.S_prior) + Inf_mu_in
        
        self.S_z =  np.linalg.inv(Inf_S)
        self.mu_z = np.einsum('ij,ijk->ik', Inf_mu , self.S_z)

    def m_step(self, x, y = None):
        
        self.mu0 = np.tile(self.a, (x.shape[0], 1))
        if self.D > 0:
            y_aug = np.append(y, np.ones((y.shape[0], 1)), axis = 1)
            V_aug  = (self.mu_z.T @ y_aug) @ np.linalg.inv(y_aug.T @ y_aug)
            self.V =  V_aug[:, :-1]
            self.a = V_aug[:, -1]
            
        mmT = np.einsum('ij,ik->ijk', self.mu_z , self.mu_z)
        ESS2 = self.mu_z.T @ self.mu_z + np.sum(self.S_z,axis = 0)
        
        
        for j in range(self.M):
            
            self.Gam[j] = np.linalg.inv(np.diag(self.lmd * np.ones(self.K))  +  self.Prec_x[j][j] * ESS2)
            self.W[j] = self.Prec_x[j][j] * ( (x[:,j] - self.b[j]).T @  self.mu_z) @ self.Gam[j]
            
            self.S_b = np.linalg.inv(np.eye(self.M) + len(x) * self.Prec_x)
            self.b = self.S_b @ self.Prec_x @ sum(x - self.mu_z @ self.W.T)

        self.Prec_x = (len(x)) / (np.sum((x - self.mu_z @ self.W.T - self.b) * (x - self.mu_z @ self.W.T - self.b), axis = 0)
                    + np.sum(np.trace((mmT + self.S_z) @ self.Gam[0], axis1=1, axis2=2))
                    + [(self.W[j] @ self.S_z @ self.W[j].T).sum() for j in range(self.M)]
                    + len(x) * np.diag(self.S_b))
        
        #self.Sigma_x = (1 / len(x)) * (np.sum((x - self.mu_z @ self.W.T - self.b) * (x - self.mu_z @ self.W.T - self.b), axis = 0)
        #            + np.sum(np.trace((mmT + self.S_z) @ self.Gam[0], axis1=1, axis2=2))
        #            + [(self.W[j] @ self.S_z @ self.W[j].T).sum() for j in range(self.M)]
        #            + len(x) * np.diag(self.S_b))
        
        if self.mod == 'FA':
            self.Sigma_x = np.diag(self.Prec_x)
        elif self.mod == 'pPCA':
            self.Sigma_x = np.mean(self.Prec_x) * np.eye(self.M)
            
        self.Prec_x = np.linalg.inv(self.Sigma_x) 
        
        self.Ai = self.Sigma_x
        
        self.lmd = (self.M) / np.array([self.W[:,j] @ self.W[:,j] + self.M * self.Gam[0][j,j] for j in range(self.K)])
        #self.lmd = 3 / np.array([self.W[j] @ self.W[j] + np.trace(self.Gam[j]) + 2 for j in range(self.M)])
  
    def fit(self, x, y = None, epoch = 50):
        
        elbo_old = 0
        self.elbo_vec = []
        
        self.init_with_data(x, y)
        
        for e in range(epoch):
            
            self.e_step(x, y)
            
            elbo = self.elbo_multi(x, y)
            cnv = elbo - elbo_old
            print(cnv)
            if abs(cnv) < 10**(-5):
                break
            elbo_old = elbo
            
            self.m_step(x, y)
            
            self.elbo_vec.append(elbo)
            
        
    def impute(self):
        
        return self.mu_z @ self.W.T + self.b
        
    def predict(self, x, y = None):
        
        self.e_step(x, y)
        
        return self.impute()
        
        
if __name__ == "__main__":
    
    
    K = 3 # Latent space dimension
    N = 100 # Number of instances
    D = 0 # Dimension of the covariates
    M = 20
    
    # Data Generate
    a = np.zeros(K)
    mu0 = np.tile(a, (N, 1))
    y = None
    if D > 0:
        V = np.random.normal(size = D * K).reshape(K, D)
        y = np.random.normal(size = N * D).reshape(N, D)
        mu0 += y @ V.T
    z = np.array([np.random.multivariate_normal(mu0[i], np.eye(K)) for i in range(N)])
    W = np.random.multivariate_normal(np.zeros(K), np.eye(K), size = M)
    b = np.random.multivariate_normal(np.zeros(M), np.eye(M))
    Sigma = 1 * np.eye(M)
    mean = z @ W.T + b
    X = np.array([np.random.multivariate_normal(mean[n], Sigma) for n in range(N)])
    
    # Model fit
    model = FA_var(M = M, K = M, D = D)
    model.fit(X, y, epoch = 2000)
    #X_pred = model.predict(X, y)
    plt.plot(model.elbo_vec)
    plt.show()
    plt.bar(np.arange(M), 1/model.lmd)
    plt.show()
    
    # Predictions
    Cov_pred = model.W @ model.W.T + model.Sigma_x
    Cov_gt = W @ W.T + Sigma
    mean_pred = model.b
    mean_gt = b
    
    plt.imshow(Cov_gt)
    plt.show()
    plt.imshow(Cov_pred)
    plt.show()
    
    print('MSE-mean:', np.mean((mean_pred - mean_gt)**2))
    print('MSE-cov:', np.mean((Cov_pred - Cov_gt)**2))
    
    
    