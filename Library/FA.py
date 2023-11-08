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

from data_gen import data_gen
from utils import logdet, gauss_entropy, gauss_loglik

'''
To-do:
    - Add sparse covariance estimation
    
'''

class FA():
    
    def __init__(self, M, K = 5, D = 0):
        
        self.M = M # Observation dimension
        self.K = K # Latent space dimension
        self.D = D # Covariate dimension
        
        if self.D > 0:
            self.V = np.random.normal(size = D * K).reshape(K, D)
    
        self.W = np.random.normal(size = M * K).reshape(M, K)
        self.b = np.zeros(M)
        self.a = np.zeros(K)
            
        self.Sigma_x = np.identity(M)
        self.Prec_x = np.identity(M)

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
        
        self.ll_obs = gauss_loglik(x, mu_Th, self.Prec_x, mu_S = mu_S)
        self.ll_lv = gauss_loglik(self.mu_z, self.mu0, np.linalg.inv(self.S_prior), self.S_z)
        self.ll_lv += gauss_entropy(self.S_z)
        
        elbo = self.ll_obs + self.ll_lv
                
        return elbo
    
    def e_step_inf(self, x, y = None):
        
        self.Inf_S_out = np.tile(self.W.T @ self.Prec_x @ self.W, (x.shape[0],1,1)) 
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
        # else:
        #     self.a =  np.mean(self.mu_z, axis = 0)
            
            
        mu_z_aug = np.append(self.mu_z, np.ones((x.shape[0], 1)), axis = 1)
        S_aug = np.block([[self.mu_z.T @ self.mu_z + np.sum(self.S_z,axis = 0), np.sum(self.mu_z, axis = 0)[None].T],
                          [np.sum(self.mu_z, axis = 0)[None] , x.shape[0]*np.ones(1)[None]]])
        W_aug = ( x.T @  mu_z_aug) @ np.linalg.inv(S_aug)
        self.b = W_aug[:, -1]
        self.W = W_aug[:, :-1]
        self.Sigma_x = np.diag((np.sum(x * x, axis = 0)  - np.diag(W_aug @ ( x.T @  mu_z_aug).T)) / x.shape[0])
        
        # self.W = ( x.T @  self.mu_z) @ np.linalg.inv(self.mu_z.T @ self.mu_z + np.sum(self.S_z,axis = 0))
        # self.Sigma_x = np.diag((np.sum(x * x, axis = 0)  - np.diag(self.W @ ( x.T @  self.mu_z).T)) / x.shape[0])
        
        
        self.Prec_x = np.diag(1/np.diag(self.Sigma_x))
        self.Ai = self.Sigma_x
        
                
    def fit(self, x, y = None, epoch = 50):
        
        elbo_old = 0
        elbo_vec = []
        
        self.init_with_data(x, y)
        
        for e in range(epoch):
            
            self.e_step(x, y)
            
            elbo = self.elbo_multi(x, y)
            cnv = elbo - elbo_old
            print(cnv)
            elbo_old = elbo
            
            self.m_step(x, y)
            
            elbo_vec.append(elbo)
            if cnv < 1e-5 and e > 0:
                break
            
        self.elbo_vec = elbo_vec
        
    def impute(self):
        
        return self.mu_z @ self.W.T + self.b
        
    def predict(self, x, y = None):
        
        self.e_step(x, y)
        
        return self.impute()
        
        
if __name__ == "__main__":
    
    
    
    K = 3  # Latent space dimension
    N = 20 # Number of instances
    D = 0 # Dimension of the covariates
    M = 6
    
    # Data Generate
    #a = np.zeros(K)
    a = np.random.multivariate_normal(np.zeros(K), np.eye(K))
    mu0 = np.tile(a, (N, 1))
    y = None
    if D > 0:
        V = np.random.normal(size = D * K).reshape(K, D)
        y = np.random.normal(size = N * D).reshape(N, D)
        mu0 += y @ V.T
    z = np.array([np.random.multivariate_normal(mu0[i], np.eye(K)) for i in range(N)])
    W = np.random.multivariate_normal(np.zeros(K), np.eye(K), size = M)
    b = np.zeros(M)
    #b = np.random.multivariate_normal(np.zeros(M), np.eye(M))
    Sigma = 1 * np.eye(M)
    mean = z @ W.T + b
    X = np.array([np.random.multivariate_normal(mean[n], Sigma) for n in range(N)])
    
    # Model fit
    model = FA(M = M, K = K, D = D)
    model.fit(X, y, epoch = 500)
    X_pred = model.predict(X, y)
    plt.plot(model.elbo_vec)
    
    # Predictions
    Cov_pred = model.W @ model.S_prior @ model.W.T + model.Sigma_x
    Cov_gt = W @ W.T + Sigma
    
    if D > 0:
        mean_pred = (y @ model.V.T + model.a) @ model.W.T +  model.b
        mean_gt = (y @ V.T + a) @ W.T + b
    else:
        mean_pred = model.W @ model.a +  model.b
        mean_gt = W @ a + b
    
    print('MSE-mean:', np.mean((mean_pred - mean_gt)**2))
    print('MSE-cov:', np.mean((Cov_pred - Cov_gt)**2))
    
    