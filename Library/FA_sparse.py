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

class FA_sparse():
    
    def __init__(self, M, K = 5, D = 5):
        
        self.M = M # Observation dimension
        self.K = K # Latent space dimension
        self.D = D # Covariate dimension
        
        self.covariates = False
        if D > 0:
            self.covariates = True
            self.V = np.random.normal(size = D * K).reshape(K, D)
    
        self.W = np.random.normal(size = M * K).reshape(M, K)
        self.b = np.zeros(M)
        self.a = np.zeros(K)
            
        self.sparse = False
        self.Sigma_x = np.identity(M)
        self.Prec_x = np.identity(M)

        self.S_prior = np.eye(K)
        self.Inf_S_out = 0
        self.Inf_mu_out = 0
        
    def init_with_data(self, x, y = None):
        
        self.mu0 = np.tile(self.a, (x.shape[0], 1))
        if self.covariates:
            self.mu0 += y @ self.V.T

        self.mu_z = self.mu0
        self.S_z = np.array([np.eye(self.K) for i in range(x.shape[0])])

    
    def elbo_multi(self, x, y = None):
        
        self.mu0 = np.tile(self.a, (x.shape[0], 1))
        if self.covariates:
            self.mu0 += y @ self.V.T

        # Observations
        mu_Th = self.mu_z @ self.W.T + self.b
        mu_S = self.W @ self.S_z @ self.W.T
        
        self.ll_obs = gauss_loglik(x, mu_Th, self.Prec_x, mu_S = mu_S)
        self.ll_lv = gauss_loglik(self.mu_z, self.mu0, np.linalg.inv(self.S_prior), self.S_z)
        self.ll_lv += gauss_entropy(self.S_z)
        
        # Entropies
        elbo = self.ll_obs + self.ll_lv
                
        return elbo
    
    def e_step_inf(self, x, y = None):
        
        self.Inf_S_out = np.tile(self.W.T @ self.Prec_x @ self.W, (x.shape[0],1,1)) 
        #self.Inf_mu_out = ((x - self.b) / np.diag(self.Sigma_x)[None]) @ self.W
        self.Inf_mu_out = (x - self.b) @ self.Prec_x @ self.W
        
        

    def e_step(self, x, y = None, Inf_mu_in = 0, Inf_S_in = 0):
        
        self.mu0 = np.tile(self.a, (x.shape[0], 1))
        if self.covariates:
            self.mu0 = y @ self.V.T + self.a
            
        Inf_mu = np.zeros((x.shape[0], self.K))
        Inf_S = np.zeros((x.shape[0], self.K, self.K))
        
        Inf_S += np.linalg.inv(self.S_prior)
        Inf_mu += self.mu0 @ np.linalg.inv(self.S_prior) 
        
        self.e_step_inf(x, y)
    
        Inf_S += self.Inf_S_out
        Inf_mu += self.Inf_mu_out
        
        Inf_S += Inf_S_in
        Inf_mu += Inf_mu_in
        
        self.S_z =  np.linalg.inv(Inf_S)
        self.mu_z = np.einsum('ij,ijk->ik', Inf_mu , self.S_z)
        

    def m_step(self, x, y = None):
        
        self.mu0 = np.tile(self.a, (x.shape[0], 1))
        if self.covariates:
            y_aug = np.append(y, np.ones((y.shape[0], 1)), axis = 1)
            V_aug  = (self.mu_z.T @ y_aug) @ np.linalg.inv(y_aug.T @ y_aug)
            self.V =  V_aug[:, :-1]
            self.a = V_aug[:, -1]
            
        mu_z_aug = np.append(self.mu_z, np.ones((x.shape[0], 1)), axis = 1)
        S_aug = np.block([[self.mu_z.T @ self.mu_z + np.sum(self.S_z,axis = 0), np.sum(self.mu_z, axis = 0)[None].T],
                          [np.sum(self.mu_z, axis = 0)[None] , x.shape[0]*np.ones(1)[None]]])
        
        
        W_aug = ( x.T @  mu_z_aug) @ np.linalg.inv(S_aug)
        self.W = W_aug[:, :-1]
        self.b = W_aug[:, -1]
        
        #self.W = ( x.T @  self.mu_z) @ np.linalg.inv(self.mu_z.T @ self.mu_z + np.sum(self.S_z,axis = 0))
        
        #self.mu_prior = np.mean(self.mu_z, axis = 0)
        #self.S_prior = ((self.mu_z - self.mu0).T @ (self.mu_z - self.mu0) + np.sum(self.S_z, axis = 0)) / x.shape[0]
        
        
        
        if self.sparse:
            
            emp_cov = (x - mu_z_aug @ W_aug.T).T @ (x - mu_z_aug @ W_aug.T) + np.sum(self.W @ self.S_z @ self.W.T, axis = 0)
            emp_cov /= x.shape[0]
            self.Sigma_x, self.Prec_x = graphical_lasso(emp_cov, alpha = 0.5, max_iter = 100)
        else:
            self.Sigma_x = np.diag((np.sum(x * x, axis = 0)  - np.diag(W_aug @ ( x.T @  mu_z_aug).T)) / x.shape[0])
            self.Prec_x = np.diag(1/np.diag(self.Sigma_x))
        
        self.Ai = self.Sigma_x
        
                
    def fit(self, x, y = None, epoch = 50):
        
        elbo_old = 0
        elbo_vec = []
        
        self.init_with_data(x, y)
        
        for e in range(epoch):
            
            self.e_step(x, y)
            
            elbo = self.elbo_multi(x, y)
            #elbo = 0
            print(elbo - elbo_old)
            elbo_old = elbo
            
            self.m_step(x, y)
            
            elbo_vec.append(elbo)
            
        self.elbo_vec = elbo_vec
        
    def impute(self):
        
        return self.mu_z @ self.W.T + self.b
        
    def predict(self, x, y = None):
        
        self.e_step(x, y)
        
        return self.impute()
        
        
if __name__ == "__main__":
    
    
    static = True # Dynamic / Static
    
    I = 1 # Number of sequences
    K = 3  # Latent space dimension
    N = 200 # Number of instances
    D = 0 # Dimension of the covariates
    mr = 0.0 # Random missing rate
    
    P = [0]  # of multinomial modalities
    M = [20] * P[0] # Observation dimension
    C_max = [100] * P[0] # Total number of counts - 1 (if 0 -> Categorical)
    
    P += [1]  # of gaussian modalities
    M = np.append(M, [20] * P[1]) # Observation dimension
    
    data = data_gen(K, P, M, C_max, mr, D, static)
    X, y, mean = data.generate_data(N, I)
    

    model = FA_sparse(M = int(M[0]), K = K, D = D)
    model.sparse = True
    model.fit(X[0][0], y)
    X_pred = model.predict(X[0][0], y)
    
    plt.plot(model.elbo_vec)
    
    print('MSE:', np.mean(np.mean((X_pred - X[0][0])**2, axis = 1)))
    
    
    cov_org = data.W[0] @ data.W[0].T
    cov_pred = model.W @ model.W.T
    
    sparse_org = data.Sigma[0]
    sparse_pred = model.Sigma_x
    
    
    Prec_x = model.Prec_x
    Sigma_x = model.Sigma_x