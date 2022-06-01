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
from multLDS import multLDS
import pickle
from data_gen import data_gen
from utils import logdet, gauss_entropy, multi_loglik_bounded, gauss_loglik, norm_cov
import time

'''
To-do:
    
    MAP estimate on W
    
    - Slow convergence when high counts
    
    done - A inversion
    done - Impute extra 0
    done - Full covariance
    
    done - Full dimension
    
    done - Add covariates
    done - Categorical problem
    done - Elbo
    
    
'''

class multPCA_map():
    
    def __init__(self, M, K = 5, D = 5, full_dim = False, lmd = 5):
        
        self.M = M # Observation dimension
        self.K = K # Latent space dimension
        self.D = D # Covariate dimension
        self.full_dim = full_dim
        
        self.lmd = lmd
        if full_dim:
            self.W = np.random.normal(size = (M+1) * K).reshape(M+1, K)
            self.A = self.A_m(self.M + 1)
            self.Gam = np.tile(1*np.eye(K), (M+1,1,1))
        else:
            self.W = np.random.normal(size = M * K).reshape(M, K)
            self.A = self.A_m(self.M)
            self.Gam = np.tile(1*np.eye(K), (M,1,1))
        
        self.Ai = np.linalg.inv(self.A)
        
        self.mu_prior = np.zeros(K)
        if self.D > 0:
            self.V = np.random.normal(size = D * K).reshape(K, D)
            
        self.S_prior = np.eye(K)
        self.Inf_S_out = 0
        self.Inf_mu_out = 0
        
        self.it_bound = 10
            
    def init_with_data(self, x, y = None):
        
        if self.D > 0:
            self.mu_prior = y @ self.V.T
            self.mu_z = self.mu_prior
        else:
            self.mu_z = np.tile(self.mu_prior, (x.shape[0], 1))

        self.S_z = [np.eye(self.K) for i in range(x.shape[0])]
        self.Phi = self.mu_z @ self.W.T

    def A_m(self, M):
        return  0.5*(np.eye(M) - (1/(M+1))*np.ones((M, M)))
    
    def elbo_multi(self, x, y = None):
        
        if self.D > 0:
            self.mu_prior = y @ self.V.T

        # Observations
        mu_Th = self.mu_z @ self.W.T
        Sigma_Th = self.W @ self.S_z @ self.W.T
        mmT = np.einsum('ij,ik->ijk', self.mu_z , self.mu_z)
        
        self.ll_obs = multi_loglik_bounded(x, mu_Th, self.Phi, self.A, self.full_dim, Sigma_Th)
        M = self.M
        if self.full_dim:
            M += 1
        #for j in range(M):
        #    self.ll_obs += - np.sum(0.5 * self.A[j][j] * np.trace((mmT + self.S_z) @ self.Gam[j], axis1=1, axis2=2))
        
        #self.ll_pr = 0
        #if self.lmd > 0:
        #    for j in range(self.K):
        #        self.ll_pr = gauss_loglik(self.W[:,j][None], np.zeros(self.M+1), self.lmd * self.A)
                
        self.ll_pr = gauss_loglik(self.W.flatten()[None], np.zeros(M * self.K), self.lmd * np.eye(M * self.K))
        
        self.ll_lv = gauss_loglik(self.mu_z, self.mu_prior, np.linalg.inv(self.S_prior), self.S_z)
        self.ll_lv += gauss_entropy(self.S_z)
        
        
        elbo = self.ll_obs + self.ll_lv + self.ll_pr
                
        return elbo
    
    def e_step_inf(self, x, y = None):
        
        A = self.A
        if self.full_dim:
            Phi_s = softmax(self.Phi, axis = 1)
            b = self.Phi @ A - Phi_s
            Ni = x[:, -1]
            xi = np.append(x[:, :-1], (Ni - np.sum(x[:, :-1], axis = 1))[None].T, axis = 1)
        else:
            Phi_ext = np.append(self.Phi, np.zeros((self.Phi.shape[0],1)), axis = 1)
            Phi_s = softmax(Phi_ext, axis = 1)
            b = self.Phi @ A - Phi_s[:, :-1]
            Ni = x[:, -1]
            xi = x[:, :-1]
        
        self.Inf_S_out = Ni[None][None].T * np.tile(self.W.T @ A @ self.W, (x.shape[0],1,1)) 
        self.Inf_mu_out = (xi + Ni[None].T * b) @ self.W

    
    def e_step(self, x, y = None, Inf_mu_in = 0, Inf_S_in = 0):
        
        if self.D > 0:
            self.mu_prior = y @ self.V.T
        
        for i in range(self.it_bound):
        
            Inf_mu = np.zeros((x.shape[0], self.K))
            Inf_S = np.zeros((x.shape[0], self.K, self.K))
            
            Inf_S += np.linalg.inv(self.S_prior)
            Inf_mu += self.mu_prior @ np.linalg.inv(self.S_prior) 
            
            self.e_step_inf(x, y)
        
            Inf_S += self.Inf_S_out
            Inf_mu += self.Inf_mu_out
            
            Inf_S += Inf_S_in
            Inf_mu += Inf_mu_in
            
            self.S_z =  np.linalg.inv(Inf_S)
            self.mu_z = np.einsum('ij,ijk->ik', Inf_mu , self.S_z)
            
            Phi_old = self.Phi.copy()
            self.Phi = self.mu_z @ self.W.T
            
            conv = np.sum((Phi_old - self.Phi)**2) / (Phi_old.shape[0] * Phi_old.shape[1])
            #print(conv)
            if conv < 1e-5:
                #print('Converged')
                break

    def m_step(self, x, y = None):
        
        A = self.A
        Ai = self.Ai
        if self.full_dim:
            Phi_s = softmax(self.Phi, axis = 1)
            b = self.Phi @ A - Phi_s
            Ni = x[:, -1][None].T
            xi = np.append(x[:, :-1], Ni - np.sum(x[:, :-1], axis = 1)[None].T, axis = 1)
        else:
            Phi_ext = np.append(self.Phi, np.zeros((self.Phi.shape[0],1)), axis = 1)
            Phi_s = softmax(Phi_ext, axis = 1)
            b = self.Phi @ A - Phi_s[:, :-1]
            Ni = x[:,-1][None].T
            xi = x[:,:-1]
        
        M = self.M
        if self.full_dim:
            M += 1
            
            
        #for j in range(M):
        #    self.Gam[j] = np.linalg.inv(self.lmd[j] * np.eye(self.K) +  (Ni * A[j,j] * self.mu_z).T @ self.mu_z + np.sum(np.expand_dims(Ni *  A[j,j],-1) * self.S_z,axis = 0))
        #    self.W[j] = (A[j,j] * (xi[:,j]  + Ni[:,0] * b[:,j]) @  self.mu_z) @ self.Gam[j]
            
        self.Gam = (Ni * self.mu_z).T @ self.mu_z + np.sum(np.expand_dims(Ni,-1) * self.S_z,axis = 0) + self.lmd * np.eye(self.K)
        self.W = ( ((xi  + Ni * b) @ Ai).T @  self.mu_z) @ np.linalg.inv(self.Gam)
        
            
        if self.D > 0:
            self.V =  (self.mu_z.T @ y) @ np.linalg.inv(y.T @ y)
            self.mu_prior = y @ self.V.T
        else:
            self.mu_prior = np.mean(self.mu_z, axis = 0)
            #self.mu_prior = np.tile(self.mu_prior, (x.shape[0], 1))
                
            
        #self.S_prior = ((self.mu_z - self.mu_prior).T @ (self.mu_z - self.mu_prior) + np.sum(self.S_z, axis = 0)) / x.shape[0]
                
    def fit(self, x, y = None, epoch = 500, step_comp_elbo = 1):
        
        elbo_old = 0
        self.elbo_vec = []
        
        self.init_with_data(x, y)
        
        for e in range(1, epoch):
            
            #s = time.time()
            self.e_step(x, y)
            #print('E-step:', time.time() - s)
            
            if e % step_comp_elbo == 0:
                
                #s = time.time()
                elbo = self.elbo_multi(x, y)
                #print('Elbo:', time.time() - s)
                
                print(elbo - elbo_old)
                elbo_old = elbo
                self.elbo_vec.append(elbo)
                self.elbo_vec = self.elbo_vec
            
            #s = time.time()
            self.m_step(x, y)
            #print('M-step:', time.time() - s)
        
    def compute_induced_cov(self):
        
        self.S_ind = self.W @ self.S_prior @ self.W.T + 0.1 * np.eye(self.W.shape[0])
        
    def impute(self):
        
        p = softmax(np.append(self.mu_z @ self.W.T, np.zeros((self.mu_z.shape[0],1)), axis = 1), axis = 1)
        
        return p
        
        
    def predict(self, x, y = None):
        
        self.init_with_data(x, y)
        self.e_step(x, y)
        
        p = self.impute()
        xp = np.round(p[:,:-1] * x[:,-1][None].T)
        xp = np.append(xp, x[:,-1][None].T, axis = 1)
        
        return xp, p
        
        
if __name__ == "__main__":
    
    #np.random.seed(0)
    
    K = 3  # Latent space dimension
    N = 34 # Number of instances
    D = 0 # Dimension of the covariates
    M = 610
    C_max = 900000 # Total number of counts - 1 (if 0 -> Categorical)
    full_dim = True
    
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
    if full_dim:
        W = np.random.multivariate_normal(np.zeros(K), np.eye(K), size = M+1)
        b = np.zeros(M+1)
    else:
        W = np.random.multivariate_normal(np.zeros(K), np.eye(K), size = M)
        b = np.zeros(M)
    #b = np.random.multivariate_normal(np.zeros(M), np.eye(M))
    mean = z @ W.T + b
    if not full_dim:
        mean = np.append(mean, np.zeros((mean.shape[0],1)), axis = 1)
    prob = softmax(mean, axis = 1)
    C = np.random.poisson(C_max,N) + 1 
    Xtmp = np.array([np.random.multinomial(C[n], prob[n])[:-1] for n in range(N)])
    X = np.append(Xtmp, C[None].T, axis = 1)  #Last column is total number of counts

    np.random.seed()
    
    model = multPCA_map(M = M, K = K, D = D, full_dim=full_dim, lmd = 2)
    model.fit(X, y, epoch = 100000, step_comp_elbo=1000)
    plt.plot(model.elbo_vec)
    plt.show()
    #Xp, prob_est = model.predict(X)
    
    
    # Predictions
    Cov_pred = model.W @ model.S_prior @ model.W.T# + model.Ai
    Cov_gt = W @ W.T# + model.Ai
    if D > 0:
        mean_pred = softmax(model.mu_prior @ model.W.T, axis = 1)
        mean_gt = softmax((y @ V.T + a) @ W.T + b, axis = 1)
        
    else:    
        mean_pred = softmax(model.W @ model.mu_prior)
        mean_gt = softmax(W @ a + b)
    
    
    print('Hel-mean:', np.mean(norm(np.sqrt(mean_pred) - np.sqrt(mean_gt), axis = -1) / np.sqrt(2)))
    print('MSE-cov:', np.mean((Cov_pred - Cov_gt)**2))
    
    plt.imshow(norm_cov(Cov_gt))
    plt.show()
    plt.imshow(norm_cov(Cov_pred))
    plt.show()
