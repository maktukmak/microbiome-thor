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
import time
from sklearn.datasets import make_sparse_spd_matrix, make_spd_matrix

from sklearn.linear_model import LogisticRegression
from scipy.linalg import block_diag

from multPCA import multPCA
from FA import FA
import copy
from data_gen import data_gen
from sklearn.covariance import graphical_lasso
import pickle
from numpy.linalg import inv as inv

from utils import norm_cov

'''
To-do:
    - Add covariates (m-step)
'''

class multi_treatment():
    
    def __init__(self, P, M, 
                 K = 5, 
                 step_comp_elbo = 50,
                 covariates = False,
                 D = 5):
        
        self.M = M # Observation dimension
        self.K = K # Latent space dimension
        self.P = P # Number of modalities (P[0] -> Multinomial, P[1] -> Gaussian)
        self.D = D # Covariate dimension
        self.covariates = covariates
        
        self.alpha = 0.5
        
        self.comps = [FA(M, K, covariates, D) for p in range(P)]
        self.step_comp_elbo = step_comp_elbo
        
    def project(self, x, mask):
        
        x_proj = [self.comps[p].impute() for p in range(P)]
        ind = np.where(mask == 0)
        
        x_imp = copy.deepcopy(x)
        
        for i in range(len(ind[0])):
            x_imp[ind[1][i]][ind[0][i]] = x_proj[ind[1][i]][ind[0][i]]

        return x_imp
        
    def e_step(self, x, y = None):
        
           
        [self.comps[p].e_step(x[p], y) for p in range(self.P)]
        
    def m_step(self, x, y = None):
        
        
        W = sum([x[p].T @  self.comps[p].mu_z for p in range(self.P)]) @ np.linalg.inv(sum([self.comps[p].mu_z.T @ self.comps[p].mu_z + np.sum(self.comps[p].S_z,axis = 0) for p in range(self.P)]))
        
        for p in range(self.P):
            self.comps[p].a = np.mean(self.comps[p].mu_z, axis = 0)
            
            emp_cov = (x[p] - self.comps[p].mu_z @ W.T).T @ (x[p] - self.comps[p].mu_z @ W.T) + np.sum(W @ self.comps[p].S_z @ W.T, axis = 0)
            emp_cov /= x[p].shape[0]
            self.comps[p].Sigma_x, self.comps[p].Prec_x = graphical_lasso(emp_cov, alpha = self.alpha,
                                                                          #cov_init = self.comps[p].Sigma_x,
                                                                          mode = 'lars',
                                                                          max_iter = 10)
            self.comps[p].W = W
            
        
    def compute_test_elbo(self, x, y = None, mask = None):
        
        [self.comps[p].init_with_data(x[p], y) for p in range(sum(self.P))]
        
        if mask is None:
            x_imp = x
        else:
            x_imp = self.project(x, mask)
        self.e_step(x_imp, y)
        
        
        elbo = sum([self.comps[p].elbo_multi(x[p], y) for p in range(sum(self.P))])
        elbo -= (self.comps[-1].ll_lv) * (sum(self.P)-1)
        return elbo
            
    def fit(self, x, y = None, mask = None, epoch = 500, step_comp_elbo = 1):
        
        elbo_old = 0
        elbo_vec = []
        
        [self.comps[p].init_with_data(x[p], y) for p in range(self.P)]
        
        for e in range(1, epoch):
            
            self.e_step(x, y)
            
            if e % step_comp_elbo == 0:
                elbo = sum([self.comps[p].elbo_multi(x[p], y) for p in range(self.P)])
                print(elbo - elbo_old)
                elbo_old = elbo
                elbo_vec.append(elbo)
            
            self.m_step(x, y)
            
            self.elbo_vec = elbo_vec
            
        #self.compute_induced_cov()
        
    def compute_induced_cov(self):
        
        W = np.vstack([self.comps[p].W for p in range(sum(self.P))])
        Ai = np.zeros((W.shape[0], W.shape[0]))
        
        cnt = 0
        for p in range(sum(self.P)):
            
            Ai[cnt:cnt + self.comps[p].Ai.shape[0], cnt:cnt + self.comps[p].Ai.shape[0]] = self.comps[p].Ai
            cnt += self.comps[p].Ai.shape[0]
        
        self.S_ind = W @ self.comps[0].S_prior @ W.T + 0.1 * np.eye(W.shape[0])
        
    def impute(self, x, y = None):
        
        if self.full_dim:
            xp = [softmax(self.comps[p].mu_z @ self.comps[p].W.T, axis = 1) for p in range(self.P[0])]
        else:
            prob = [softmax(np.append(self.comps[p].mu_z @ self.comps[p].W.T, np.zeros((x[p].shape[0],1)), axis = 1), axis = 1) for p in range(self.P[0])]
            xp = [np.round(prob[p][:,:-1] * x[p][:,-1][None].T) for p in range(self.P[0])]
            xp = [np.append(xp[p], x[p][:,-1][None].T, axis = 1) for p in range(self.P[0])]
        
        xp += [self.comps[p + self.P[0]].mu_z @ self.comps[p + self.P[0]].W.T  for p in range(self.P[1])]
        prob += [self.comps[p + self.P[0]].mu_z @ self.comps[p + self.P[0]].W.T  for p in range(self.P[1])]
        
        return xp, prob
    
    def predict(self, x, y = None, mask = None):
        
        [self.comps[p].init_with_data(x[p], y) for p in range(sum(self.P))]

        if mask is None:
            x_imp = x
        else:
            x_imp = self.project(x, mask)
        
        self.e_step(x_imp, y)
        xp, prob = self.impute(x_imp, y)

        return xp, prob 
        


if __name__ == "__main__":
    
    def norm_cov(cov):
        d = np.sqrt(np.diag(cov))
        cov = cov / d
        cov /= d[:, np.newaxis]

        return cov
    
    
    static = True # Dynamic / Static
        
    I = 1 # Number of sequences
    K = 5  # Latent space dimension
    N = 100 # Number of instances
    D = 0 # Dimension of the covariates
    mr = 0.0 # Random missing rate
    T = 2 # Number of treatments
    
    P = [0]  # of multinomial modalities
    M = [20] * P[0] # Observation dimension
    C_max = [100] * P[0] # Total number of counts - 1 (if 0 -> Categorical)
    
    P += [1]  # of gaussian modalities
    M = np.append(M, [120] * P[1]) # Observation dimension
    
    
    X_t = []
    Prec_sp_gt = []
    data = data_gen(K, P, M, C_max, mr, D, static)
    data.alpha = 0.95
    data.S0 = np.eye(K)
    data.Q = np.eye(K)
    data.mu0 = np.zeros(K)
    for t in range(T):
        X, y, mean = data.generate_data(N, I)
        Prec_sp_gt.append(np.linalg.inv(data.Sigma[0]))
        X_t += X[0]
        
    M = int(M[0])
    
    Cov_sample = (X_t[0].T @ X_t[0])/N
    print(np.linalg.cond(Cov_sample))
    
    S, P = graphical_lasso(Cov_sample, alpha = 0.5,
                                    #cov_init = self.comps[p].Sigma_x,
                                    mode = 'cd',
                                    max_iter = 100)
    
    
    
    
    
    
    
    
    
    
    
    # Fit
    model = multi_treatment(T, M, K, step_comp_elbo = 2, covariates = (D > 0), D = D)
    model.alpha = 0.5
    model.fit(X_t, epoch = 50)
    plt.plot(model.elbo_vec)
    plt.show()
    
    Cov_lr_gt = data.W[0] @ Cov_l @ data.W[0].T
    Prec_sp0_gt = Prec_sp_gt[0]
    Prec_sp1_gt = Prec_sp_gt[1]
    
    Cov_lr = model.comps[0].W @ model.comps[0].W.T
    Prec_sp0 = model.comps[0].Prec_x
    Prec_sp1= model.comps[1].Prec_x
    
    plt.imshow(Cov_lr)
    plt.grid(None)
    plt.show()
    plt.imshow(Cov_lr_gt)
    plt.grid(None)
    plt.show()
    
    plt.imshow((norm_cov(Prec_sp0)))
    plt.grid(None)
    plt.show()
    plt.imshow((norm_cov(Prec_sp0_gt)))
    plt.grid(None)
    plt.show()
    
    plt.imshow(norm_cov(Prec_sp1))
    plt.grid(None)
    plt.show()
    plt.imshow(norm_cov(Prec_sp1_gt))
    plt.grid(None)
    plt.show()
    
    
    W = np.concatenate((data.W[0], data.W[0]), axis = 0)
    Prec_sp = block_diag(Prec_sp_gt[0],Prec_sp_gt[1])
    Cov = W @ Cov_l @ W.T + np.linalg.inv(Prec_sp)
    w, v = np.linalg.eig(Cov)
    w = np.real(w)
    v = np.real(v)
    
    Cov_eig = v[:, 3:] @ np.diag(w[3:]) @  v[:, 3:].T
    asd = Cov - Cov_eig
    
    # plt.imshow(norm_cov(Cov))
    # plt.grid(None)
    # plt.show()
    
    

