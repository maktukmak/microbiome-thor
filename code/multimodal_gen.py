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

from sklearn.linear_model import LogisticRegression

from multPCA import multPCA
from multPCA_var import multPCA_var
from multPCA_map import multPCA_map
from FA import FA
import copy
from data_gen import data_gen
import pickle

'''
To-do:
    - Missing modalities MultPCA
    - (done) Induced covariance A
    - (done) FA induced covariance
    - Convergence criteria
    - (done) Add FA
    - (done) Joint training
    - (done) Joint elbo
'''

class multimodal_gen():
    
    def __init__(self, P, M, 
                 K = 5, 
                 D = 5,
                 full_dim = False):
        
        self.M = M # Observation dimension
        self.K = K # Latent space dimension
        self.P = P # Number of modalities (P[0] -> Multinomial, P[1] -> Gaussian)
        self.D = D # Covariate dimension
        self.full_dim = full_dim
        
        self.comps = [multPCA_map(M[p], K, D, full_dim) for p in range(P[0])]
        self.comps += [FA(M[p + P[0]], K, D) for p in range(P[1])]
        for p in range(P[0]):
            self.comps[p].it_bound = 1
            
        self.it_bound_multi = 1
        
    def project(self, x, mask):
        
        x_proj = [self.comps[p].impute() for p in range(sum(P))]
        ind = np.where(mask == 0)
        
        x_imp = copy.deepcopy(x)
        
        for i in range(len(ind[0])):
            x_imp[ind[1][i]][ind[0][i]] = x_proj[ind[1][i]][ind[0][i]]

        return x_imp
        
    def e_step(self, x, y = None):
        

        for _ in range(self.it_bound_multi):                
            [self.comps[p].e_step_inf(x[p], y) for p in range(sum(self.P))]
            Inf_mu = sum([self.comps[p].Inf_mu_out  for p in range(sum(self.P))]) + self.comps[0].mu_prior @ np.linalg.inv(self.comps[0].S_prior)
            Inf_S = sum([self.comps[p].Inf_S_out for p in range(sum(self.P))]) + np.linalg.inv(self.comps[0].S_prior)
            
            self.S_z =  np.linalg.inv(Inf_S)
            self.mu_z = np.einsum('ij,ijk->ik', Inf_mu , self.S_z)
            
            for p in range(sum(self.P)):
                self.comps[p].S_z = self.S_z
                self.comps[p].mu_z = self.mu_z
                
            conv = 0
            if self.P[0] > 0:
                Phi_old = np.hstack([self.comps[p].Phi for p in range(self.P[0])])
                for p in range(self.P[0]):
                    self.comps[p].Phi = self.mu_z @ self.comps[p].W.T
                    
                Phi_new = np.hstack([self.comps[p].Phi for p in range(self.P[0])])
                
                conv = np.sum((Phi_old - Phi_new)**2) / (Phi_old.shape[0] * Phi_old.shape[1])
            #print(conv)
            if conv < 1e-5:
                #print('Converged')
                break
            
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
        
        [self.comps[p].init_with_data(x[p], y) for p in range(sum(self.P))]
        
        for e in range(1, epoch):
            
            if mask is None:
                x_imp = x
            else:
                x_imp = self.project(x, mask)
            
            self.e_step(x_imp, y)
            
            if e % step_comp_elbo == 0:
                elbo = sum([self.comps[p].elbo_multi(x_imp[p], y) for p in range(sum(self.P))])
                elbo -= (self.comps[-1].ll_lv) * (sum(self.P)-1)
                print(elbo - elbo_old)
                elbo_old = elbo
                elbo_vec.append(elbo)
            
            [self.comps[p].m_step(x_imp[p], y) for p in range(sum(self.P))]
            
            self.elbo_vec = elbo_vec
            
        self.compute_induced_cov()
        
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

    
    P = [3, 0]
    M = [1500, 1500, 1500]
    K = 3  # Latent space dimension
    N = 4 # Number of instances
    D = 0 # Dimension of the covariates
    
    C_max = 600000 # Total number of counts - 1 (if 0 -> Categorical)
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
    z = np.array([np.random.multivariate_normal(mu0[n], np.eye(K)) for n in range(N)])
    X = []
    W = []
    b = []
    for i in range(P[0]):
        if full_dim:
            W.append(np.random.multivariate_normal(np.zeros(K), np.eye(K), size = M[i]+1))
            b.append(np.zeros(M[i]+1))
        else:
            W.append(np.random.multivariate_normal(np.zeros(K), np.eye(K), size = M[i]))
            b.append(np.zeros(M[i]))
        #b = np.random.multivariate_normal(np.zeros(M), np.eye(M))
        mean = z @ W[-1].T + b[-1]
        if not full_dim:
            mean = np.append(mean, np.zeros((mean.shape[0],1)), axis = 1)
        prob = softmax(mean, axis = 1)
        C = np.random.poisson(C_max,N) + 1 
        Xtmp = np.array([np.random.multinomial(C[n], prob[n])[:-1] for n in range(N)])
        X.append(np.append(Xtmp, C[None].T, axis = 1))  #Last column is total number of counts
    
    
    
    # Fit
    model = multimodal_gen(P, M, K, D = D, full_dim = full_dim)
    model.fit(X, y, mask = None, epoch = 500000, step_comp_elbo = 15000)
    plt.plot(model.elbo_vec)
    
    
    
    # Predictions
    Cov_pred = np.vstack([model.comps[p].W for p in range(sum(P))]) @ np.vstack([model.comps[p].W for p in range(sum(P))]).T# + model.Ai
    Cov_gt = np.vstack(W) @ np.vstack(W).T
    if D > 0:
        mean_pred = [softmax(model.comps[p].mu_prior @ model.comps[p].W.T, axis = 1) for p in range(sum(P))]
        mean_gt = [softmax((y @ V.T + a) @ W[p].T + b[p], axis = 1) for p in range(sum(P))]
        
    else:    
        mean_pred = [softmax(model.comps[p].W @ model.comps[p].mu_prior) for p in range(sum(P))]
        mean_gt = [softmax(W[p] @ a + b[p]) for p in range(sum(P))]
        
        
        
    print('Hel-mean:', np.mean([np.mean(norm(np.sqrt(mean_pred[p]) - np.sqrt(mean_gt[p]), axis = -1) / np.sqrt(2))for p in range(sum(P))])) 
    print('MSE-cov:', np.mean((Cov_pred - Cov_gt)**2))
    
    plt.imshow(Cov_gt)
    plt.show()
    plt.imshow(Cov_pred)
    plt.show()