# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:21:21 2019

@author: Mehmet
"""

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
from scipy.linalg import cholesky

import time

from scipy.stats import invwishart, invgamma


'''
To-do:
   - Positive definite problem
   - Convergence break
    
'''

class MVN():
    
    def __init__(self, M):
        
        self.M = M # Observation dimension
        
        self.mu = np.zeros(M)
        self.Sigma = np.identity(M)
        self.Prec = np.identity(M)

    def init_with_data(self, x, mask):
        
        # Init
        x_imp = x.copy()
        x_imp[np.where(mask == 0)] = np.random.randn(len(np.where(mask == 0)[0]))
        
        self.mu = np.mean(x_imp, axis = 0)
        self.Sigma = np.cov(x_imp.T)
        

    def logdet(self, X):
        return np.linalg.slogdet(X)[0] * np.linalg.slogdet(X)[1]
        
    def gauss_loglik(self, x, mu, Lambda, x_S = None, mu_S = None):
        ''' 
        If x_S -> x is integrated out
        
        x -> NxD
        mu -> 1XD
        Lamda -> DXD
        '''
        
        M = x.shape[1]
        N = x.shape[0]
        
        ll = - 0.5 * np.trace((x - mu) @ Lambda @ (x - mu).T)
        ll += 0.5 * N * self.logdet(Lambda)
        ll += - 0.5 * N * M * np.log(2*np.pi)
        if x_S is not None:
            ll += - np.sum(0.5 * np.trace(Lambda @ x_S, axis1=1, axis2=2))
        if mu_S is not None:
            ll += - np.sum(0.5 * np.trace(Lambda @ mu_S, axis1=1, axis2=2))
            
        return ll
    
    def elbo_multi(self, x, mask):
        
        L = 0
        Ls = []
        for i in range(0, len(x)):
            o = np.where(mask[i] == 1)[0]
            R = cholesky(self.Sigma[np.ix_(o,o)])
            Ls.append(-(0.5 * len(o)) * np.log(2*np.pi) - 0.5 * (self.logdet(self.Sigma[np.ix_(o,o)])) - 0.5 * np.sum((x[i, o] - self.mu[o]).T * (np.linalg.inv(R) @ (x[i, o] - self.mu[o]).T)))
            L += Ls[-1]
                
        return L, np.array(Ls)
    
    def e_step(self, x, mask):
        
        self.x_imp = np.zeros(x.shape)
        self.EXsum = np.zeros(self.M)
        self.EXXsum = np.zeros((self.M, self.M))
        for i in range(0, len(x)):
            
            EX = np.zeros(self.M)
            EXX = np.zeros((self.M, self.M))
            
            o = np.where(mask[i] == 1)[0]
            u = np.where(mask[i] == 0)[0]
            Vi = self.Sigma[np.ix_(u,u)] - self.Sigma[np.ix_(u,o)] @ np.linalg.inv(self.Sigma[np.ix_(o,o)]) @ self.Sigma[np.ix_(o,u)]
            mi = self.mu[u] + self.Sigma[np.ix_(u,o)] @ np.linalg.inv(self.Sigma[np.ix_(o,o)]) @ (x[i, o] - self.mu[o])
            EX[u] = mi
            EX[o] = x[i, o]
            EXX[np.ix_(u,u)] = EX[u][None].T @ EX[u][None] + Vi
            EXX[np.ix_(o,o)] = EX[o][None].T @ EX[o][None]
            EXX[np.ix_(o,u)] = EX[o][None].T @ EX[u][None]
            EXX[np.ix_(u,o)] = EX[u][None].T @ EX[o][None]
            self.EXsum += EX
            self.EXXsum += EXX
            self.x_imp[i] = EX

    def m_step(self, x):
        
        n = len(x)
        
        self.mu = self.EXsum / n
        self.Sigma = self.EXXsum / n - self.mu[None].T @ self.mu[None]
        
                
    def fit(self, x, mask, epoch = 200):
        
        elbo_old = 0
        self.elbo_vec = []
        
        self.init_with_data(x, mask)
        cnv_vec = []
        
        for e in range(epoch):
            
            self.e_step(x, mask)
            
            elbo,_ = self.elbo_multi(x, mask)
            #print(elbo - elbo_old)
            self.m_step(x)
            
            self.elbo_vec.append(elbo)
            
            elbo_old = elbo
            
            cnv_vec.append(np.sum(abs(elbo - elbo_old)))
            if e > 3 and cnv_vec[-1] < 1e-4:
                break
            
            
        
    def predict(self, x, mask):
        
        self.e_step(x, mask)
        
        return self.x_imp
        
        
if __name__ == "__main__":
    
    
    if False: # Missing value impute
    
        K = 3
        M = 25
        N = 200
        pcMissing = 0.2
    
        Sigma = 0.1 * np.identity(M)
        z = np.random.multivariate_normal(np.zeros(K), np.eye(K), size = N)
        W = np.random.multivariate_normal(np.zeros(K), np.eye(K), size = M)
        Th = z @ W.T
        X = np.array([np.random.multivariate_normal(Th[i], Sigma) for i in range(N)])
        
        mask = np.random.uniform(0, 1, (N, M)) < pcMissing
        Xmiss = X.copy()
        Xmiss[np.where(mask == 0)] = np.nan
        
        
        model = MVN(M = M)
        model.fit(Xmiss, mask)
        X_pred = model.predict(Xmiss, mask)
        
        plt.plot(model.elbo_vec)
        
        X_pred_vec = X_pred[np.where(mask == 0)]
        Xmiss_vec = X[np.where(mask == 0)]
        
        print('MSE:', np.mean((X_pred_vec - Xmiss_vec)**2))
        
    if True: # Domain adaptation
        
        start = time.time()
    
        K = 2
        M = 2
        N = 200
        N1f = 0.6
        N2f = 0.2
        N3f = 0.1
        C = 2
        
        
        def data_gen():
            
            mu = np.random.multivariate_normal(np.zeros(K), 1 * np.eye(K))
            #Sigma = invwishart.rvs(df=K+1, scale=np.eye(K))
            Sigma = np.eye(K)
            Sigma_x = 1 * np.identity(M)
            z = np.random.multivariate_normal(mu, Sigma, size = N)
            W = np.random.multivariate_normal(np.zeros(K), np.eye(K), size = M)
            Th = z @ W.T
            X = np.array([np.random.multivariate_normal(Th[i], Sigma_x) for i in range(N)])
            return X, W @ mu, (W @ W.T + Sigma_x)
            
        
        
        
            
            
        X = [] 
        m = []
        S = []
        for c in range(C):
            Xc, mc, Sc = data_gen()
            X.append(Xc)
            m.append(mc)
            S.append(Sc)
            
        N1 = int(N*N1f)
        N2 = N1 + int(N2f * N)
        N3 = N2 + int(N3f * N)
        
        mask = np.zeros((N,M))
        mask[0:N1, 0:int(M/2)] = 1
        mask[N1:N2, :] = 1 
        mask[N2:, int(M/2):] = 1
            
            
        Xmiss = X.copy()
        for c in range(C):
            Xmiss[c][np.where(mask == 0)] = np.nan
            
        ind_tr_multi =  np.arange(0, N3)
        ind_tr_single = np.arange(N1, N3)
        ind_te = np.arange(N3, N)
        
        mask_tr_single = mask[ind_tr_single]
        mask_tr_single[:, 0:int(M/2)] = 0
        mask_tr_multi = mask[ind_tr_multi]
        mask_te = mask[ind_te]
        
        Xmiss_tr_single = [Xmiss[c][ind_tr_single] for c in range(C)]
        Xmiss_tr_multi = [Xmiss[c][ind_tr_multi] for c in range(C)]
        Xmiss_te = [Xmiss[c][ind_te] for c in range(C)]
        
        
        y_te = np.array([0]*len(ind_te) + [1]*len(ind_te))
        Xmiss_te = np.vstack(Xmiss_te)
        mask_te = np.concatenate((mask_te, mask_te), axis = 0)
        
        
        model_m = [MVN(M = M) for c in range(C)]
        [model_m[c].fit(Xmiss_tr_multi[c], mask_tr_multi, epoch = 100) for c in range(C)]
        ll = np.array([model_m[c].elbo_multi(Xmiss_te, mask_te)[1] for c in range(C)]).T
        y_pred_m = np.argmax(ll, axis = 1)
        
        model_s = [MVN(M = M) for c in range(C)]
        [model_s[c].fit(Xmiss_tr_single[c], mask_tr_single, epoch = 20) for c in range(C)]
        ll = np.array([model_s[c].elbo_multi(Xmiss_te, mask_te)[1] for c in range(C)]).T
        y_pred_s = np.argmax(ll, axis = 1)
        
        print('Acc-Single:', np.mean(y_pred_s == y_te))
        print('Acc-Multi:', np.mean(y_pred_m == y_te))
        
        
        m_pred = [model_m[c].mu for c in range(C)]
        S_pred = [model_m[c].Sigma for c in range(C)]
        print('Multi-Mean-MSE:', np.mean((m_pred[c][int(M/2):] - m[c][int(M/2):])**2))
        print('Multi-Sigma-MSE:', np.mean((S_pred[c][int(M/2):, int(M/2):] - S[c][int(M/2):, int(M/2):])**2))
        m_pred = [model_s[c].mu for c in range(C)]
        S_pred = [model_s[c].Sigma for c in range(C)]
        print('Single-Mean-MSE:', np.mean((m_pred[c][int(M/2):] - m[c][int(M/2):])**2))
        print('Single-Sigma-MSE:', np.mean((S_pred[c][int(M/2):, int(M/2):] - S[c][int(M/2):, int(M/2):])**2))
        