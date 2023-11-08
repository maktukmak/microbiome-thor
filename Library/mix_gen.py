import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from glm import glm
import matplotlib.pyplot as plt
from scipy.special import expit, logsumexp, softmax
import time
from FA import FA
from utils import gauss_loglik
'''
To-do


- ELBO implement


'''

class mix_gen():
    def __init__(self, K, D, L):
        
        self.K = K
        self.D = D
        
        self.comps = [FA(D, L) for k in range(K)]
                
    def compute_ll(self, model, x):
        
        ll = gauss_loglik(x[i], model.mu, model.Prec_x, mu_S = mu_S)
            
        return model.elbo_multi(x)
        
    def fit(self, x):
        
        [self.comps[k].init_with_data(x) for k in range(self.K)]
        
        Q = -1e8
        cnv_vec = []
        Q_vec = []
        for it in range(1000):
            
            # E-step
            
            r = [self.comps[k].elbo_multi(x) for k in range(self.K)]
            
            
            pi = self.gate.predict(t)[1]
            if self.expert_type =='glm':
                ll = np.array([self.experts[k].ll(self.experts[k].pars, x, y, np.zeros(len(x))) for k in range(self.K)]).T
            elif self.expert_type == 'sklearn':
                ll =  np.array([ self.compute_ll(self.experts[k], x, y, self.s2_experts[k])  for k in range(self.K)]).T
            r = softmax(ll + np.log(pi), axis = 1)

            # Compute lower bound
            Q_old = Q
            #Q = np.sum(logsumexp(ll + np.log(pi), axis = 1))
            Q = np.sum(r * (ll + np.log(pi))) 
            #Q -= np.sum(r * np.log(r))
            #Q += np.sum(self.gate.W ** 2) + sum([np.sum(self.experts[k].W ** 2) for k in range(self.K)])
            Q_vec.append(Q)
            
            # M-step
            self.gate.fit(t, r[:, :-1], niter = 1)
            if self.expert_type =='glm':
                [self.experts[k].fit(x, y, r[:,k], niter = 1) for k in range(self.K)]
            elif self.expert_type == 'sklearn':
                if self.obs == 'gauss':
                    [self.experts[k].fit(x, y, r[:,k]) for k in range(self.K)]
                    self.s2_experts = [np.sum(r[:, k] * np.sum((y - self.experts[k].predict(x))**2, axis = 1)) / sum(r[:, k])   for k in range(self.K)]
                elif self.obs == 'cat':
                    y_int = np.argmax(np.concatenate((y, 1 - y.sum(axis = 1)[None].T), axis = 1), axis = 1)
                    [self.experts[k].fit(x, y_int, r[:,k]) for k in range(self.K)]
            
            if Q - Q_old < 0:
                print('EM warning')
            
            cnv_vec.append(np.sum(abs(Q_old - Q)))
            #print(cnv_vec[-1])
            #print(Q)
            #print(it)
            if it > 20 and cnv_vec[-1] < 1e-4:
                return cnv_vec, Q_vec
            
        print('Not converged')
        return cnv_vec, Q_vec
        
            
    def predict(self, x, t):
        
        pi = self.gate.predict(t)[1]
        
        if self.obs == 'cat':
            
            ind = np.argmax(pi, axis = 1)
            if self.expert_type == 'glm': 
                sc =np.array([self.experts[k].predict(x)[0] for k in range(self.K)])
            elif self.expert_type == 'sklearn':
                sc =np.array([np.eye(self.M + 1)[self.experts[k].predict(x)][:,:-1] for k in range(self.K)])

            yp = np.array([sc[ind[i], i] for i in range(len(x))])
            mu = yp

        elif self.obs == 'gauss':
            if self.expert_type == 'glm': 
                sc =np.array([self.experts[k].predict(x)[0]  for k in range(self.K)])
            elif self.expert_type == 'sklearn':
                sc =np.array([self.experts[k].predict(x)  for k in range(self.K)])   
            
            
            ind = np.argmax(pi, axis = 1)
            mu = np.array([sc[ind[i], i] for i in range(len(x))])
            yp = mu
            
        return yp, mu
            

if __name__ == "__main__":
    
    K = 3
    L = 2
    D = 2
    I = 300
    
    #pi = np.random.dirichlet(np.ones(K))
    pi = np.ones(K) / K
    q = np.random.multinomial(1, pi, size = I)
    z = np.random.multivariate_normal(np.zeros(L), np.eye(L), size = I)
    
    mu = np.random.multivariate_normal(np.zeros(D), np.eye(D), size = K)
    W = [np.random.multivariate_normal(np.zeros(D), np.eye(D), size = L).T for k in range(K)]
    S = 0.1 * np.eye(D)
    X = np.array([np.random.multivariate_normal(mu[np.argmax(q[i])] + W[np.argmax(q[i])] @ z[i],  S) for i in range(I)])
    plt.scatter(X[:,0], X[:,1], c=q)
    plt.show()
    
    model = mix_gen(K, D, L)
    model.fit(X)