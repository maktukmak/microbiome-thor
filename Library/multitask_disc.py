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

from glm import glm

'''
To-do:
    - Multi vs Single
'''

class multitask_disc():
    
    def __init__(self, P, M, D, obs, sp2, ss2):
        
        self.M = M # Observation dimension
        self.D = D # Latent space dimension
        self.P = P # Number of tasks 
        
        self.Wp = np.random.multivariate_normal(np.zeros(D+1), np.eye(D+1), M)
        self.comps = [glm(M, D, obs, reg = sp2[p], Wreg = None) for p in range(P)]
        
        self.pars = self.Wp.flatten()
        for p in range(P):
            self.pars = np.append(self.pars , self.comps[p].W)
        
        self.ss2 = ss2
        self.sp2 = sp2
            
        self.test_vec = []
        
    def nll(self, pars, x, y):
        
        Wp = pars[0:(self.D+1)*(self.M)].reshape((self.M, self.D+1))
        W  = [pars[(p+1)*(self.D+1)*(self.M):(p+2)*(self.D+1)*(self.M)].reshape((self.M, self.D+1)) for p in range(P)]
        
        r = [np.ones(len(x[p]))  for p in range(P)]
        nll = [self.comps[p].nll(W[p].flatten(), x[p], y[p], r[p], Wp) for p in range(P)]
        
        nll = np.sum(nll) + np.sum(Wp[:, :-1])**2 / (2 * self.ss2)
        
        self.test_vec.append(nll)
        
        return nll
    
    def der(self, pars, x, y):
        
        Wp = pars[0:(self.D+1)*(self.M)].reshape((self.M, self.D+1))
        W  = [pars[(p+1)*(self.D+1)*(self.M):(p+2)*(self.D+1)*(self.M)].reshape((self.M, self.D+1)) for p in range(P)]
        
        r = [np.ones(len(x[p]))  for p in range(P)]
        
        der = ((sum([(Wp - W[p]) / self.sp2[p] for p in range(P)]) + Wp / self.ss2).flatten())
        der = np.append(der, [self.comps[p].der(W[p], x[p], y[p], r[p], Wp) for p in range(P)])
        
        return np.array(der).flatten()
        
    def fit(self, x, y, niter = None):
        
        x = [np.append(x[p], np.ones((x[p].shape[0],1)), axis = 1) for p in range(P)]
        
        if niter:
            options={'maxiter': niter, 'disp': False}
        else:
            options = None
            
        opt = minimize(self.nll, 
                       self.pars, 
                       args = (x, y), 
                       jac = self.der,  
                       #hess = self.hess,
                       method='L-BFGS-B', # 'Newton-CG', 'L-BFGS-B'
                       options=options)
        
        
        self.Wp = opt['x'][0:(self.D+1)*(self.M)].reshape((self.M, self.D+1))
        self.W = [opt['x'][(p+1)*(self.D+1)*(self.M):(p+2)*(self.D+1)*(self.M)].reshape((self.M, self.D+1)) for p in range(P)]
        
        self.pars = self.Wp.flatten()
        for p in range(P):
            self.pars = np.append(self.pars , self.comps[p].W)
            self.comps[p].W = self.W[p]
        
        
    def predict(self, x):
        
        yp = [self.comps[p].predict(x[p])[0] for p in range(self.P)]
        
        return yp 
        
    


if __name__ == "__main__":
           
    
    N = [1000, 1000]
    D = 5
    M = 4
    P = 2
    
    ss2 = 1
    s2 = 1 * np.random.gamma(1, 1, size = P)
    obs = 'gauss'
    sp2 = 0.1 * np.random.gamma(1, 1, size = P)
    
    wp = np.random.multivariate_normal(np.zeros(D+1), ss2 * np.eye(D+1), M)
    w = [np.array([np.random.multivariate_normal(wp[m], sp2[p] * np.eye(D+1)) for m in range(M)]) for p in range(P)]
    
    x = [np.random.multivariate_normal(np.zeros(D), np.eye(D), N[p]) for p in range(P)]
    x = [np.append(x[p], np.ones((x[p].shape[0],1)), axis = 1) for p in range(P)]

    if obs == 'gauss':
        y = [np.random.normal(x[p] @ w[p].T, np.sqrt(s2[p])) for p in range(P)]
    elif obs == 'cat':
        Th = [x[p] @ w[p].T for p in range(P)]
        Th_ext = [np.append(Th[p], np.zeros((Th[p].shape[0],1)), axis = 1) for p in range(P)]
        prob = [softmax(Th_ext[p], axis = 1) for p in range(P)]
        y = [np.array([np.random.multinomial(1, prob[p][i])[:-1] for i in range(N[p])]) for p in range(P)]
    
    x = [x[p][:, :-1]  for p in range(P)]
    
    model_mt = multitask_disc(P, M, D, obs, sp2, ss2)
    start = time.time()
    model_mt.fit(x, y)
    end = time.time()
    print('Fit time:', end- start)
    y_pred = model_mt.predict(x)
    print('Results:')
    [print('MSE-multi:', np.mean(np.sum((y_pred[p] - y[p])**2, axis = 1))) for p in range(P)]
    
    for p in range(P):
        model = glm(M = M, D = D, obs = obs)
        model.fit(x[p], y[p])
        y_pred,_ = model.predict(x[p])
        print('MSE-single:', np.mean(np.sum((y_pred - y[p])**2, axis = 1)))
        