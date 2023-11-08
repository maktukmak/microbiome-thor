import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from scipy.special import expit, logsumexp, softmax
import time
import scipy.linalg


'''
To-do

- Multitask prior

done - Categorical numerical stability 
done - Estimate s2
done - Multivariate output
done - Weighted for MOE
done - pars variable simplify

- Prior w for Multitask

opt - Poisson regression
opt - Hessian closed form for multivariate

'''


class glm():
    def __init__(self, M, D, obs = 'gauss', reg = 1, Wreg = None):
        
        '''
        D:      Dimension of input
        M:      Output dimension
        reg:    Regularization strength     
        obs:    Obervation distribution ( gauss/cat )
        '''
        
        self.M = M  # Output dimension
        self.D = D  # Input dimension
        
        self.W = np.random.multivariate_normal(np.zeros(D+1), np.eye(D+1), M)
        self.s2 = 1 # sigma_2 for regression
        self.reg = reg # Regularization strength
        
        if Wreg is None:
            self.Wreg = np.zeros(self.W.shape)
        else:
            self.Wreg = Wreg
        
        self.pars = self.W.flatten()
        
        self.obs = obs
        self.pred_s2 = False
        if self.obs == 'gauss':
            self.pred_s2 = True
            
        self.test_vec = []
        
    def ll(self, pars, x, y, r, bias = True):
        
        if bias:
            x = np.append(x, np.ones((x.shape[0],1)), axis = 1)
        
        W = pars[0:(self.D+1)*(self.M)].reshape((self.M, self.D+1))
        s2 = self.s2
    
        if self.obs == 'gauss':
            Th = x @ W.T
            A = 0.5 * np.diag( Th @ Th.T) 
            h = - 0.5 * np.diag(y @ y.T) / (s2)  - 0.5 * self.M * np.log(s2) - 0.5 * self.M * np.log(2*np.pi)
        elif self.obs == 'cat':
            Th = x @ W.T
            Th_ext = np.append(Th, np.zeros((Th.shape[0],1)), axis = 1)
            A = logsumexp(Th_ext, axis = 1)
            h = 0
            
        ll = (np.diag(y @ Th.T) - A) / s2  + h
        
        return ll
    
    def nll(self, pars, x, y, r, Wreg):
        
        W = pars[0:(self.D+1)*(self.M)].reshape((self.M, self.D+1))
        ll = self.ll(pars, x, y, r, bias = False)
        nll = -np.sum(ll * r) + np.sum((W[:, :-1] - Wreg[:, :-1])**2) / (2 * self.reg)
        
        self.test_vec.append(nll)
        
        return nll
        
    def der(self, pars, x, y, r, Wreg):
        
        W = pars[0:(self.D+1)*(self.M)].reshape((self.M, self.D+1))
        s2 = self.s2
        
        if self.obs == 'gauss':
            mu = x @ W.T
        elif self.obs == 'cat':
            Th = x @ W.T
            Th_ext = np.append(Th, np.zeros((Th.shape[0],1)), axis = 1)
            mu = softmax(Th_ext, axis = 1)[:, :-1]

        grad_th = (y - mu).T @ np.diag(r) @ x
        
        grad_th[:, :-1] += - (W[:, :-1] - Wreg[:, :-1])/self.reg
        
        grad = grad_th.flatten() / self.s2
        return -grad
    
    def hess(self, pars, x, y):
        
        W = pars[0:(self.D+1)*(self.M)].reshape((self.M, self.D+1))
        s2 = self.s2
        
        if self.obs == 'gauss':
            S = np.diag(np.ones(len(x)))
            hess_th = x.T @ S @ x
        elif self.obs == 'cat':
            Th = x @ W.T
            Th_ext = np.append(Th, np.zeros((Th.shape[0],1)), axis = 1)
            mu = softmax(Th_ext, axis = 1)[:, :-1]
            #S = np.diag((mu * (1 - mu))[:,0])
            #S = np.diag((mu).flatten()) - scipy.linalg.block_diag(*np.einsum('ij, ik->ijk',mu, mu))
            #x = np.repeat(np.repeat(x, 2, axis = 0), 2, axis = 1)
            #x = np.tile(x, (2,2))

            hess_th = sum([np.kron((x[i, None].T @ x[i, None]), (np.diag(mu[i]) - mu[i, None].T @ mu[i, None])) for i in range(len(x))])

        
        #hess_th = x.T @ S @ x
        hess = hess_th.flatten() / self.s2
        
        return hess_th
    
    def fit(self, x, y, r = None, niter = None):
        
        x = np.append(x, np.ones((x.shape[0],1)), axis = 1)
        
        if niter:
            options={'maxiter': niter, 'disp': False}
        else:
            options = None
            
        if r is None:
            r = np.ones(len(x))
            
        opt = minimize(self.nll, 
                       self.pars, 
                       args = (x, y, r, self.Wreg), 
                       jac = self.der,  
                       #hess = self.hess,
                       method='L-BFGS-B', # 'Newton-CG', 'L-BFGS-B'
                       options=options)
        
        
        self.W = opt['x'][0:(self.D+1)*(self.M)].reshape((self.M, self.D+1))
        #print(opt['success'])
        
        if self.pred_s2:
            self.s2 = np.sum(r * np.sum((x @ self.W.T - y) ** 2, axis = 1)) / sum(r)
        
        self.pars = self.W.flatten()
        
    def predict(self, x):
        
        x = np.append(x, np.ones((x.shape[0],1)), axis = 1)
        
        if self.obs == 'gauss':
            mu = x @ self.W.T
            yp = mu
        elif self.obs == 'cat':
            Th = x @ self.W.T
            Th_ext = np.append(Th, np.zeros((Th.shape[0],1)), axis = 1)
            mu = softmax(Th_ext, axis = 1)
            yp = np.eye(self.M + 1)[np.argmax(mu, axis = 1)][:, :-1]
            
        return yp, mu


if __name__ == "__main__":


    N = 100
    D = 5
    M = 4
    s2 = 0.1
    obs = 'gauss'
    
    x = np.random.multivariate_normal(np.zeros(D), np.eye(D), N)
    x = np.append(x, np.ones((x.shape[0],1)), axis = 1)
    w = np.random.multivariate_normal(np.zeros(D+1), np.eye(D+1), M)
    r = np.random.uniform(size = N)
    r = np.ones(N)
    if obs == 'gauss':
        y = np.random.normal(x @ w.T, np.sqrt(s2))
    elif obs == 'cat':
        Th = x @ w.T
        Th_ext = np.append(Th, np.zeros((Th.shape[0],1)), axis = 1)
        prob = softmax(Th_ext, axis = 1)
        y = np.array([np.random.multinomial(1, prob[i])[:-1] for i in range(N)])
    
    x = x[:, :-1]
    
    model = glm(M = M, D = D, obs = obs)
    start = time.time()
    model.fit(x, y, r)
    end = time.time()
    print('Fit time:', end- start)
    y_pred,_ = model.predict(x)
    
    
    
    if obs == 'gauss':
        print(np.sum((y - y_pred)**2) / N)
        start = time.time()
        model_lin = LinearRegression().fit(x, y, r)
        end = time.time()
        y_pred = model_lin.predict(x)
        print(np.sum((y - y_pred)**2) / N)
        print('Fit time:', end- start)
        
    elif obs == 'cat':
        print(np.sum(np.all(y == y_pred, axis = 1)) / N)
        
        y_ext = np.concatenate((y, 1-y.sum(axis = 1)[None].T), axis = 1)
        start = time.time()
        model_log = LogisticRegression().fit(x, np.argmax(y_ext, axis = 1))
        end = time.time()
        y_pred = model_log.predict(x)
        print(np.mean(np.argmax(y_ext, axis = 1) == y_pred))
        print('Fit time:', end- start)




