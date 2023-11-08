import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from glm import glm
import matplotlib.pyplot as plt
from scipy.special import expit, logsumexp, softmax
import time

'''
To-do


- ELBO implement

done - Regularization strength
done - Expert multivariate (Sklearn)
done - Categorical test

'''

class moe_std():
    def __init__(self, K, Dx, M,  sk = 1, ss = 1, obs = 'cat'):
        
        
        '''
        init:
            
        K:      Number of experts
        Dx:     Dimension of expert inputs
        Dt:     Dimension of gate input
        M:      Output dimension
        sk:     Regularization strength for experts (sigma^2)
        ss:     Regularization strength for gating module (sigma^2)       
        obs:    Obervation distribution ( gauss/cat )
        
        fit:
            
        x:      Expert input (N x Dx)
        y:      Response variable (N x M) (if obs is cat, then y is one-hot encoded with last dimension excluded, i.e., M = C-1)

        '''
        
        
        self.obs = obs
        
        self.K = K
        self.Dx = Dx
        self.Dt = Dx
        self.M = M
        
        # Configurations
        self.expert_type = 'sklearn' # 'glm', 'sklearn'
        self.sk = sk  # Regularization strength of experts
        self.ss = ss # Regularization strength of gating
        
        
        # Init components
        self.init_params()
        
    def init_params(self):
        self.gate = glm(M = self.K-1, D = self.Dx, obs = 'cat', reg = self.ss)
        if self.expert_type == 'glm':
            self.experts = [glm(M = self.M, D = self.Dx, obs = self.obs, reg = self.sk) for k in range(self.K)]
        elif self.expert_type == 'sklearn':
            if self.obs == 'gauss':
                self.experts = [Ridge(alpha = self.sk) for k in range(K)]
                x = np.random.normal(size=100 * self.Dx).reshape((100, Dx))
                y = np.random.normal(size=100 * self.M).reshape((100, self.M))
                [self.experts[k].fit(x, y) for k in range(self.K)]
            elif self.obs == 'cat':
                self.experts = [LogisticRegression(C = self.sk) for k in range(self.K)]
                x = np.random.normal(size=100 * self.Dx).reshape((100, self.Dx))
                y = np.random.choice(self.M+1, size=100)
                [self.experts[k].fit(x, y) for k in range(self.K)]
                
            self.s2_experts = [1] * self.K
                
    def get_params(self, deep=True):
        
        return {"K": self.K, "Dx": self.Dx, "M": self.M}    
    
    def compute_ll(self, model, x, y, s):
        
        if self.obs == 'cat':
            Th = x @ model.coef_.T + model.intercept_
            
            if self.M == 1:
                Th_ext = np.append(Th, np.zeros((Th.shape[0],1)), axis = 1)
                A = logsumexp(Th_ext, axis = 1)
                ll = (np.diag((1-y) @ Th.T) - A)
            else:
                A = logsumexp(Th, axis = 1)
                ll = (np.diag(np.concatenate((y, 1 - y.sum(axis = 1)[None].T), axis = 1) @ Th.T) - A)
            
            #ll = np.sum(np.concatenate((1 - y.sum(axis = 1)[None].T, y), axis = 1) * model.predict_log_proba(x), axis = 1)
        elif self.obs == 'gauss':
            ll = -np.sum(((y - model.predict(x))**2) / (2*s) + 0.5 * np.log(2*np.pi) + 0.5 * np.log(s) , axis = 1)
            
        return ll
        
    def fit(self, x, y):
        
        t = x
        
        Q = -1e8
        self.cnv_vec = []
        Q_vec = []
        for it in range(100):
            
            # E-step
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
            
            #if Q - Q_old < 0:
            #    print('EM warning')
            
            self.cnv_vec.append(np.sum(abs(Q_old - Q)))
            #print(cnv_vec[-1])
            #print(Q)
            #print(it)
            if it > 20 and self.cnv_vec[-1] < 1e-4:
                return self.cnv_vec, Q_vec
            
        print('Not converged')
        return self.cnv_vec, Q_vec
        
            
    def predict(self, x):
        
        t = x
        
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
            
        return yp
            

if __name__ == "__main__":


    if False: # Regression test    
        
        multivariate = False
    
        n = 200;
        y = np.random.uniform(size = n)
        eta = np.random.normal(size = n) * 0.05
        
        x = y + 0.3*np.sin(2*np.pi*y) + eta
        Dx = 1
        Dt = 1
        
        
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        x = x[:,None]
        y = y[:,None]
        
        M = 1
        if multivariate:
            M = 2
            y = np.append(y, y, axis = 1)
        
        model_moe = moe_std(K = 3,
                        Dx = Dx,
                        Dt = Dt,
                        M = M,
                        sk = 1, ss = 1,
                        obs = 'gauss')
        
        start = time.time()
        cnv_vec, q_vec = model_moe.fit(x, y)
        end = time.time()
        print('Fit time:', end- start)
        
        yp = model_moe.predict(x)
        
        plt.plot(q_vec)
        plt.show()
        
        plt.scatter(x, y[:,0])
        t = np.arange(min(x), max(x), 1e-2)[None].T
        resp = model_moe.gate.predict(t)[1]
        plt.plot(t, resp[:,0])
        plt.plot(t, resp[:,1])
        plt.plot(t, resp[:,2])
        
        plt.scatter(x, yp[:,0])
        plt.show()
        
    
    if False: # Categorical test    
            
        multiclass = True
        
        n = 200
        Dx = 2
        Dt = 2
        
        M = 1
        if multiclass:
            M = 2
            
        K = M + 1
        
        m1 = [[0, -2], [0, 2]]
        s1 = 0.5
        
        m2 = [[-2, 0], [2, 0]]
        s2 = 0.5
        
        m3 = [[-3, 3], [3, -3]]
        s3 = 0.5
        
        x = np.array([np.random.multivariate_normal(m1[np.random.choice(2)], s1 * np.eye(2)) for i in range(n)])
        y = np.array([0]*n)
        x = np.append(x, np.array([np.random.multivariate_normal(m2[np.random.choice(2)], s2 * np.eye(2)) for i in range(n)]), axis = 0)
        y = np.append(y, np.array([1]*n))
        if multiclass:
            x = np.append(x, np.array([np.random.multivariate_normal(m3[np.random.choice(2)], s3 * np.eye(2)) for i in range(n)]), axis = 0)
            y = np.append(y, np.array([2]*n))
        
        y = np.eye(M + 1)[y][:,:-1]
        
        model_moe = moe_std(K = K,
                        Dx = Dx,
                        #Dt = Dt,
                        M = M,
                        sk = 1, ss = 1,
                        obs = 'cat')
        
        start = time.time()
        cnv_vec, q_vec = model_moe.fit(x, y)
        end = time.time()
        print('Fit time:', end- start)
        plt.plot(q_vec)
        plt.show()
        
        yp = model_moe.predict(x)
        print('Acc:', np.sum(np.all(yp == y, axis = 1)) / (n*(M+1)))
        
        yp = np.concatenate((yp, 1 - yp.sum(axis = 1)[None].T), axis = 1)
        plt.scatter(x[:,0], x[:,1], c = np.argmax(yp, axis = 1))
