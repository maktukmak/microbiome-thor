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

from multLDS import multLDS
from LDS import LDS
from data_gen import data_gen
import copy
import pickle

from utils import performance


'''
To-do:
    - Elbo when alpha
    - Convergence criteria
    - Covariates
    - (done) Posterior predictive
    - (done) Missing data
    - (done) Add FA
    - (done) Joint training
    - (done) Joint elbo
'''

class multimodal_LDS():
    
    def __init__(self, P, M, K = 5, step_comp_elbo = 50, D = 5, full_dim = False, dynamic = False, alpha = None):
        
        self.M = M # Observation dimension
        self.K = K # Latent space dimension
        self.P = P # Number of modalities (P[0] -> Discrete, P[1] -> Gaussian)
        self.D = D # Covariate dimension
        
        
        self.covariates = False
        if D > 0:
            self.covariates = True
        self.dynamic = dynamic
        
        self.comps = [multLDS(M[p], K, D, dynamic) for p in range(P[0])]
        self.comps += [LDS(M[p + P[0]], K, D, dynamic) for p in range(P[1])]
        for p in range(P[0]):
            self.comps[p].it_bound = 1
            
        self.alpha = [1] * sum(P)
        if alpha:
            self.alpha = alpha
            
            
        self.it_bound_multi = 10
        self.step_comp_elbo = step_comp_elbo
        
    
    def e_step(self, x, y = None, backward = True):
        
        I = len(x[0])
        
        self.m_v = []
        self.S_v = []
        self.cov_v = []
        self.m_t_vec = []
        self.S_t_vec = []
        for i in range(I):
            for _ in range(self.it_bound_multi):
                
                [self.comps[p].e_step_inf(x[p][i], self.comps[p].Phi[i], y) for p in range(self.P[0])]
                [self.comps[p + self.P[0]].e_step_inf(x[p + self.P[0]][i], y) for p in range(self.P[1])]
                
                Inf_S = sum([self.alpha[p] * self.comps[p].Inf_S_out for p in range(sum(self.P))])
                Inf_m = sum([self.alpha[p] * self.comps[p].Inf_mu_out for p in range(sum(self.P))])
                
                yin = None
                if y:
                    yin = y[i]
                m, S, cov, m_t, S_t = self.comps[0].e_step(x[0][i], yin, Inf_m, Inf_S, backward)
                
                if self.P[0] == 0:
                    break
                
                Phi_old = np.hstack([self.comps[p].Phi[i] for p in range(self.P[0])])
                for p in range(self.P[0]):
                    self.comps[p].Phi[i] = m @ self.comps[p].C + self.comps[p].c
                Phi_new = np.hstack([self.comps[p].Phi[i] for p in range(self.P[0])])
                
                conv = np.sum((Phi_old - Phi_new)**2) / (Phi_old.shape[0] * Phi_old.shape[1])
                #print(conv)
                if conv < 1e-5:
                    #print('Converged')
                    break
            self.m_v.append(m)
            self.S_v.append(S)
            self.cov_v.append(cov)
            self.m_t_vec.append(m_t)
            self.S_t_vec.append(S_t)
            
        for p in range(sum(self.P)):
            self.comps[p].m_v = self.m_v
            self.comps[p].S_v = self.S_v
            self.comps[p].cov_v = self.cov_v
            self.comps[p].m_t_vec = self.m_t_vec
            self.comps[p].S_t_vec = self.S_t_vec
            
            
    
    def fit(self, x, y = None, epoch = 500, step_comp_elbo = 1):
        
        
        [self.comps[p].init_with_data(x[p], y) for p in range(sum(self.P))]
        self.m_v = self.comps[0].m_v
        self.S_v = self.comps[0].S_v
        
        elbo_vec = [sum([self.alpha[p] * self.comps[p].elbo_multi(self.m_v, self.S_v, x[p], y) for p in range(sum(self.P))]) - (self.comps[-1].ll_lv) * (sum(self.alpha)-1)]
        
        
        for e in range(epoch):
            
            self.e_step(x, y)
            
            if e % step_comp_elbo == 0:
                
                elbo = sum([self.alpha[p] * self.comps[p].elbo_multi(self.m_v, self.S_v, x[p], y) for p in range(sum(self.P))]) - (self.comps[-1].ll_lv) * (sum(self.alpha)-1)
  
                print(elbo - elbo_vec[-1])
                elbo_vec.append(elbo)
            
            [self.comps[p].m_step(self.m_v, self.S_v, self.cov_v, x[p], y) for p in range(sum(self.P))]
            
            self.elbo_vec = elbo_vec
        
        
    def predict(self, x, y = None, mask = None, ahead = False):
        
        [self.comps[p].init_with_data(x[p], y) for p in range(sum(self.P))]
        
        self.e_step(x, y, backward = False)
        
        return self.impute(x, ahead)
    
    def impute(self, x, ahead = False):
        

        Xp = []
        prob = []
        for p in range(self.P[0]):
            xp_p, prob_p = self.comps[p].impute(x[p], ahead)
            Xp.append(xp_p)
            prob.append(prob_p)
            
        for p in range(self.P[1]):
            xp_p = self.comps[p + self.P[0]].impute(ahead)
            Xp.append(xp_p)
            prob.append(xp_p)
        
        return Xp, prob
        


if __name__ == "__main__":
           
    static = False # Dynamic / Static
    
    I = 5 # Number of sequences
    K = 3  # Latent space dimension
    N = 200 # Number of instances
    D = 0 # Dimension of the covariates
    mr = 0.0 # Random missing rate
    
    
    P = [1]  # of multinomial modalities
    M = [1] * P[0] # Observation dimension
    C_max = [0] * P[0] # Total number of counts - 1 (if 0 -> Categorical)
    
    P += [1]  # of gaussian modalities
    M = np.append(M, [20] * P[1]) # Observation dimension
    
    alpha = [1] * P[0] + [1] * P[0]
    
    data = data_gen(K, P, M, C_max, mr, D, static)
    X, y, mean = data.generate_data(N, I)
    mask = data.generate_mask(X)
    X, y, mean, Xte, yte, meante = data.train_test_split(X, y, mean)
    

    # Fit
    model = multimodal_LDS(P, M, K, D = D, dynamic = not static, alpha = alpha)
    model.fit(X, y, epoch = 50, step_comp_elbo = 1)
    plt.plot(model.elbo_vec)
    
    
    # Predictions
    Xp, prob_est = model.predict(X, y, mask, ahead = True)
    print('Hellinger results:')
    for p in range(P[0]):
        d_hel = np.mean(norm(np.sqrt(prob_est[p]) - np.sqrt(mean[p]), axis = -1) / np.sqrt(2))
        print(d_hel)
        
    print('MSE results:')
    for p in range(P[1]):
        mse = np.mean([np.mean(np.sum((Xp[p + P[0]][i]- X[p + P[0]][i])**2, axis = 1), axis = -1) for i in range(I)])
        print(mse)
    
    
    if False: # Classification on test set
    
        for i in range(I):
            maskte[i][:, 0] = 0
        Xp_te, prob_est_te = model.predict(Xte, yte)
        
        print('Model acc-hss2:', performance(np.vstack(Xp_te[0])[:, 0, None], np.vstack(Xte[0])[:, 0, None], regression = False))
        
        yp = LogisticRegression().fit(np.vstack(X[1]), np.vstack(X[0])[:,0]).predict(np.vstack(Xte[1]))
        print('LR acc-hss2:', performance(yp[None].T, np.vstack(Xte[0])[:, 0, None], regression = False))
        
        