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

'''
To-do:
    - A inversion
    - Impute extra 0
    - Full covariance
    
    done - Full dimension
    
    done - Add covariates
    done - Categorical problem
    done - Elbo
    
    
'''

class multi_res_mult():
    
    def __init__(self, M, K = 5, covariates = False, D = 5, full_dim = False):
        
        self.M = M # Observation dimension
        self.K = K # Latent space dimension
        self.D = D # Covariate dimension
        self.covariates = covariates
        self.full_dim = full_dim
        
        if full_dim:
            self.W = np.random.normal(size = (M+1) * K).reshape(M+1, K)
            self.A = self.A_m(self.M + 1)
        else:
            self.W = np.random.normal(size = M * K).reshape(M, K)
            self.A = self.A_m(self.M)
        self.Ai = np.linalg.inv(self.A)
        
        #self.mu_prior = np.random.normal(size = K)
        self.mu_prior = np.zeros(K)
        if covariates:
            self.V = np.random.normal(size = D * K).reshape(K, D)
            
        self.S_prior = np.eye(K)
        self.Inf_S_out = 0
        self.Inf_mu_out = 0
        
        self.it_bound = 10
            
    def init_with_data(self, x, y = None):
        
        if self.covariates:
            self.mu_prior = y @ self.V.T
            self.mu_z = self.mu_prior
        else:
            self.mu_z = np.tile(self.mu_prior, (x.shape[0], 1))

        self.S_z = [np.eye(self.K) for i in range(x.shape[0])]
        
        #if self.covariates:
        self.Phi = self.mu_z @ self.W.T
        #else:
        #    self.Phi = np.tile(self.mu_z @ self.W.T, (x.shape[0], 1))

    def logdet(self, X):
        return np.linalg.slogdet(X)[0] * np.linalg.slogdet(X)[1]

    def A_m(self, M):
        return  0.5*(np.eye(M) - (1/(M+1))*np.ones((M, M)))
    
    def gauss_entropy(self, Sigma):
        
        M = Sigma.shape[-1]

        return np.sum(0.5 * self.logdet(Sigma) + 0.5 * M * np.log(2 * np.pi))
        
    def gauss_loglik(self, x, mu, Lambda, x_S = None):
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
            
        return ll
    
    def multi_loglik_bounded(self, x, Th, Phi, Th_S = None):
        '''
        If Th_s -> Th integrated out
        '''
        A = self.A
        if self.full_dim:
            Phi_s = softmax(Phi, axis = 1)
            Phi_l =  logsumexp(Phi, axis = 1)
            b = Phi @ A - Phi_s
            c = 0.5 * np.diag(Phi @ A @ Phi.T) - np.diag(Phi_s @ Phi.T) +  Phi_l
            Ni = x[:, -1]
            xi = np.append(x[:, :-1], (Ni - np.sum(x[:, :-1], axis = 1))[None].T, axis = 1)           
        else:
            Phi_ext = np.append(Phi, np.zeros((Phi.shape[0],1)), axis = 1)
            Phi_s = softmax(Phi_ext, axis = 1)
            Phi_l =  logsumexp(Phi_ext, axis = 1)
            b = Phi @ A - Phi_s[:, :-1]
            c = 0.5 * np.diag(Phi @ A @ Phi.T) - np.diag(Phi_s[:, :-1] @ Phi.T) +  Phi_l
            Ni = x[:, -1]
            xi = x[:, :-1]
        
        ll = np.sum(xi * Th) 
        ll += np.sum(Ni * np.diag(- 0.5 * Th @ A @ Th.T))
        ll += np.sum(Ni * np.diag(b @ Th.T))
        ll += np.sum(-Ni * c)
        ll += np.sum(loggamma(Ni + 1)) - np.sum(loggamma(xi +1)) - np.sum(loggamma(1 + Ni - np.sum(xi, axis = 1)))

        if Th_S is not None:
            ll += np.sum(Ni * (- 0.5 * np.trace(A @ Th_S, axis1=1, axis2=2))[None])
        
        return ll
    
    
    def elbo_multi(self, x, y = None):
        
        if self.covariates:
            self.mu_prior = y @ self.V.T

        # Observations
        mu_Th = self.mu_z @ self.W.T
        Sigma_Th = self.W @ self.S_z @ self.W.T
        
        self.ll_obs = self.multi_loglik_bounded(x, mu_Th, self.Phi, Sigma_Th)
        
        self.ll_lv = self.gauss_loglik(self.mu_z, self.mu_prior, np.linalg.inv(self.S_prior), self.S_z)
        self.ll_lv += self.gauss_entropy(self.S_z)
        
        # Entropies
        elbo = self.ll_obs + self.ll_lv
                
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
        
        if self.covariates:
            self.mu_prior = y @ self.V.T
        
        for _ in range(self.it_bound):
        
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
        
        
        self.W = ( ((xi  + Ni * b) @ Ai).T @  self.mu_z) @ np.linalg.inv((Ni * self.mu_z).T @ self.mu_z + np.sum(np.expand_dims(Ni,-1) * self.S_z,axis = 0))
        
        if self.covariates:
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
            
            self.e_step(x, y)
            
            if e % step_comp_elbo == 0:
                elbo = self.elbo_multi(x, y)
                print(elbo - elbo_old)
                elbo_old = elbo
                self.elbo_vec.append(elbo)
                self.elbo_vec = self.elbo_vec
            
            self.m_step(x, y)
        
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
    

    R = 3 # Number of resolutions
    factor = 2
    K = 3  # Latent space dimension
    N = 200 # Number of instances
    M = 20
    C = 100
    lmd = 0.0000001

    z = np.random.multivariate_normal(np.zeros(K), np.eye(K), size = N)
    W = []
    m = []
    W.append(np.random.multivariate_normal(np.zeros(K), np.eye(K), size = M).T)
    A_vec = []
    Mtmp = M
    for r in range(1,R):
        A_r = np.eye(Mtmp)
        A_r = np.vstack([np.array([r]*2) for r in A_r]).T
        A_r = A_r / A_r.sum(axis = 1)[None].T
        A_vec.append(A_r)
        W.append(np.vstack([np.random.multivariate_normal(W[-1] @ A_r[:,j], lmd*np.eye(K)) for j in range(Mtmp*factor)]).T)
        Mtmp *= 2
        
    X = []
    p_vec = []
    m_vec = []
    for r in range(0,R):
        m = z @ W[r]
        m_vec.append(m)
        #m = np.concatenate((m, np.zeros((m.shape[0],1))), axis = 1)
        p = softmax(m, axis = 1)
        p_vec.append(p)
        X.append(np.vstack([np.random.multinomial(C, p[i]) for i in range(N)]))
        
        
        

    
    model = multi_res_mult(M = int(M[0]), K = K, D = D, covariates = (D>0))
    
    
    
    

    
    #model = multLDS(M = M, K = K, D = D, covariates = covariates)
    model.fit(X[0][0], y)
    Xp, prob_est = model.predict(X[0][0])
    
    print('Hellinger results:')
    d_hel = np.mean(norm(np.sqrt(prob_est) - np.sqrt(mean[0][0]), axis = 1) / np.sqrt(2))
    print(d_hel)

    plt.plot(model.elbo_vec)
            
    #print('Uncertainty amount:', np.mean(np.abs(model.S_z)))
            