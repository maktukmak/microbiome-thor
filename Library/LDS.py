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

from numpy.linalg import inv as inv
from data_gen import data_gen
from utils import logdet, gauss_entropy, gauss_loglik, cov_data, mean_data, norm_cov
from scipy.stats import ortho_group
from scipy.linalg import block_diag

'''
To-do:
    
    - Put constraint on A,R
    (opt) - First input can be incorporated
    
    (opt) - Estimate a
    
    - Negative ELBO problem (When D > 0)
    
    done - Covariate
    done - Fast E-step when static
    done - Add state space bias
    done - Problem bias high mse
    
    done - Convergence
    done - Multimodal compability
    done - Multiple series
    
    
'''

class LDS():
    
    def __init__(self, M, K = 5, D = 5):
        
        self.M = M # Observation dimension
        self.K = K # Latent space dimension
        self.D = D # Covariate dimension
        

        self.C = np.random.normal(size = M * K).reshape(K, M)
        self.c = np.zeros(M)
        self.R = 1 * np.identity(M)
        
        self.v = np.zeros(K)
        self.V = np.zeros((K, D))
        if self.D > 0:
            self.V = np.random.normal(size = D * K).reshape(K, D)
            #self.V = np.zeros(D * K).reshape(K, D)
            
        self.a = np.zeros(K)
        self.A = np.random.multivariate_normal(np.zeros(K), np.eye(K), size = K)
        #self.A = np.eye(K)
        #self.A = np.zeros((K,K))
        self.svd_thresh = 0.5
            
        self.Q = np.identity(K)
        
        self.mu0 = np.zeros(K)
        self.S0 = np.eye(K)
        
        self.Inf_S_out = 0
        self.Inf_mu_out = 0
        
    def init_with_data(self, x, y = None):
        
        mu_z = [np.tile(self.v + self.a, (x[i].shape[0], 1))  for i in range(len(x))] 
        if self.D > 0:
            for i in range(len(x)):
                mu_z[i] += y[i] @ self.V.T
        
        self.m_v = mu_z
        self.S_v = [np.array([np.eye(self.K)] * (x[i].shape[0]))  for i in range(len(x))]


    def elbo_multi(self, m_v, S_v, x, y = None):
        
        mu_z = [np.tile(self.v + self.a, (x[i].shape[0], 1))  for i in range(len(x))] 
        if self.D > 0:
            for i in range(len(x)):
                mu_z[i] += y[i] @ self.V.T
            
        # Observations
        self.ll_obs = 0
        self.ll_lv = 0
        for i in range(len(x)):
            mu_z[i][0] = self.mu0
            mu_obs = m_v[i] @ self.C + self.c
            S_obs = self.C.T @ S_v[i] @ self.C
            mu_z_1 = m_v[i][:-1,:] @ self.A.T + mu_z[i][1:,:]
            S_z_1 = self.A @ S_v[i][:-1] @ self.A.T
            
            S_pair1 = np.einsum('jk,ilk->ijl', self.A, self.cov_v[i])
            S_pair2 = np.einsum('ijk,lk->ijl', self.cov_v[i], self.A) # Might be no transpose
            
            self.ll_obs += gauss_loglik(x[i], mu_obs, inv(self.R), mu_S = S_obs)
            self.ll_lv += gauss_loglik(m_v[i][1:,:], mu_z_1, inv(self.Q), S_v[i][1:], S_z_1, S_pair1, S_pair2)
            self.ll_lv += gauss_loglik(m_v[i][0:1,:], mu_z[i][0], inv(self.S0), x_S = S_v[i][0:1])
            self.ll_lv += gauss_entropy(S_v[i])
        
        elbo = self.ll_obs + self.ll_lv
                
        return elbo
    
    def e_step_seq(self, x, y = None, backward = True):
        
        self.m_v = []
        self.S_v = []
        self.cov_v = []
        self.m_t_vec = []
        self.S_t_vec = []
        for i in range(len(x)):
            yin = None
            if y:
                yin = y[i]
            self.e_step_inf(x[i])
            m, S, cov, m_t, S_t = self.e_step(x[i], yin, self.Inf_mu_out, self.Inf_S_out, backward=backward)
            self.m_v.append(m)
            self.S_v.append(S)
            self.cov_v.append(cov)
            self.m_t_vec.append(m_t)
            self.S_t_vec.append(S_t)
    
    def e_step_inf(self, x, y = None):
        
        self.Inf_S_out = np.tile(self.C @ inv(self.R) @ self.C.T, (x.shape[0], 1, 1))
        self.Inf_mu_out = ((x - self.c) / np.diag(self.R)[None]) @ self.C.T

    def e_step(self, x, y = None, Inf_mu_in = 0, Inf_S_in = 0, backward = True):
        
        T = x.shape[0]
        
        mu_z = np.tile(self.v + self.a, (x.shape[0], 1))
        if self.D > 0:
            mu_z += y @ self.V.T
        mu_z[0] = self.mu0

        # Forward_pass
        S = [inv(inv(self.S0) + Inf_S_in[0])]
        mu = [S[0] @ (inv(self.S0) @ (mu_z[0]) + Inf_mu_in[0])]
        
        mu_t_t_1_vec = [mu_z[0]]
        S_t_t_1_vec = [self.S0]
        for t in np.arange(1,T):
            
            S_t_t_1 = self.A @ S[t-1] @ self.A.T + self.Q
            mu_t_t_1 = self.A @ mu[t-1]  + mu_z[t]
            
            S.append(inv(inv(S_t_t_1) + Inf_S_in[t]))
            mu.append(S[t] @ (inv(S_t_t_1) @ mu_t_t_1 + Inf_mu_in[t]))
            
            mu_t_t_1_vec.append(mu_t_t_1)
            S_t_t_1_vec.append(S_t_t_1)
        
        # Backward pass
        pair_cov = []
        if backward:
            for t in np.arange(T-2,-1,-1):
                S_t_1_t = self.A @ S[t] @ self.A.T + self.Q
                mu_t_1_t = self.A @ mu[t] + mu_z[t+1]
                J = S[t] @ self.A.T @ inv(S_t_1_t)
                
                S[t] = S[t] + J @ ( S[t+1] - S_t_1_t) @ J.T
                mu[t] = mu[t] + J @ (mu[t+1] - mu_t_1_t)
                
                pair_cov.append( J @ S[t+1])

        S_v =  np.array(S)
        m_v = np.array(mu)
        
        return m_v, S_v, pair_cov, np.array(mu_t_t_1_vec), np.array(S_t_t_1_vec)

    def m_step(self, m_v, S_v, cov_v, x, y = None):
        
        I = len(x)
        
        # Augment
        #y_aug = []
        m_v_aug = []
        S_v_aug = []
        for i in range(I):
            #if self.D > 0:
            #    y_aug.append(np.append(y[i], np.ones((y[i].shape[0], 1)), axis = 1))
            S_v_aug.append(np.block([[m_v[i].T @ m_v[i] + np.sum(S_v[i],axis = 0), np.sum(m_v[i], axis = 0)[None].T],
                                  [np.sum(m_v[i], axis = 0)[None] , x[i].shape[0]*np.ones(1)[None]]]))
            m_v_aug.append(np.append(m_v[i], np.ones((x[i].shape[0], 1)), axis = 1))
        
        #if self.D > 0:
            # V_aug =  (sum([m_v[i].T @ y_aug[i] for i in range(I)])) @ np.linalg.inv(sum([y_aug[i].T @ y_aug[i] for i in range(I)]))
            # self.V = V_aug[:, :-1]
            # self.v = V_aug[:, -1]
        #    self.V = (sum([(m_v[i][1:] - m_v[i][:-1] @ self.A.T).T @ y[i][1:] for i in range(I)])) @ np.linalg.inv(sum([y[i][1:].T @ y[i][1:] for i in range(I)]))
        
        
        C_aug = ( (sum([x[i].T @  m_v_aug[i] for i in range(I)])) @ np.linalg.inv(np.sum(S_v_aug ,axis = 0)) )
        self.C = C_aug[:, :-1].T
        self.c = C_aug[:, -1]
        
        self.R = np.diag( (sum([np.sum(x[i] * x[i], axis = 0) for i in range(I)])
                         - np.diag(sum([C_aug @ ( x[i].T @  m_v_aug[i]).T for i in range(I)]))) / (sum([x[i].shape[0] for i in range(I)])))
        
        if self.D > 0:
            E_n_n_1 = [np.block([[m_v[i][1:,:].T @ m_v[i][:-1,:] + np.sum(cov_v[i], axis = 0)], [y[i][1:].T @ m_v[i][1:]]])  for i in range(I)]
            E_n_1_n_1 = [np.block([ [m_v[i][:-1,:].T @ m_v[i][:-1,:]  + np.sum(S_v[i][:-1],axis = 0), m_v[i][:-1].T @ y[i][1:]], 
                                      [y[i][1:].T @ m_v[i][:-1], y[i][1:].T @ y[i][1:]]])  for i in range(I)]
            Aaug = (sum(E_n_n_1).T @ np.linalg.inv(sum(E_n_1_n_1)))
            self.A = Aaug[:, :self.K]
            self.V = Aaug[:, self.K:]
                
        else:
            E_n_n_1 = [(m_v[i][1:,:].T @ m_v[i][:-1,:] + np.sum(cov_v[i], axis = 0)) for i in range(I)]
            E_n_1_n_1 = [m_v[i][:-1,:].T @ m_v[i][:-1,:]  + np.sum(S_v[i][:-1],axis = 0)  for i in range(I)]
            self.A = (sum(E_n_n_1) @ np.linalg.inv(sum(E_n_1_n_1)))
        
        #U,S,V = np.linalg.svd(self.A)
        #S[S > self.svd_thresh] = self.svd_thresh
        #self.A = U @ np.diag(S) @ V
        
        self.mu0 = np.mean([m_v[i][0] for i in range(I)], axis = 0)
        self.S0 = np.mean([(m_v[i][0] - self.mu0)[None].T @ (m_v[i][0] - self.mu0)[None] + S_v[i][0] for i in range(I)], axis = 0)
            
    def fit(self, x, y = None, epoch = 200, step_elbo = 1):
        
        elbo_old = 0

        self.init_with_data(x, y)
        elbo_vec = []
        
        for e in range(epoch):
            
            self.e_step_seq(x, y)
            
            if e%step_elbo == 0:
                elbo = self.elbo_multi(self.m_v, self.S_v, x, y)
                cnv = elbo - elbo_old
                elbo_old = elbo
                print(cnv)
                # if cnv < 0:
                #     print(cnv) 
                elbo_vec.append(elbo)
                self.elbo_vec = elbo_vec
            
            self.m_step(self.m_v, self.S_v, self.cov_v, x, y)
            
            # if abs(cnv) < 1e-5 and e > 10:
            #     break
        
        
    def predict(self, x, y = None, ahead = False):
        
        self.e_step_seq(x, y, backward=False)
        Xp = self.impute(ahead)
        
        return Xp
        
    def impute(self, ahead = False):
        
        Xp = []
        for i in range(len(self.m_v)):
            if ahead:
                Xp.append(self.m_t_vec[i] @ self.C + self.c)
            else:
                Xp.append(self.m_v[i] @ self.C + self.c)
        
        return Xp
        
        
if __name__ == "__main__":
    
    np.random.seed()

    I = 10 # Number of sequences
    K = 3  # Latent space dimension
    N = 10 # Number of instances
    D = 0 # Dimension of the covariates
    M = 10
    
    
    # Data Generate
    mu0 = np.zeros(K)
    S0 = np.eye(K)
    v = np.zeros(K)
    V = np.random.normal(size = D * K).reshape(K, D)
    #a = np.random.normal(size = K)
    a = np.zeros(K)
    A = ortho_group.rvs(dim=K)
    A = A @ np.diag(np.random.uniform(high = 0.5, size = K)) @ A.T
    #A = np.eye(K)
    Q = 1 * np.eye(K)
    C = np.random.multivariate_normal(np.zeros(K), np.eye(K), size = M)
    c = np.random.multivariate_normal(np.zeros(M), np.eye(M))
    Sigma = 0.1 * np.eye(M)
    
    y = None
    if D > 0:
        y = []
        
    mean = []
    cov = []
    X = []
    for i in range(I):
        mu0_in = np.tile(a + v, (N, 1))
        if D > 0:
            y.append(np.random.normal(size = N * D).reshape(N, D)) # We ignore first y
            mu0_in += y[-1] @ V.T
        z = [np.random.multivariate_normal(mu0, S0)]
        for n in range(1, N):
            z.append(np.random.multivariate_normal(A @ z[-1] + mu0_in[n], Q))
        z = np.array(z)
        mean.append(z @ C.T + c)
        X.append(np.array([np.random.multivariate_normal(mean[-1][n], Sigma) for n in range(N)]))
        

    # Ground truths
    mean_gt = mean_data(I, N, D, y, A, a, V, v, C, c, mu0)
    Cov_gt = norm_cov(cov_data(N, A, Q, C))
    
    np.random.seed()
    
    # Model fit
    model = LDS(M = M, K = K, D = D)
    model.fit(X, y, step_elbo=1, epoch = 10000)
    plt.plot(model.elbo_vec)
    plt.show()
    X_pred = model.predict(X, y)
    #print('Train MSE: ', np.mean([np.mean((X_pred[i] - X[i])**2) for i in range(I)]))
    
    Xp_a = model.predict(X, y, ahead = True)
    
    #print('MSE-ahead: ', np.mean([np.mean((Xp_a[i] - X[i])**2) for i in range(I)]))
    #print('Mean val:', np.mean([np.mean((X[i].mean(axis = 0) - X[i])**2) for i in range(I)]))
    
    # Estimations - model
    mean_pred = mean_data(I, N, D, y, model.A, model.a, model.V, model.v, model.C.T, model.c, model.mu0)
    Cov_pred = norm_cov(cov_data(N, model.A, model.Q, model.C.T))
    
    # Estimations - sample
    X_vec = np.array(X).reshape(I, N*M)
    Cov_sample = norm_cov(np.cov(X_vec.T))
    mean_sample = np.mean(X_vec, axis = 0)
    mean_sample = X_vec
    
    
    plt.imshow(mean_gt)
    plt.grid(False)
    plt.show()
    plt.imshow(mean_sample)
    plt.grid(False)
    plt.show()
    plt.imshow(mean_pred)
    plt.grid(False)
    plt.show()
    
    plt.imshow(Cov_gt)
    plt.grid(False)
    plt.show()
    plt.imshow(Cov_sample)
    plt.grid(False)
    plt.show()
    plt.imshow(Cov_pred)
    plt.grid(False)
    plt.show()
    
    print('MSE-mean-model:', np.mean((mean_pred[:, :1*M] - mean_gt[:, :1*M])**2)) # First 2 timesteps to cover params
    print('MSE-mean-sample:', np.mean((mean_pred[:, :1*M] - mean_sample[:, :1*M])**2))
    print('MSE-cov-model:', np.mean((Cov_pred - Cov_gt)**2))
    print('MSE-cov-sample:', np.mean((Cov_sample - Cov_gt)**2))
    
    print('Largest eigenvalue-gt:',max(np.linalg.eig(A)[0]))
    print('Largest eigenvalue-model:',max(np.linalg.eig(model.A)[0]))
    