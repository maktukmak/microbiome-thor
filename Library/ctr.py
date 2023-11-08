from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import numpy as np
import scipy as sp
from scipy.stats import multinomial
import numpy.matlib as npm
from scipy.special import loggamma, digamma, polygamma
import matplotlib.pyplot as plt
from numpy.linalg import inv
#from utils import softmax
from scipy.special import softmax
import time

def logdet(x):
    
    return np.log(np.linalg.det(x))

def logsumexp(x):
    
    lse = np.log(np.sum(np.exp(x - np.amax(x,axis=0,keepdims=True)), axis = 0, keepdims=True)) + np.amax(x,axis=0,keepdims=True)
    
    return lse

def generate_cat(K = 3, V = 10, N = 100, R = 50):

    pi = np.random.dirichlet(np.ones(K) / K, size = N)
    B = np.zeros((R, K, V))
    for r in range(R):
        B[r] = np.random.dirichlet(np.ones(V) / K, size = K)
    
    q = np.zeros((N, R)).astype(int)
    Y = np.zeros((N, R))
    #X = np.zeros((N, V))
    for i in range(N):
        for r in range(R):
            q[i, r] = np.argmax(np.random.multinomial(1, pi[i]))
            Y[i, r] = np.argmax(np.random.multinomial(1, B[r, q[i, r]]))
    
    #for i in range(N):
    #    for v in range(V):
    #        X[i] = np.histogram(y[i], bins = V, range = (0, V))[0]
            
    return Y

def ll(B, Y, pi):
    
    ll = 0
    for r in range(Y.shape[1]):
        prob_v = pi @ (B[r] / B[r].sum(axis = 1)[None].T)
        prob_v = prob_v / prob_v.sum(axis = 1)[None].T
    
        for i in range(Y.shape[0]):
            ll = ll + np.log(prob_v[i, Y[i, r].astype(int)])
    
    #print("Train LL:", ll)
    return ll

def E_step_cat_ctr_seq(Y_train, B, alpha, mu, Sigma):
    
    lmbd = mu.copy()
    v = np.diag(Sigma)
    zeta = 1
    
    conv = 1
    cnt = 0
    while( conv > 0.001 and cnt < 100 ):

        zeta = np.exp(lmbd + (v**2)/2).sum()
        q = []
        [q.append(B[i][:, Y_train[i].astype(int)] * np.exp(lmbd)) for i in np.arange(len(Y_train))]
        #q = B[np.arange(len(Y_train))][:, Y_train.astype(int)] * np.exp(lmbd)
        q = np.array(q) / np.array(q).sum(axis = 1)[None].T
        
        d_lmbd = -inv(Sigma) @ (lmbd - mu) + q.sum(axis = 0) - (len(Y_train) / zeta) * np.exp(lmbd + (v**2)/2)
        d_v = -np.diag(Sigma)/2 - (len(Y_train)/(2*zeta)) * np.exp(lmbd  + (v**2)/2) + 1/(2*v**2)
        
        lmbd = lmbd + 0.1 * d_lmbd
        v = v + 0.1 * d_v
        v = v.clip(min=1e-2)
        
        conv = (1/len(lmbd)) * abs(d_lmbd).sum()
        cnt += 1
        #print(conv)
        
    return lmbd, v, q


def cat_ctr_seq_fit(K, V, R, B_init, Y_train, Y_test, epoch = 30):

    alpha = np.ones(K) / K
    eta = 1 / K
    Sigma = np.eye(K)
    mu = np.ones(K)
    B = B_init.copy()
    lmbd = np.zeros((Y_train.shape[0], K))
    v2 = np.zeros((Y_train.shape[0], K))
    lmbd_test = np.zeros((Y_test.shape[0], K))
    v2_test = np.zeros((Y_test.shape[0], K))
    q = np.zeros((Y_train.shape[0], R, K))
    
    ll_vec = []
    ll_vec_test = []
    for iter in range(epoch): 
        start = time.time()
        
        for i in range(Y_train.shape[0]):
            lmbd[i], v2[i], q[i] = E_step_cat_ctr_seq(Y_train[i], B, alpha, mu, Sigma)
        
        
        for r in range(R):
            for v in range(V[r]):
                B[r][:, v] = eta + q[np.where(Y_train[:,r] == v)[0], r, :].sum(axis = 0)
            
        for r in range(R):
            B[r] = B[r] / B[r].sum(axis = 1)[None].T
            
        mu = lmbd.mean(axis = 0)
        Sigma = (lmbd - mu).T @ (lmbd - mu) / len(Y_train) + np.diag(v2.mean(axis = 0))
            
        for i in range(Y_test.shape[0]):
            lmbd_test[i], v2_test, _ = E_step_cat_ctr_seq(Y_test[i, 0:-1], B, alpha,  mu, Sigma)
        
        ll_vec.append(ll(B, Y_train, softmax(lmbd, axis = 1)))
        ll_vec_test.append(ll(B, Y_test, softmax(lmbd_test, axis = 1)))
        #ll_vec_test.append(ll([B[-1]], Y_test[:,-1], softmax(lmbd_test, axis = 1)))
        
        #print(time.time() - start) 
        
        
    return B, ll_vec, ll_vec_test


if __name__ == "__main__":
    
    K = 3
    R = 10
    V = np.array([15]*R)
    N = 100
    
    Y = generate_cat(K = K, V = V[0], N = N, R = R)
    
    Y_train, Y_test, _, _ = train_test_split(Y, Y, test_size=0.10)
    
    B = []
    for r in range(R):
        B.append(np.random.dirichlet(np.ones(V[r]) / K, size = K))
    
    B_est, ll_vec_seq, ll_vec_seq_test = cat_ctr_seq_fit(K, V, R, B, Y_train, Y_test )
    
    plt.plot(np.array(ll_vec_seq) / len(Y_train), label = 'Train')
    plt.plot(np.array(ll_vec_seq_test) / len(Y_test), label = 'Test')
    plt.legend()
