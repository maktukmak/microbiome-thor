import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.special import loggamma, multigammaln, gamma, digamma, softmax, logsumexp

def logdet(X):
    return np.linalg.slogdet(X)[0] * np.linalg.slogdet(X)[1]


def norm_cov(cov):
    d = np.sqrt(np.diag(cov))
    cov = cov / d
    cov /= d[:, np.newaxis]

    return cov

def gauss_entropy(Sigma):
    
    M = Sigma.shape[-1]

    return np.sum(0.5 * logdet(Sigma) + 0.5 * M * np.log(2 * np.pi))


def gauss_loglik(x, mu, Lambda, x_S = None, mu_S = None, x_mu_S1 = None, x_mu_S2 = None):
    ''' 
    If x_S -> x is integrated out
    
    x -> NxD
    mu -> 1XD
    Lamda -> DXD
    '''
    
    M = x.shape[1]
    N = x.shape[0]
    
    ll = - 0.5 * np.trace((x - mu) @ Lambda @ (x - mu).T)
    ll += 0.5 * N * logdet(Lambda)
    ll += - 0.5 * N * M * np.log(2*np.pi)
    if x_S is not None:
        ll += - np.sum(0.5 * np.trace(Lambda @ x_S, axis1=1, axis2=2))
    if mu_S is not None:
        ll += - np.sum(0.5 * np.trace(Lambda @ mu_S, axis1=1, axis2=2))
    if x_mu_S1 is not None:
        ll +=  np.sum(0.5 * np.trace(Lambda @ x_mu_S1, axis1=1, axis2=2)) 
    if x_mu_S2 is not None:
        ll +=  np.sum(0.5 * np.trace(Lambda @ x_mu_S2, axis1=1, axis2=2)) 
        
    return ll

def multi_loglik_bounded(x, Th, Phi, A, full_dim, Th_S = None):
        '''
        If Th_s -> Th integrated out
        '''
        if full_dim:
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
        ll += np.sum(loggamma(Ni + 1)) - np.sum(loggamma(xi +1)) - np.sum(loggamma(1 + (Ni - np.sum(xi, axis = 1))))

        if Th_S is not None:
            ll += np.sum(Ni * (- 0.5 * np.trace(A @ Th_S, axis1=1, axis2=2))[None])
        
        return ll
    
def cov_data(N, A, Q, C):
        A_p = [A]
        for i in range(N):
            A_p.append(A @ A_p[-1])
        
        Cov_gt = []
        z_sec = Q
        for i in range(N):
            Cov_gt_row = []
            for j in range(N):
                if j == 0:
                    z_sec = A @ z_sec @ A.T + Q
                if i == j:
                    Cov_gt_row.append(C @ z_sec @ C.T)
                elif j > i:
                    Cov_gt_row.append(C @ A_p[j - i-1] @ z_sec @ C.T)
                elif i > j:
                    Cov_gt_row.append(C @ z_sec @ A_p[i - j-1]  @ C.T)
            Cov_gt.append(Cov_gt_row)
        Cov_gt = np.block(Cov_gt)
        return Cov_gt
    
def mean_data(I, N, D, y, A, a, V, v, C, c, m0):
    mean_gt = []
    for i in range(I):
        mu0_in = np.tile(a + v, (N, 1))
        if D > 0:
            mu0_in += y[i] @ V.T
        z_exp = m0
        m = [c + C @ z_exp]
        for n in range(1, N):
            z_exp = A @ z_exp + mu0_in[n]
            m.append(c + C @ z_exp)
            
        mean_gt.append(np.hstack(m))
        
        
    return np.array(mean_gt)


def norm_cov(cov):
        d = np.sqrt(np.diag(cov))
        cov = cov / d
        cov /= d[:, np.newaxis]

        return cov

def performance(y, yp, regression):
    
    if regression:
        mse = np.mean((yp - y)**2)
        return mse
    else:
        y = np.concatenate((y, 1-y.sum(axis = 1)[None].T), axis = 1)
        yp = np.concatenate((yp, 1-yp.sum(axis = 1)[None].T), axis = 1)
        
        tn, fp, fn, tp = confusion_matrix(np.argmax(y, axis = 1), np.argmax(yp, axis = 1), labels = [0,1]).ravel()
        acc = (tp + tn) / (tn + fp + fn + tp)
        hss2 = (2 * ((tp * tn) - (fn*fp))) / ((tp + fn) * (fn + tn) + (tn + fp) * (tp + fp))
        return acc, hss2