import sys
import os
from os.path import dirname
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle

sys.path.insert(1, os.path.join(dirname(os.getcwd()), 'Library'))

from multimodal_gen import multimodal_gen
import matplotlib.pyplot as plt
from scipy.special import loggamma, multigammaln, gamma, digamma, softmax, logsumexp
from scipy.linalg import block_diag
from multiprocessing import Pool, cpu_count
import itertools

# Read data
with open(os.getcwd() + '\\cache\\Thor_mm.pickle', 'rb') as file:
    Xc = pickle.load(file)
    
with open(os.getcwd() + '\\cache\\Thor_paths.pickle', 'rb') as file:
    paths = pickle.load(file)
    
epoch = 500000
step = 50000

# Preprocess
X = [[np.append(x[:-1], x.sum(axis = 0)[None], axis = 0).T for x in Xm] for Xm in Xc]

def func(comb):
    K, P, M, X = comb
    model = multimodal_gen(P, M, K = K, D = 0, full_dim = True)
    model.fit(X, y = None, mask = None, epoch = epoch, step_comp_elbo = step)
    return model.elbo_vec

if __name__ == '__main__':
    
    
    exp = {
           'BFKEC':0,
           'BFK':1,
           }

    for c, i in exp.items():
        
        
        # Config
        P = [len(X[i]), 0]
        M = [x.shape[1]-1 for x  in X[i]]
        N = len(X[i][0])
        
        zdim_vec = [10]
        bic = 0
        if False:
            zdim_vec = np.arange(10, 51, 10)
            combs = [r for r in itertools.product(zdim_vec, [P], [M], [X[i]])]
            p = Pool(int(cpu_count()*0.75))
            res_vec = p.map(func, combs)
            bic = [res_vec[j][-1] - zdim_vec[j] * (zdim_vec[j] + 1 + sum(M)) * np.log(N) /2 for j in range(len(zdim_vec))]
            plt.plot(bic)
            plt.show()
            print('Best K:', zdim_vec[np.argmax(bic)])
        
        model = multimodal_gen(P, M, K = zdim_vec[np.argmax(bic)], D = 0, full_dim = True)
        model.fit(X[i], y = None, mask = None, epoch = epoch, step_comp_elbo = step)
        plt.plot(model.elbo_vec)
        plt.show()
            
        # Predictions
        Cov_pred = np.vstack([model.comps[p].W for p in range(sum(P))]) @ np.vstack([model.comps[p].W for p in range(sum(P))]).T
        Cov_pred += block_diag(*[model.comps[p].Ai for p in range(sum(P))])
        mean_pred = np.hstack([softmax(model.comps[p].W @ model.comps[p].mu_prior) for p in range(sum(P))])
        plt.imshow(Cov_pred)
        plt.show()
        plt.plot(mean_pred)
        plt.show()
        
        # Save
        #d[c]= [bic, mean_pred, Cov_pred]
        with open(os.getcwd() + '\\cache\\Thor_res2_' + c + '.pickle', 'wb') as file:
            pickle.dump([bic, mean_pred, Cov_pred], file)
            
        