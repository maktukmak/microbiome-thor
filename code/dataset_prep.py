import sys
import os
from os.path import dirname
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import copy
import pickle
import pandas
from collections import defaultdict

sys.path.insert(1, os.path.join(dirname(os.getcwd()), 'Library'))
path_data = os.path.join(dirname(os.getcwd()), 'Dataset_bio\\THOR_treatment\count\\')
path_onth = os.path.join(dirname(os.getcwd()), 'Dataset_bio\\THOR_path\\')


# Read dataset
dataset = np.genfromtxt(path_data + 'all_htseq.tsv', dtype = str, skip_header = True)
df_path = pandas.read_table(path_onth + 'all_ko.txt', 
                       delim_whitespace=True, 
                       header=None, 
                       usecols=(0,1), 
                       engine='python')


d_path_b = defaultdict(list)
d_path_f = defaultdict(list)
d_path_k = defaultdict(list)
for A, B in zip(df_path[1], df_path[0]):
    if B[0] == 'B':
        d_path_b[A].append(B)
    elif B[0] == 'F':
        d_path_f[A].append(B)
    elif B[0] == 'P':
        d_path_k[A].append(B)
        
#del d_path_b[None]
#del d_path_f[None]
#del d_path_k[None]
d_path_b['KXXXXX'] = d_path_b.pop(None)
d_path_f['KXXXXX'] = d_path_f.pop(None)
d_path_k['KXXXXX'] = d_path_k.pop(None)

                

# Preprocess
sites = np.unique(dataset[:,0])
genes = np.unique(dataset[:,3])
X_cond1 = []
X_cond2 = []
for s in sites:
    if s[3] == 'e':
        X_cond1.append(dataset[np.where(dataset[:, 0] == s)[0]][:,-1].astype(np.float32))
    else:
        X_cond2.append(dataset[np.where(dataset[:, 0] == s)[0]][:,-1].astype(np.float32))
X = [np.array(X_cond1), np.array(X_cond2)]

K = 2
P = 3
N = [len(np.array(X_cond1)), len(np.array(X_cond2))]


# Remove zero count species
xs = np.vstack(X)
ind_r = np.where(np.sum(xs, axis = 0) == 0)[0]
print('Number of zero count species:', len(ind_r))
ind = np.setdiff1d(np.arange(xs.shape[1]), ind_r)
X_z = [X[0][:, ind], X[1][:, ind]]
genes_z = genes[ind]


# Divide to three modalities
X_m = []
for k in range(K):
    X_b = np.array([X_z[k][:,i] for i in range(len(genes_z)) if genes_z[i][0] == 'B']).T
    X_f = np.array([X_z[k][:,i] for i in range(len(genes_z)) if genes_z[i][0] == 'F']).T
    X_k = np.array([X_z[k][:,i] for i in range(len(genes_z)) if genes_z[i][0] == 'P']).T
    X_m.append([X_b, X_f, X_k])
      
    
genes_b = np.array([g for g in genes_z if g[0] == 'B'])
genes_f = np.array([g for g in genes_z if g[0] == 'F'])
genes_k = np.array([g for g in genes_z if g[0] == 'P'])
genes_sep = [genes_b, genes_f, genes_k]





# Feature Transform
comm = np.intersect1d(np.intersect1d(np.array(list(d_path_b.keys())), np.array(list(d_path_f.keys()))), np.array(list(d_path_k.keys())))
X_f = []
for k in range(K):
    X_fb = []
    X_ff = []
    X_fk = []
    for o in comm:
    #for o in d_path_b.keys():
        X_fb.append(np.array([X_m[k][0][:, np.where(genes_b == i)[0]] for i in d_path_b[o] if len(np.where(genes_b == i)[0])>0])[:,:,0].sum(axis = 0))
    #for o in d_path_f.keys():
        X_ff.append(np.array([X_m[k][1][:, np.where(genes_f == i)[0]] for i in d_path_f[o] if len(np.where(genes_f == i)[0])>0])[:,:,0].sum(axis = 0))
    #for o in d_path_k.keys():
        X_fk.append(np.array([X_m[k][2][:, np.where(genes_k == i)[0]] for i in d_path_k[o] if len(np.where(genes_k == i)[0])>0])[:,:,0].sum(axis = 0))
    X_f.append([np.array(X_fb), np.array(X_ff), np.array(X_fk)])
    
#onth = [list(d_path_b.keys()), list(d_path_f.keys()), list(d_path_k.keys())] 
onth = comm


with open(os.getcwd() + '\\cache\\Thor_mm.pickle', 'wb') as file:
    pickle.dump(X_f, file)

with open(os.getcwd() + '\\cache\\Thor_paths.pickle', 'wb') as file:
    pickle.dump(onth, file)


