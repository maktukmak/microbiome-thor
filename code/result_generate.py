import sys
import os
from os.path import dirname
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
import csv

sys.path.insert(1, os.path.join(dirname(os.getcwd()), 'Library'))

# Read data
path_ways = os.path.join(dirname(os.getcwd()), 'Dataset_bio\\THOR_path\\')
path_data = os.path.join(dirname(os.getcwd()), 'Dataset_bio\\THOR_comm\\')

with open(os.getcwd() + '\\cache\\Thor_mm.pickle', 'rb') as file:
    Xc = pickle.load(file)

def norm_cov(cov):
    d = np.sqrt(np.diag(cov))
    cov = cov / d
    cov /= d[:, np.newaxis]

    return cov

d_res = {}
for c in ['BFKEC', 'BFK']:
    with open(os.getcwd() + '\\cache\\Thor_res2_' + c + '.pickle', 'rb') as file:
        d_res[c] = pickle.load(file)

#with open(path_data + 'cache\\Thor_res.pickle', 'rb') as file:
#    d_res = pickle.load(file)
    
with open(os.getcwd() + '\\cache\\Thor_paths.pickle', 'rb') as file:
    paths = pickle.load(file)
    
# Plot BIC
if False:
    zdim_vec = np.arange(1, 10, 1)
    [plt.plot(zdim_vec, data[0], marker = 'o', label = c) for c, data in d_res.items()]
    plt.legend()
    plt.xlabel('Latent Space Dimension')
    plt.ylabel('BIC')
    plt.tight_layout()
    #plt.savefig(path_output + 'bic_micro.pdf')
    plt.show()
    

# Plot means
mean_BFKEC = d_res['BFKEC'][1]
mean_BFK = d_res['BFK'][1]

plt.plot(mean_BFKEC, label = 'BFKEC', alpha = 0.7)
plt.plot(mean_BFK, label = 'BFK', alpha = 0.7)
plt.ylim(0,0.20)
plt.xlabel('Gene ID')
plt.ylabel('Abundance Ratio')
plt.legend()
plt.show()


# Plot emprical dist of means
labels = ['b', 'f', 'k']
for i in range(3):
    n = len(mean_BFK)//3
    m_w = mean_BFK[i*n : (i+1)*(n)]
    m_m = mean_BFKEC[i*n : (i+1)*(n)]
    n_bins = 40
    # get positions and heights of bars
    heights_w, bins_w = np.histogram(m_w[(m_w > 0.005) * (m_w < 0.30)], density=True, bins=n_bins)
    bin_width_w = np.diff(bins_w)[0]
    bin_pos_w =( bins_w[:-1] + bin_width_w / 2) * 1
    # plot
    pw=plt.bar(bin_pos_w, heights_w, width=bin_width_w,
    edgecolor='black',label='Wildtype')
    heights_m, bins_m = np.histogram(m_m[(m_m > 0.005) * (m_m < 0.030)], density=True, bins=n_bins)
    bin_width_m = np.diff(bins_m)[0]
    bin_pos_m =( bins_m[:-1] + bin_width_m / 2) * 1
    pm=plt.bar(bin_pos_m, -heights_m, width=bin_width_m, edgecolor='black',
    label='Mutant')
    #plt.title('Empirical distributions')
    plt.legend()
    plt.xlabel('Ratio')
    plt.ylabel('Counts')
    plt.ylim(-540, 540)
    tmp = "output\\mean_hist_" + labels[i] + ".pdf"
    filename = os.path.join(os.getcwd(), tmp)
    plt.savefig(filename, quality = 95)
    plt.show()
    
    print(labels[i] + ' mean change (Hellinger)', (1/np.sqrt(2)) * np.sqrt(((np.sqrt(m_w) - np.sqrt(m_m))**2).sum()))
    
p_m = (np.array([Xc[0][k].sum(axis = 0) for k in range(3)]) / np.array([Xc[0][k].sum(axis = 0) for k in range(3)]).sum(axis = 0)).T
p_w = (np.array([Xc[1][k].sum(axis = 0) for k in range(3)]) / np.array([Xc[1][k].sum(axis = 0) for k in range(3)]).sum(axis = 0)).T
print('Abundance change ratio:', (p_m.mean(axis=0) - p_w.mean(axis=0)))



# Plot Covariances
M = len(d_res['BFK'][1])
Mi = M // 3
Ai = block_diag(*[np.linalg.inv(0.5*(np.eye(Mi) - (1/(Mi+1))*np.ones((Mi, Mi))))]*3)
Cor_BFKEC = norm_cov(d_res['BFKEC'][2])
Cor_BFK = norm_cov(d_res['BFK'][2])
#Cor_BFKEC = np.corrcoef(np.vstack(Xc[0]))
#Cor_BFK = np.corrcoef(np.vstack(Xc[0]))
plt.imshow(Cor_BFKEC)
plt.grid(None)
plt.show()
plt.imshow(Cor_BFK)
plt.grid(None)
plt.show()

# Plot Sparsity
th = 0.95
Cor_BFKEC_p = np.zeros(Cor_BFKEC.shape)
ind = abs(Cor_BFKEC) > th
Cor_BFKEC_p[ind] = Cor_BFKEC[ind] 
Cor_BFK_p = np.zeros(Cor_BFK.shape)
ind = abs(Cor_BFK) > th
Cor_BFK_p[ind] = Cor_BFK[ind] 

pcork = np.sqrt(abs(Cor_BFK_p*Cor_BFKEC_p))*np.sign(Cor_BFK_p*Cor_BFKEC_p)
np.fill_diagonal(pcork, 0.0)
pcork[0,0]=-1 #This is to obtain same colormap as for signed non-persistent
pcork[-1,-1]=1
pcorks=csr_matrix(pcork)
pcorksum=np.sum(abs(pcorks)>0,axis=0)
print('#Persistent nodes cor_:',np.count_nonzero(pcorksum))
plt.spy(pcork)
plt.show()

npcork=Cor_BFKEC_p+Cor_BFK_p
np.fill_diagonal(npcork, 0.0)
npcork[abs(pcork)>0]=0
npcorksum=np.sum(abs(npcork)>0,axis=0)
print('#connected nodes distinct to cor_w and cor_m: {0}'.format(np.count_nonzero(npcorksum)))
plt.spy(npcork)
plt.show()



Cor_BFKEC[abs(Cor_BFKEC)<th]=0
Cor_BFK[abs(Cor_BFK)<th]=0

for i in range(3):
    
    degrees_w=np.sum(abs(Cor_BFK)>0,axis=0)
    degrees_m=np.sum(abs(Cor_BFKEC)>0,axis=0)
    
    inds=np.argsort(degrees_w[i*Mi:(i+1)*Mi])
    fig, axs = plt.subplots(2)
    #fig.suptitle(' node degrees for w (top) and m (bottom)')
    axs[0].plot(degrees_w[i*Mi:(i+1)*Mi][inds[::-1]])
    axs[1].plot(degrees_m[i*Mi:(i+1)*Mi][inds[::-1]])
    tmp = "output\\degree_change_" + labels[i] + ".pdf"
    filename = os.path.join(os.getcwd(), tmp)
    axs[0].set_ylim(0,250)
    axs[0].set_ylabel('Degree-wild')
    axs[1].set_ylim(0,400)
    axs[1].set_ylabel('Degree-mutant')
    plt.xlabel('Orthology ID')
    plt.savefig(filename, quality = 95)
    plt.show()
    
    inds=np.argsort(degrees_m[i*Mi:(i+1)*Mi])
    fig, axs = plt.subplots(2)
    #fig.suptitle(' node degrees for w (top) and m (bottom)')
    axs[0].plot(degrees_w[i*Mi:(i+1)*Mi][inds[::-1]])
    axs[1].plot(degrees_m[i*Mi:(i+1)*Mi][inds[::-1]])
    plt.show()



dmean_importance = mean_BFKEC - mean_BFK
for i in range(3):
    Ddegrees=degrees_m-degrees_w
    inds=np.argsort(Ddegrees[i*Mi:(i+1)*Mi])
    fig, axs = plt.subplots(1)
    #fig.suptitle(' node degree difference m-w')
    axs.plot(Ddegrees[i*Mi:(i+1)*Mi][inds[::-1]], label = 'Degree diff')
    axs.plot(dmean_importance[i*Mi:(i+1)*Mi][inds[::-1]]*20000 , label = 'Mean diff')
    indsr=inds[::-1]
    plt.legend()
    plt.ylim(-500,500)
    plt.xlabel('Orthology ID')
    tmp = "output\\degree_mean_change_" + labels[i] + ".pdf"
    filename = os.path.join(os.getcwd(), tmp)
    plt.savefig(filename, quality = 95)
    plt.show()


cols = ['Orthology ID',
        'Ddegree', 'degree_w', 'degree_m',
        'Dmean', 'mean_w', 'mean_m',
        'connections_w', 'connections_m']


for i in range(3):
    
    # Extract marginally thresholded connection
    cor_m = Cor_BFKEC
    cor_w = Cor_BFK
    
    np.fill_diagonal(cor_w, 0.0)
    connections = np.nonzero(cor_w)
    dict_conn = defaultdict(list)
    for k, v in zip(connections[0], connections[1]):
        s = paths[v % len(paths)]
        if Cor_BFKEC[k,v] > 0:
            s = '+' + s + '_' + labels[v // len(paths)]
        else:
            s = '-' + s + '_' + labels[v // len(paths)]
        dict_conn[paths[k % len(paths)] + '_' + labels[k // len(paths)]].append(s)
    conn_list_w = [dict_conn[g + '_' + labels[i]] for g in paths]
    

    np.fill_diagonal(cor_m, 0.0)
    connections = np.nonzero(cor_m)
    dict_conn = defaultdict(list)
    for k, v in zip(connections[0], connections[1]):
        s = paths[v % len(paths)]
        if Cor_BFKEC[k,v] > 0:
            s = '+' + s + '_' + labels[v // len(paths)]
        else:
            s = '-' + s + '_' + labels[v // len(paths)]
        dict_conn[paths[k % len(paths)] + '_' + labels[k // len(paths)]].append(s)
    conn_list_m = [dict_conn[g + '_' + labels[i]] for g in paths]
    
    
    X_save = np.vstack([paths,
                        Ddegrees[i*Mi:(i+1)*Mi], degrees_w[i*Mi:(i+1)*Mi], degrees_m[i*Mi:(i+1)*Mi],
                        1000*dmean_importance[i*Mi:(i+1)*Mi], 1000*mean_BFK[i*Mi:(i+1)*Mi], 1000*mean_BFKEC[i*Mi:(i+1)*Mi], 
                        np.array(conn_list_m, dtype = object), np.array(conn_list_w, dtype = object)]).T
    X_save = X_save[np.argsort(Ddegrees[i*Mi:(i+1)*Mi])[::-1]]
    #np.savetxt('corscreen_' + species + '.csv', np.append([cols], X_save, axis = 0), delimiter = ',', fmt="%s")
    with open('output\\count_analysis_' + labels[i] + '.csv', 'w', newline='') as f:
        csv.writer(f).writerows(np.concatenate((np.array(cols)[None], X_save)))

