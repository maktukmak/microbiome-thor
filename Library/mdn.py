import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from glm import glm
import matplotlib.pyplot as plt
from scipy.special import expit, logsumexp, softmax
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Add, LSTM
from tensorflow.keras import regularizers

tf.config.experimental_run_functions_eagerly(False)

'''
To-do
    
    opt - nan problem in seperate nns (very rarely happened)
    
    done - Eager mode   
    done - mini batch training
    done - convergence criteria
    done - One component
    done - Categorical output
    done - Regularization
    
'''

class mdn(tf.keras.Model):
    def __init__(self, K, Dx, Dt, M,
                 sk = 1, ss = 1,
                 obs = None,
                 expert_type = 'lr',
                 cnv_th = 1e-2,
                 batch_size = 0):
        
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
        t:      Gate input (N x Dt)
        y:      Response variable (N x M) (if obs is cat, then y is one-hot encoded with last dimension excluded, i.e., M = C-1)

        '''
        
        super().__init__()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        self.obs = obs
        self.K = K
        self.Dx = Dx
        self.Dt = Dt
        self.M = M
        
        # Configurations
        self.expert_type = expert_type # 'nn', 'rnn', 'lr'
        self.sk = sk  # Regularization strength of experts
        self.ss = ss # Regularization strength of gating
        self.cnv_th = cnv_th # LogLikelihood change treshold for convergence
        self.batch_size = batch_size 
        
        # Init components
        self.x_init()
        self.g_init()
        
    def x_init(self):
        
        if self.expert_type != 'lr':
            if self.expert_type == 'nn':
                self.x_enc_hidden = Dense(32, activation = 'softplus', kernel_regularizer=regularizers.l1_l2(l2 = self.sk))
            elif self.expert_type == 'rnn': 
                self.x_enc_hidden = LSTM(32)
            
        if self.obs == 'gauss':
            self.x_enc_mean = [Dense(self.M, kernel_regularizer=regularizers.l1_l2(l2 = self.sk)) for _ in range(self.K)]
            self.x_enc_var = [Dense(self.M, kernel_regularizer=regularizers.l1_l2(l2 = self.sk)) for _ in range(self.K)]
        elif self.obs == 'cat':
            self.x_enc_mean = [Dense(self.M, kernel_regularizer=regularizers.l1_l2(l2 = self.sk)) for _ in range(self.K)]
    
    def g_init(self):
        self.g_enc_mean = Dense(self.K, activation = 'softmax', kernel_regularizer=regularizers.l1_l2(l2 = self.ss))
        
    #@tf.function
    def x_encode(self, x, i):
        if self.expert_type != 'lr':
            #x = self.x_enc_hidden[i](x)
            x = self.x_enc_hidden(x)
        if self.obs == 'gauss':
            mean = self.x_enc_mean[i](x)
            logvar = self.x_enc_var[i](x)
            return mean, logvar
        elif self.obs == 'cat':
            mean = self.x_enc_mean[i](x)
            mean = tf.concat((mean, tf.zeros((mean.shape[0], 1))), axis = 1)
            mean = tf.nn.softmax(mean, axis = 1)
            return mean
    
    @tf.function
    def g_encode(self, x):
        mean = self.g_enc_mean(x)
        return mean
    
    @tf.function
    def train_batch_classifier(self, data):
    
        with tf.GradientTape() as s_tape:
            loss_s = self.compute_loss(data)
        gradients = s_tape.gradient(loss_s, self.trainable_variables)
        self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.trainable_variables) if grad is not None)
        
        return loss_s
    
    @tf.function
    def gauss_ll(self, x, m, s):
        
        ll = -tf.reduce_sum(((x - m) ** 2) / (2 * tf.exp(s)) + 0.5 * tf.math.log(2*np.pi) + 0.5 * s, axis = 1)
        
        return ll
    
    @tf.function
    def cat_ll(self, x, m):
        
        ll = tf.reduce_sum(tf.math.log(m) * tf.concat((x, 1 - tf.reduce_sum(x, axis = 1, keepdims = True)), axis = 1), axis = 1)
        
        return ll
        
    @tf.function
    def compute_loss(self, data, train = True):
        
        x = data[0]
        t = data[1]
        y = data[2]
        
        pi = self.g_encode(t)
        
        loss = 0
        for k in range(self.K):
            
            if self.obs == 'gauss':
                m, s = self.x_encode(x, k)
                ll = self.gauss_ll(y, m, s)
            elif self.obs == 'cat':
                m = self.x_encode(x, k)
                ll = self.cat_ll(y, m)
            
            loss += pi[:,k] * tf.exp(ll)
            
        loss = -tf.reduce_sum(tf.math.log(loss)) + sum(self.losses)
        
        return loss
    
    
    def fit(self, x, t, y, epoch = 10):
        
        if self.batch_size == 0:
            n = x.shape[0]
        else:
            n = self.batch_size
        train_ds = tf.data.Dataset.from_tensor_slices(tuple((x, t, y))).shuffle(256).batch(n)
        
        self.cnv_vec = []
        Q_vec = [1e8]
        
        for i in range(epoch):
            
            loss_vec = []
            for data in train_ds:
                loss = self.train_batch_classifier(data)
                loss_vec.append(loss.numpy())

            self.cnv_vec.append((Q_vec[-1] - sum(loss_vec)))
            Q_vec.append(sum(loss_vec))
            
            if self.cnv_vec[-1] < self.cnv_th:
                break
            
        return np.array(Q_vec)[1:]
    
    
    def predict(self, x, t):
        
        pi = self.g_encode(t)
        ind = np.argmax(pi, axis = 1)
        
        if self.obs == 'gauss':
            sc = np.array([self.x_encode(x, k)[0] for k in range(self.K)])
        elif self.obs == 'cat':
            sc = np.array([self.x_encode(x, k) for k in range(self.K)])

        yp = np.array([sc[ind[i], i] for i in range(len(x))])
        
        if self.obs == 'cat':
            yp = np.eye(self.M + 1)[np.argmax(yp, axis = 1)][:,:-1]
        
        return yp


if __name__ == "__main__":


    if True: # Regression test    
        
        multivariate = False
    
        n = 200;
        y = np.random.uniform(size = n)
        eta = np.random.normal(size = n) * 0.05
        
        x = y + 0.3*np.sin(2*np.pi*y) + eta
        Dx = 1
        Dt = 1
        
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        x = x[:,None]
        y = y[:,None]
        
        M = 1
        if multivariate:
            M = 2
            y = np.append(y, y, axis = 1)
        
        model_moe = mdn(K = 3,
                        Dx = Dx,
                        Dt = Dt,
                        M = M,
                        sk = 1, ss = 1,
                        obs = 'gauss',
                        expert_type = 'lr',
                        cnv_th=1e-5,
                        batch_size=0)
        
        q_vec = model_moe.fit(x, x, y, epoch = 2000)
        
        plt.plot(q_vec)
        plt.show()
        
        
        # plt.plot(q_vec)
        # plt.show()
        
        yp = model_moe.predict(x, x)
        
        plt.scatter(x, y[:,0])
        plt.scatter(x, yp[:,0])
        
        
        t = np.arange(min(x), max(x), 1e-2)[None].T
        resp = model_moe.g_encode(t)
        plt.plot(t, resp[:,0])
        plt.plot(t, resp[:,1])
        plt.plot(t, resp[:,2])
        
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
        #K = 1
        
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
            
        t = x
        #x = np.tile(x, (1, 5)).reshape((x.shape[0], 5, -1))
        
        y = np.eye(M + 1)[y][:,:-1]
        
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        
        model_moe = mdn(K = K,
                        Dx = Dx,
                        Dt = Dt,
                        M = M,
                        sk = 1, ss = 1,
                        obs = 'cat',
                        expert_type = 'nn',
                        batch_size=0)
        
        start = time.time()
        q_vec = model_moe.fit(x, t, y, epoch = 5000)
        end = time.time()
        print('Fit time:', end- start)
        plt.plot(q_vec)
        plt.show()
        
        yp = model_moe.predict(x, t)
        print('Acc:', np.sum(np.all(yp == y, axis = 1)) / (n*(M+1)))
        
        yp = np.concatenate((yp, 1 - yp.sum(axis = 1)[None].T), axis = 1)
        plt.scatter(x[:,0], x[:,1], c = np.argmax(yp, axis = 1))

        