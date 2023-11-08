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
from data_gen import data_gen
#tf.keras.backend.set_floatx('float64')

tf.config.experimental_run_functions_eagerly(True)
#tf.config.run_functions_eagerly(True)

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

class nn(tf.keras.Model):
    def __init__(self, D, M,
                 sk = 1,
                 obs = 'cat',
                 cnv_th = 1e-1,
                 batch_size = 0,
                 expert_type = 'nn'):
        
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
        self.D = D
        self.M = M
        
        # Configurations
        self.expert_type = expert_type
        self.sk = sk  # Regularization strengt
        self.cnv_th = cnv_th # LogLikelihood change treshold for convergence
        self.batch_size = batch_size 
        
        # Init components
        self.x_init()
        
    def get_params(self, deep=True):
        
        return {"D": self.D, "M": self.M}
        
    def x_init(self):

        if self.expert_type == 'nn':
            self.x_enc_hidden = Dense(32, activation = 'softplus', kernel_regularizer=regularizers.l1_l2(l2 = self.sk))
        elif self.expert_type == 'rn': 
            self.x_enc_hidden = LSTM(32, kernel_regularizer=regularizers.l1_l2(l2 = self.sk))
            #self.x_enc_hidden2 = LSTM(32)
            
        if self.obs == 'gauss':
            self.x_enc_mean = Dense(self.M, kernel_regularizer=regularizers.l1_l2(l2 = self.sk)) 
            self.x_enc_var = Dense(self.M, kernel_regularizer=regularizers.l1_l2(l2 = self.sk)) 
        elif self.obs == 'cat':
            self.x_enc_mean = Dense(self.M, kernel_regularizer=regularizers.l1_l2(l2 = self.sk))
    
      
    #@tf.function
    def x_encode(self, x):
        x = self.x_enc_hidden(x)
        #x = self.x_enc_hidden2(x)
        
        if self.obs == 'gauss':
            mean = self.x_enc_mean(x)
            logvar = self.x_enc_var(x)
            return mean, logvar
        elif self.obs == 'cat':
            mean = self.x_enc_mean(x)
            mean = tf.concat((mean, tf.zeros((mean.shape[0], 1))), axis = 1)
            mean = tf.nn.softmax(mean, axis = 1)
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
        
        ll = -tf.reduce_sum(((x - m) ** 2) / (2 * tf.exp(s)) + 0.5 * tf.math.log(2*tf.math.pi) + 0.5 * s, axis = 1)
        
        return ll
    
    @tf.function
    def cat_ll(self, x, m):
        
        ll = tf.reduce_sum(tf.math.log(m) * tf.concat((x, 1 - tf.reduce_sum(x, axis = 1, keepdims = True)), axis = 1), axis = 1)
        
        return ll
        
    @tf.function
    def compute_loss(self, data, train = True):
        
        x = data[0]
        y = data[1]

        if self.obs == 'gauss':
            m, s = self.x_encode(x)
            ll = self.gauss_ll(y, m, s)
        elif self.obs == 'cat':
            m = self.x_encode(x)
            ll = self.cat_ll(y, m)

        loss = -tf.reduce_sum((ll)) + sum(self.losses)
        
        return loss
    
    
    def fit(self, x, y, epoch = 1000):
        
        self.x_init()
        
        if self.batch_size == 0:
            n = x.shape[0]
        else:
            n = self.batch_size
        train_ds = tf.data.Dataset.from_tensor_slices(tuple((x, y))).shuffle(256).batch(n)
        
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
    
    
    def predict(self, x):

        if self.obs == 'gauss':
            yp = self.x_encode(x)[0]
        elif self.obs == 'cat':
            yp = self.x_encode(x)

        if self.obs == 'cat':
            yp = np.eye(self.M + 1)[tf.argmax(yp, axis = 1)][:,:-1]
        
        return yp


if __name__ == "__main__":
    
    if True: #LSTM test
    
    
        static = False # Dynamic / Static
        
        I = 100 # Number of sequences
        K = 3  # Latent space dimension
        N = 20 # Number of instances
        D = 0 # Dimension of the covariates
        mr = 0.0 # Random missing rate
        
        P = [1]  # of multinomial modalities
        M = [5] * P[0] # Observation dimension
        C_max = [0] * P[0] # Total number of counts - 1 (if 0 -> Categorical)
        
        P += [1]  # of gaussian modalities
        M = np.append(M, [22] * P[1]) # Observation dimension
        
        data = data_gen(K, P, M, C_max, mr, D, static)
        X, y, mean = data.generate_data(N, I)
        #X, y, mean, Xte, yte, meante = data.train_test_split(X, y, mean)
        
        y = np.array(X[0])[:,-1,:-1]
        y = y.astype(np.float32)
        X = np.array(X[1])
        
    
        model = nn(
                    D = M[1],
                    M = M[0],
                    sk = 1,
                    obs = 'cat',
                    cnv_th=1e-5,
                    batch_size=0)
        
        
        
        q_vec = model.fit(np.array(X), y, epoch = 2000)
        plt.plot(q_vec)
        
        


    if False: # Regression test    
        
        multivariate = False
    
        n = 200;
        y = np.random.uniform(size = n)
        eta = np.random.normal(size = n) * 0.05
        
        x = y + 0.3*np.sin(2*np.pi*y) + eta
        D = 1
        
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
        
        model_moe = nn(
                        D = D,
                        M = M,
                        sk = 1,
                        obs = 'gauss',
                        cnv_th=1e-5,
                        batch_size=0)
        
        q_vec = model_moe.fit(x, y, epoch = 2000)
        
        plt.plot(q_vec)
        plt.show()
        
        
        # plt.plot(q_vec)
        # plt.show()
        
        yp = model_moe.predict(x)
        
        plt.scatter(x, y[:,0])
        plt.scatter(x, yp[:,0])
        
        
        
    if False: # Categorical test    
            
        multiclass = True
        
        n = 200
        Dx = 2
        
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
            

        y = np.eye(M + 1)[y][:,:-1]
        
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        
        model_moe = nn(
                        D = Dx,
                        M = M,
                        sk = 1,
                        obs = 'cat',
                        cnv_th=1e-5,
                        batch_size=0)
        
        start = time.time()
        q_vec = model_moe.fit(x, y, epoch = 5000)
        end = time.time()
        print('Fit time:', end- start)
        plt.plot(q_vec)
        plt.show()
        
        yp = model_moe.predict(x)
        print('Acc:', np.sum(np.all(yp == y, axis = 1)) / (n*(M+1)))
        
        yp = np.concatenate((yp, 1 - yp.sum(axis = 1)[None].T), axis = 1)
        plt.scatter(x[:,0], x[:,1], c = np.argmax(yp, axis = 1))

        