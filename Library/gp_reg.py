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

tf.config.experimental_run_functions_eagerly(True)

'''
To-do
    
    - Classification
    - Overfitting due to w
    
    
    (done)- Tie option
    (done) - Accuracy when M large
    
'''

class gp_reg(tf.keras.Model):
    def __init__(self, K, D, M, s2 = 1):
        
        super().__init__()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        self.K = K
        self.D = D
        self.M = M
        self.s2 = s2
        
        self.cnv_th = 1e-1
        self.tie = False
        
        # Init components
        self.x_init()

    def x_init(self):
        self.x_enc_mean = [Dense(self.K, kernel_regularizer = regularizers.l2(1)) for _ in range(self.M)]
    
    #@tf.function
    def x_encode(self, x, i):
        mean = self.x_enc_mean[i](x)
        return mean

    @tf.function
    def train_batch_classifier(self, data):
    
        with tf.GradientTape() as s_tape:
            loss_s = self.compute_loss(data)
        gradients = s_tape.gradient(loss_s, self.trainable_variables)
        self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.trainable_variables) if grad is not None)
        
        return loss_s

    @tf.function
    def compute_loss(self, data, train = True):
        
        X = data[0]
        y = data[1]
        
        loss = 0
        for i in range(self.M):
            if self.tie:
                m = self.x_encode(X, 0)
            else:
                m = self.x_encode(X, i)
            m_0 = tf.reshape(m, (-1, 1, self.K))
            m_1 = tf.reshape(m, (1, -1, self.K))
            K = tf.reduce_sum(tf.exp(-(m_1 - m_0) * (m_1 - m_0)), axis = -1)
            K += self.s2 * tf.eye(y.shape[0])
            
            loss += tf.matmul(tf.matmul(tf.transpose(y[:,i, None]), tf.linalg.inv(K)), y[:,i, None])[0,0]
            loss += tf.linalg.logdet(K)
        loss += sum(self.losses)
        
        return loss
    
    
    def fit(self, X, y, epoch = 3000):
        
        n = X.shape[0]
        train_ds = tf.data.Dataset.from_tensor_slices(tuple((X, y))).shuffle(256).batch(n)
        
        self.cnv_vec = []
        Q_vec = [1e8]
        
        for i in range(epoch):
            
            loss_vec = []
            for data in train_ds:
                loss = self.train_batch_classifier(data)
                loss_vec.append(loss.numpy())

            self.cnv_vec.append((Q_vec[-1] - sum(loss_vec)))
            Q_vec.append(sum(loss_vec))
            
            print(Q_vec[-1])
            
            if self.cnv_vec[-1] < self.cnv_th:
                break
            
        return np.array(Q_vec)[1:]
    
    
    def predict(self, x, X, y):
        
        yp = []        
        for i in range(self.M):
            if self.tie:
                m = self.x_encode(X, 0)
            else:
                m = self.x_encode(X, i)
            m_0 = tf.reshape(m, (-1, 1, self.K))
            m_1 = tf.reshape(m, (1, -1, self.K))
            K = tf.reduce_sum(tf.exp(-(m_1 - m_0) * (m_1 - m_0)), axis = -1)
            K += self.s2 * tf.eye(y.shape[0])
        
            if self.tie:
                ms = self.x_encode(x, 0)
            else:
                ms = self.x_encode(x, i)
            m_0 = tf.reshape(ms, (-1, 1, self.K))
            Ks = tf.reduce_sum(tf.exp(-(m_1 - m_0) * (m_1 - m_0)), axis = -1)
            yp.append(tf.matmul(tf.matmul(Ks, tf.linalg.inv(K)), y[:,i, None]))
               
        return tf.concat(yp, axis = 1)


if __name__ == "__main__":


    if True: # Regression test    
        
        N = 100
        D = 10
        M = 1
        K = 1
        
        s2 = 0.1

        x = np.random.multivariate_normal(np.zeros(D), np.eye(D), N)
        x = np.append(x, np.ones((x.shape[0],1)), axis = 1)
        w = np.random.multivariate_normal(np.zeros(D+1), np.eye(D+1), M)
        r = np.random.uniform(size = N)
        r = np.ones(N)
        
        y = np.random.normal(x @ w.T, np.sqrt(s2)).astype(np.float32)
        x = x[:, :-1].astype(np.float32)
        x = (x - x.mean(axis = 0))
        xtr = x[0:int(N*0.8)]
        xte = x[int(N*0.8):]
        ytr = y[0:int(N*0.8)]
        yte = y[int(N*0.8):]
        
        model = gp_reg(K = K, D = D, M = M, s2 = s2)
        start = time.time()
        q_vec = model.fit(xtr, ytr, epoch = 3000)
        end = time.time()
        print('Fit time:', end- start)
        
        plt.plot(q_vec)
        plt.show()
        
        yp = model.predict(xte, xtr, ytr).numpy()
        if (D == 1) and (M == 1):
            yptr = model.predict(xtr, xtr, ytr).numpy()
            plt.scatter(xtr, ytr)
            plt.plot(xtr[np.argsort(xtr[:,0])], yptr[np.argsort(xtr[:,0])])
            plt.show()
        
        
        print('GP MSE:', np.mean((yp - yte)**2))
        
        model_reg = LinearRegression().fit(x, y)
        yp = model_reg.predict(xte)
        print('LinReg MSE:', np.mean((yp - yte)**2))
        
        
            
        
        
        

   