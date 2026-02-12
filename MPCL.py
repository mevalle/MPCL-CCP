import numpy as np
import cvxpy as cp
import dccp
import matplotlib.pyplot as plt
import time

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.cluster import kmeans_plusplus
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import tensorflow as tf
import tensorflow.keras as keras

import scipy.sparse as sp
from scipy.optimize import linprog

# Auxiliary function:
def hs(Z,w):
  W = np.tile(w,(Z.shape[0],1))
  return np.max(W+Z,axis=1)


#############################################################################
#                               MPCL-Greedy
############################################################################# 

def get_intercept(X, y):
    w = list()
    for i in range(2):
        ind = (y == i)
        w.append(
            np.hstack([-np.min(X[ind, :], axis=0), np.max(X[ind, :], axis=0)]))
    return np.minimum(w[0], w[1])

def get_boxes(X, y, w, label):
    boxes = list()
    n = X.shape[1]
    for i in range(n):
        try:
            ind = np.logical_and((y == label), (X[:, i] < -w[i]))
            boxes.append(
                np.hstack([-np.min(X[ind, :], axis=0), np.max(X[ind, :], axis=0)]))
        except:
            pass

        try:
            ind = np.logical_and((y == label), (X[:, i] > w[i+n]))
            boxes.append(
                np.hstack([-np.min(X[ind, :], axis=0), np.max(X[ind, :], axis=0)]))
        except:
            pass
    return boxes


def get_inside(X, y, w):
    n = X.shape[1]
    ind = np.min(np.minimum(X+w[0:n],-X+w[n:]),axis=1)>0
    return X[ind], y[ind]

class MPCL_Greedy(BaseEstimator, ClassifierMixin):
    
    def __init__(self, verbose = False, myInf = 1.e+10):
        self.verbose = verbose
        self.myInf = 1.e+10 
    
    def fit(self, Xtr, ytr):
        self.Nclasses_ = np.size(np.unique(ytr))
        self.le = preprocessing.LabelEncoder()
        self.le.fit(ytr)
        self.classes_ = np.unique(self.le.transform(ytr))
        ytr = self.le.transform(ytr)
        self.boxes_ = [[] for i in range(self.Nclasses_)]
        
        self.dim_X_ = np.min(Xtr, axis=0) != np.max(Xtr, axis=0)
        Xtr = Xtr[:, self.dim_X_]
        
        for label in self.classes_:
            self.fit_class(Xtr, ytr, label)
        return self

    def fit_class(self, Xtr, ytr, label):
        ytrl = (ytr == label)
        if (Xtr[ytrl].shape[0] == 0):
            return self
        
        K, N = Xtr.shape
        
        if np.unique(ytrl).shape[0] > 1:
        
            w = get_intercept(Xtr, ytrl)
            [self.boxes_[label].append(a) for a in get_boxes(Xtr,ytr,w,label)]

            Xi, yi = get_inside(Xtr,ytr,w)
            return self.fit_class(Xi, yi, label)
        else:
            w = np.hstack([-np.min(Xtr, axis=0), np.max(Xtr, axis=0)])
            self.boxes_[ytr[0]].append(w)
        return self
            
    
    def decision_function(self,X, label):

        X = X[:, self.dim_X_]
        X = np.hstack([X,-X])
        if self.boxes_[label] == []:
            return np.ones(X.shape[0],) * -1.e+12
        
        return np.max(np.vstack([np.min(X+w, axis=1) for w in self.boxes_[label]]),axis=0)
    
    def predict(self,X):
        Y = np.vstack([self.decision_function(X,label) for label in self.classes_])
        pred = np.argmax(Y,axis=0)
        return self.le.inverse_transform(pred)
        
        
class MPCL_Module(keras.layers.Layer):
    def __init__(self, K = 2, # Number of hyperboxes
                 ):
        super().__init__()
        self.K = K

    def build(self, input_shape):
        self.boxes = [self.add_weight(
            shape=(2*input_shape[-1],),
            initializer="random_uniform",
            trainable=True,
        ) for _ in range(self.K)]

    def call(self, inputs):
        Z = keras.layers.Concatenate(axis=1)([inputs,-inputs])
        return tf.expand_dims(tf.reduce_max(tf.stack([tf.reduce_min(Z-tf.expand_dims(w, 0),axis=1) for w in self.boxes],axis=1),axis=1),1)
        

#############################################################################
#                               MPCL-Adam
############################################################################# 
#
# Tensorflow implementation - Referred to as MPCL-Adam in the paper but you can change the optimizer! 
#
class MPCL_Adam(BaseEstimator, ClassifierMixin):
    def __init__(self, K=2,  # Number of hyperboxes per class.
                 verbose = False,
                 solver=keras.optimizers.Adam()
                 ):
        self.verbose = verbose
        self.solver = solver
        self.K = K

    def fit(self, Xtr, ytr, epochs = 100, batch_size = 32):
        start_time = time.time()

        # Check that X and y have correct shape
        Xtr, ytr = check_X_y(Xtr, ytr)

        self.Nclasses_ = np.size(np.unique(ytr))
        self.le = preprocessing.LabelEncoder()
        self.le.fit(ytr)
        self.classes_ = np.unique(self.le.transform(ytr))
        ytr = self.le.transform(ytr)
        self.boxes_ = [[] for i in range(self.Nclasses_)]

        self.dim_X_ = np.min(Xtr, axis=0) != np.max(Xtr, axis=0)
        Xtr = Xtr[:, self.dim_X_]

        history_list = []
        for label in self.classes_:
            history = self.fit_class(Xtr, ytr, label, epochs = epochs, batch_size = batch_size)
            history_list.append(history)

        if self.verbose == True:
            print("\nTime to train: %2.2f seconds." % (time.time() - start_time))
        return history_list

    def fit_class(self, X, y, label, epochs = 100, batch_size = 32):
        start_time_class = time.time()
        keras.backend.clear_session()

        ind1 = np.where(y==label)
        # Initialize the weights with kmeans_plusplus
        BBpx, inds = kmeans_plusplus(X[ind1],self.K)
        W = tf.concat([np.hstack([a,-a]).reshape(1,-1) for a in BBpx],axis=0)

        # Define the tensorflow MPCL models
        model = keras.models.Sequential([
            keras.layers.Input(shape=(X.shape[1],)),
            MPCL_Module(K=self.K),
        ],name="MPCL_class_"+str(label))

        # Initialize the weights using kmeans++
        for i,w in enumerate(model.weights):
            w.assign(W[i])

        # Compile and fit the model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # Assuming the model output is logits
              metrics=["Accuracy"]
              )

        history = model.fit(X, y==label, epochs=epochs, batch_size=batch_size, verbose=self.verbose)

        # Get the weights of the trained model
        self.boxes_[label] = np.array([w.numpy() for w in model.weights])

        if self.verbose == True:
            print("\nTime to train class %s: %2.2f seconds." % (str(label),time.time() - start_time_class))

        return history

    def decision_function(self,X, label):
        # Check is fit had been called
        check_is_fitted(self,attributes="boxes_")

        # Input validation
        X = check_array(X)
        
        Z = np.hstack([X,-X])
        return np.max(np.vstack([np.min(Z - np.tile(w,(Z.shape[0],1)),axis=1) for w in self.boxes_[label]]),axis=0)
    
    def predict(self,X):
        Y = np.vstack([self.decision_function(X,label) for label in self.classes_])
        pred = np.argmax(Y,axis=0)
        return self.le.inverse_transform(pred)

    def show_boxes(self,X,y):
        y = self.le.transform(y)
        tab10_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        for i,label in enumerate(self.classes_):
            plt.scatter(X[y==label,0],X[y==label,1],label="class "+str(label),c=tab10_colors[i%10])
            for w in self.boxes_[label]:
                ptsx = np.array([w[0],-w[2],-w[2],w[0]])
                ptsy = np.array([w[1],w[1],-w[3],-w[3]])
                plt.fill(ptsx,ptsy,c=tab10_colors[i%10],alpha=0.2)
        plt.legend()
        plt.grid()
        
        
#############################################################################
#                               MPCL-DCCP
############################################################################# 
# 
# This implementation is not recommended! Use MPCL-CCP instead!
#
class MPCL_DCCP(BaseEstimator, ClassifierMixin):
    def __init__(self, K=2,  # Number of hyperboxes per class.
                 gamma = 1.e-2,
                 verbose = False, 
                 solver="CLARABEL"):
        self.verbose = verbose
        self.solver = solver
        self.K = K
        self.gamma = gamma
    
    def fit(self, Xtr, ytr):
        start_time = time.time()
        
        # Check that X and y have correct shape
        Xtr, ytr = check_X_y(Xtr, ytr)
        
        self.Nclasses_ = np.size(np.unique(ytr))
        self.le = preprocessing.LabelEncoder()
        self.le.fit(ytr)
        self.classes_ = np.unique(self.le.transform(ytr))
        ytr = self.le.transform(ytr)
        self.boxes_ = [[] for i in range(self.Nclasses_)]
        
        self.dim_X_ = np.min(Xtr, axis=0) != np.max(Xtr, axis=0)
        Xtr = Xtr[:, self.dim_X_]
        
        obj_values = []
        for label in self.classes_:
            obj_values.append(self.fit_class(Xtr, ytr, label))
            
        if self.verbose == True:
            print("\nTime to train: %2.2f seconds." % (time.time() - start_time))
        return obj_values
    
    def Psi(self,w,Z):
        return cp.max(Z+cp.kron(cp.reshape(w,(1,-1),order='F'),np.ones((Z.shape[0],1))),axis=1)
    
    def f(self,W,Z):
        return cp.max(cp.vstack([cp.sum(cp.vstack([self.Psi(W[j],Z) for j in range(W.shape[0]) if j != k]),axis=0) for k in range(W.shape[0])]),axis=0)
    
    def g(self,W,Z):
        return cp.sum(cp.vstack([self.Psi(W[k],Z) for k in range(W.shape[0])]),axis=0)
    
    def fit_class(self, X, y, label):
        start_time_class = time.time()
        
        # Getting parameters and preparing auxiliary variables (Z0 and Z1)
        N = X.shape[1]
        ind0 = np.where(y!=label)
        ind1 = np.where(y==label)
        
        Z0 = np.hstack([-X[ind0],X[ind0]])
        Z1 = np.hstack([-X[ind1],X[ind1]])
        
        M0 = Z0.shape[0]
        M1 = Z1.shape[0]

        # DCCP Problem:
        
        # Define the variables of the optimization problem:
        a = cp.Variable((self.K,N))
        b = cp.Variable((self.K,N))
        V = cp.hstack([a,-b])
        xi0 = cp.Variable(M0,nonneg=True)
        xi1 = cp.Variable(M1,nonneg=True)

        # Defining the constraints:
        constraints = [a-b<=0]
        # Constraints for the class 0:
        for k in range(self.K):
            constraints.append(-xi0 <= self.Psi(V[k],Z0))
        # constraints.append(self.f(V,Z0)-xi0 <= self.g(V,Z0))
        
        # Constraings for the class 1:
        constraints.append(self.g(V,Z1)-xi1 <= self.f(V,Z1))
            
        # objective = cp.Minimize(cp.sum(xi0)/M0+cp.sum(xi1)/M1+self.gamma*cp.sum(b-a)/(self.K*max_box_length))
        objective = cp.Minimize(cp.sum(xi0)+cp.sum(xi1)+self.gamma*cp.sum(b-a))
        
        prob = cp.Problem(objective,constraints)
        prob.solve(method='dccp',verbose=self.verbose, solver=self.solver)
        
        self.boxes_[label] = V.value
        return np.array(prob.value)    
    
    def decision_function(self,X, label):
        # Check is fit had been called
        check_is_fitted(self,attributes="boxes_")

        # Input validation
        X = check_array(X)
        
        Z = np.hstack([X,-X])
        return np.max(np.vstack([np.min(Z - np.tile(w,(Z.shape[0],1)),axis=1) for w in self.boxes_[label]]),axis=0)
    
    def predict(self,X):
        Y = np.vstack([self.decision_function(X,label) for label in self.classes_])
        pred = np.argmax(Y,axis=0)
        return self.le.inverse_transform(pred)

    def show_boxes(self,X,y):
        y = self.le.transform(y)
        tab10_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        for i,label in enumerate(self.classes_):
            plt.scatter(X[y==label,0],X[y==label,1],label="class "+str(label),c=tab10_colors[i%10])
            for w in self.boxes_[label]:
                ptsx = np.array([w[0],-w[2],-w[2],w[0]])
                ptsy = np.array([w[1],w[1],-w[3],-w[3]])
                plt.fill(ptsx,ptsy,c=tab10_colors[i%10],alpha=0.2)
        plt.legend()
        plt.grid()

#############################################################################
#                               MPCL-CCP
############################################################################# 
#
# Implemented using scipy
#

class MPCL_CCP(BaseEstimator, ClassifierMixin):
    def __init__(self, K=2,  # Number of hyperboxes per class.
                 gamma = 1.e-2,
                 verbose = False,
                 random_state = None):
        self.verbose = verbose
        self.K = K
        self.gamma = gamma
        self.random_state = random_state

    def ComputeMatrices(self,W,Z0,Z1):
        # Compute dimensions
        N = Z0.shape[1]//2
        M0 = Z0.shape[0]
        M1 = Z1.shape[0]
        M = M0+M1
        num_vars = 2*N*self.K + M

        b = []

        # Define the row and column indexes and the corresponding values;
        rows = []
        cols = []
        datas = []

        #
        # Generate the matrix for the negative samples
        #
        ind = np.arange(M0)
        for k in range(self.K):
          # Term of the hyperboxes;
          Ws = np.tile(W[k], (M0, 1))
          Js = np.argmax(Ws+Z0,axis=1)
          rows.append(k*M0+ind)
          cols.append(k*2*N+Js)
          datas.append(-np.ones(M0))

          # Term for xi
          rows.append(k*M0+ind)
          cols.append(2*N*self.K+ind)
          datas.append(-np.ones(M0))

          # Right hand-side term
          b.append(Z0[ind,Js])

        #
        # Generate the matrix of the positive samples
        #

        # Define the column indexes
        Ks = np.argmin(np.vstack([hs(Z1,W[k]) for k in range(self.K)]),axis=0)

        ind = np.arange(M1)
        for i in range(2*N):
          # Term with a
          rows.append(self.K*M0+i*M1+ind)
          cols.append(2*N*Ks+i)
          datas.append(np.ones(M1))

          # Term for xi
          rows.append(self.K*M0+i*M1+ind)
          cols.append(2*N*self.K+M0+ind)
          datas.append(-np.ones(M1))

        b.append(-Z1.T.flatten())

        #
        # Include the box constraints
        #
        ind = np.arange(N)
        for k in range(self.K):
          rows.append(self.K*M0+2*N*M1+k*N+ind)
          cols.append(ind+2*N*k)
          datas.append(np.ones(N))

          rows.append(self.K*M0+2*N*M1+k*N+ind)
          cols.append(ind+N+2*N*k)
          datas.append(np.ones(N))

        b.append(np.zeros(self.K*N))

        cols = np.hstack(cols)
        rows = np.hstack(rows)
        datas = np.hstack(datas)

        A = sp.csr_array((datas,(rows,cols)),shape=(self.K*M0+2*N*M1+self.K*N,num_vars))

        return A, np.hstack(b)

    def fit(self, Xtr, ytr, tau=1.e-4, it_max = 100):
        start_time = time.time()

        # Check that X and y have correct shape
        Xtr, ytr = check_X_y(Xtr, ytr)

        self.Nclasses_ = np.size(np.unique(ytr))
        self.le = preprocessing.LabelEncoder()
        self.le.fit(ytr)
        self.classes_ = np.unique(self.le.transform(ytr))
        ytr = self.le.transform(ytr)
        self.boxes_ = [[] for i in range(self.Nclasses_)]

        self.dim_X_ = np.min(Xtr, axis=0) != np.max(Xtr, axis=0)
        Xtr = Xtr[:, self.dim_X_]

        obj_values = []
        for label in self.classes_:
            obj_values.append(self.fit_class(Xtr, ytr, label, tau = tau, it_max = it_max))

        if self.verbose == True:
            print("\nTime to train: %2.2f seconds." % (time.time() - start_time))
        return obj_values

    def fit_class(self, X, y, label, tau=1.e-4, it_max = 100):
        start_time_class = time.time()

        # Getting parameters and preparing auxiliary variables (Z0 and Z1)
        N = X.shape[1]
        ind0 = np.where(y!=label)
        ind1 = np.where(y==label)

        Z0 = np.hstack([-X[ind0],X[ind0]])
        Z1 = np.hstack([-X[ind1],X[ind1]])

        M0 = Z0.shape[0]
        M1 = Z1.shape[0]
        M = M0+M1 # Total number of xi variables

        # max_box_length = np.sum(np.max(X,axis=0)-np.min(X,axis=0))

        # Initialize the weights with kmeans_plusplus
        BBpx, inds =kmeans_plusplus(X[ind1],self.K,random_state=self.random_state)
        W = np.array([np.hstack([a,-a]) for a in BBpx])

        # Compute the initial objective value
        xi0 = np.maximum(0,np.max(np.vstack([np.min(Z0 - np.tile(w,(Z0.shape[0],1)),axis=1) for w in W]),axis=0))
        xi1 = np.maximum(0,-np.max(np.vstack([np.min(Z1 - np.tile(w,(Z1.shape[0],1)),axis=1) for w in W]),axis=0))
        obj_values = [np.sum(xi0)+np.sum(xi1)]

        # Objetive vector and inbounds values
        c = np.hstack([-self.gamma*np.ones(2*N*self.K),np.ones(M0+M1)])
        bounds_list = [(None, None)] * (2*N*self.K) + [(0.0, None)] * M

        # Initialize the recursive process:
        Diff_Objective = tau+1
        it = 0

        while (it<=it_max) and (Diff_Objective>tau):
            it += 1

            A,b = self.ComputeMatrices(W,Z0,Z1)
            sol = linprog(c, A_ub = A, b_ub = b, A_eq = None, b_eq = None, bounds=bounds_list, method="highs")

            obj_values.append(sol.fun)
            Diff_Objective = np.abs(obj_values[-2]-obj_values[-1])
            W = sol.x[:2*N*self.K].reshape(self.K,2*N)

            if self.verbose == True:
                print("=====")
                print("Conclude iteration %d of %d." % (it,it_max))
                print("Objective found:",obj_values[-1])
                print("The difference in the objective is ",Diff_Objective)

        self.boxes_[label] = W
        if it>it_max:
            print("Warning: Reached the maximum number of iterations to find the boxes of class %s." % str(label))
        if self.verbose == True:
            print("\nTime to train class %s: %2.2f seconds." % (str(label),time.time() - start_time_class))
        return np.array(obj_values)

    def decision_function(self,X, label):
        # Check is fit had been called
        check_is_fitted(self,attributes="boxes_")

        # Input validation
        X = check_array(X)

        Z = np.hstack([X,-X])
        return np.max(np.vstack([np.min(Z - np.tile(w,(Z.shape[0],1)),axis=1) for w in self.boxes_[label]]),axis=0)

    def predict(self,X):
        Y = np.vstack([self.decision_function(X,label) for label in self.classes_])
        pred = np.argmax(Y,axis=0)
        return self.le.inverse_transform(pred)

    def show_boxes(self,X,y):
        y = self.le.transform(y)
        tab10_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        for i,label in enumerate(self.classes_):
            plt.scatter(X[y==label,0],X[y==label,1],label="class "+str(label),c=tab10_colors[i%10])
            for w in self.boxes_[label]:
                ptsx = np.array([w[0],-w[2],-w[2],w[0]])
                ptsy = np.array([w[1],w[1],-w[3],-w[3]])
                plt.fill(ptsx,ptsy,c=tab10_colors[i%10],alpha=0.3)
        plt.legend()
        plt.grid()
