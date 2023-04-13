# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Mathias)s
"""
#Based on 8.3.1

from matplotlib.pyplot import figure, show, title
from scipy.io import loadmat
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
from sklearn import model_selection
import numpy as np
import torch

#Importing data
from Loading_data import * 

#The X data is the features
X = Y2[:,:]
#y = glass_type.squeeze() #All 7 glass types
y = BinaryGlassType.squeeze() #Binary class problem. 1 = Window glass 2 = Non windowd glass 
#Define number of K folds
K1 = 5

#Partionining data into a test and training set
CV = model_selection.KFold(K1, shuffle=True)


# Load Matlab data file and extract variables of interest


# =============================================================================
# mat_data = loadmat('../Data/synth1.mat')
# X = mat_data['X']
# X = X - np.ones((X.shape[0],1)) * np.mean(X,0)
# X_train = mat_data['X_train']
# X_test = mat_data['X_test']
# y = mat_data['y'].squeeze()
# y_train = mat_data['y_train'].squeeze()
# y_test = mat_data['y_test'].squeeze()
# 
# attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
# classNames = [name[0][0] for name in mat_data['classNames']]
# =============================================================================

N, M = X.shape
C = len(ClassNames)+1
classNames = ClassNames
attributeNames = AttributeNames
#%% Model fitting and prediction

for (k1, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    # Extract training and test set of the CV fold
    X_train = X[train_index]
    y_train = y[train_index].squeeze()
    X_test = X[test_index]
    y_test = y[test_index].squeeze()
    
    # Define the model structure
    n_hidden_units = 9 # number of hidden units in the signle hidden layer
    model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                                torch.nn.ReLU(), # 1st transfer function
                                # Output layer:
                                # H hidden units to C classes
                                # the nodes and their activation before the transfer 
                                # function is often referred to as logits/logit output
                                torch.nn.Linear(n_hidden_units, C), # C logits
                                # To obtain normalised "probabilities" of each class
                                # we use the softmax-funtion along the "class" dimension
                                # (i.e. not the dimension describing observations)
                                torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit output
                                )
    # Since we're training a multiclass problem, we cannot use binary cross entropy,
    # but instead use the general cross entropy loss:
    loss_fn = torch.nn.CrossEntropyLoss()
    # Train the network:
    net, _, _ = train_neural_net(model, loss_fn,
                                 X=torch.tensor(X_train, dtype=torch.float),
                                 y=torch.tensor(y_train, dtype=torch.long),
                                 n_replicates=3,
                                 max_iter=10000)
    # Determine probability of each class using trained network
    softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
    # Get the estimated class as the class with highest probability (argmax on softmax_logits)
    y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy() 
    # Determine errors
    e = (y_test_est != y_test)
    print('Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(sum(e),len(e)))
    

print('Ran Exercise 8.3.1')