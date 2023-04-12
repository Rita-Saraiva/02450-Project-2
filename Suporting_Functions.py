# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:27:46 2023

@author: Rita and Mathias
"""

"""
Based on the 'rlr_validate' funtion from the package give
"""


import torch
import numpy as np
from sklearn import model_selection
from toolbox_02450 import train_neural_net

def ANN_validate(X,y,h_units,cvf=5,n_replicates = 3):
    
    CV = model_selection.KFold(cvf, shuffle=True,random_state=1)
    M = X.shape[1]
    w = np.empty((M,cvf,len(h_units)))
    
    # Parameters for neural network classifier
    #n_replicates = 3        # number of networks trained in each k-fold
    max_iter = 10000
    
    train_error = np.empty((cvf,len(h_units)))
    test_error = np.empty((cvf,len(h_units)))
    
    for (f, (train_index, test_index)) in enumerate(CV.split(X,y)):
        
        print('\nCrossvalidation Inner Fold: {0}/{1}'.format(f+1,cvf)) 
        
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        for count,n_hidden_units in enumerate(h_units):
            
            print('\n Crossvalidation of {0} Hidden Units '.format(n_hidden_units)) 
            
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            
            # Extract training and test set for current CV fold, convert to tensors
            X_train = torch.Tensor(X[train_index,:])
            y_train = torch.Tensor(y[train_index])
            X_test = torch.Tensor(X[test_index,:])
            y_test = torch.Tensor(y[test_index])
            
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)            
            # Determine estimated class labels for test set
            y_test_est = net(X_test)
            # Determine errors and errors
            se = (y_test_est.float()-y_test.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
            
            train_error[f,count]=final_loss
            test_error[f,count]=mse # store error rate for current CV fold 
    
    opt_val_err = np.min(np.mean(test_error,axis=1))
    opt_hunits = h_units[np.argmin(np.mean(test_error,axis=1))]
    train_err_vs_hunits = np.mean(train_error,axis=1)
    test_err_vs_hunits = np.mean(test_error,axis=1)
    mean_w_vs_hunits = np.squeeze(np.mean(w,axis=1))
    
    return opt_val_err, opt_hunits, mean_w_vs_hunits, train_err_vs_hunits,test_err_vs_hunits

