# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:31:48 2023

@author: Rita and Mathias
"""


import os
os.chdir('C:/Users/ritux/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DTU/1 6º Semester/1 3 02450 Machine Learning/Project 2/02450-Project-2')


#Importing data
from Loading_data import * 


# exercise 8.2.6
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

# Refractive Index - the feature we are trying predict
y = Y2[:,[0]]
#The X data is the rest of the features
X = Y2[:,1:]

N, M = X.shape
C = 2

# Normalize data
#X = stats.zscore(X)

# K1-fold crossvalidation
K1 = 5
# K2-fold crossvalidation
K2 = 5


hidden_units = np.array([1, 2, 3, 4, 5])
regul_lamdas = np.power(10.,np.arange(0,2,0.01))

#for each outer fold contains for both models
#the best tweaked paramter and its error
Table_Info= np.zeros((K1,2,2)) #outer_fold,model,[parameter,error]

#for within all inner fold of one outer fold for each models 
#the for each of the tested parameters stores the assocated error
Regul_Info= np.zeros((K2,len(regul_lamdas))) #inner_fold, parameter
ArtNN_Info= np.zeros((K2,len(hidden_units))) #inner_fold, parameter

CV_Inner = model_selection.KFold(K1, shuffle=True,random_state=1)
for (k1, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    
    print('\nStarting Crossvalidation Outer Fold: {0}/{1}'.format(k1+1,K1))  
    
    # extract training and test set of the Outer CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    CV_Outer = model_selection.KFold(K2, shuffle=True,random_state=1)
    for (k2, (train_index, test_index)) in enumerate(CV.split(X_train,y_train)): 
        print('\nStarting Crossvalidation Inner Fold: {0}/{1}'.format(k2+1,K1)) 

        for count,n_hidden_units in enumerate(h):
            
            print('Number of hidden units: {0}'.format(n_hidden_units))
            # Parameters for neural network classifier
            n_replicates = 5        # number of networks trained in each k-fold
            max_iter = 10000
            
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            
            


                