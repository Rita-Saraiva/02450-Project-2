# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:45:33 2023

@author: Rita
"""


import os
os.chdir('C:/Users/ritux/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DTU/1 6º Semester/1 3 02450 Machine Learning/Project 2/02450-Project-2/02450-Project-2/Final stuff')

#Importing data
from Loading_data import * 

#Importing packages
import sklearn.linear_model as lm
from sklearn import model_selection


#Importing packages
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np


#%%

def RLogR_5_validate(X,y,lambdas,cvf=5):
    
    CV = model_selection.KFold(cvf, shuffle=True)
    
    RLogR_5_train_error = np.empty((cvf,len(lambdas)))
    RLogR_5_test_error = np.empty((cvf,len(lambdas)))
    
    for (f, (train_index, test_index)) in enumerate(CV.split(X,y)):
        
        print('   Crossvalidation of Inner_CV: {0}/{1}'.format(f+1,cvf))
        print('    Logistic Regression')
        
        # Extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
        
        # Standardize the training and set set based on training set mean and std
        mu_train = np.mean(X_train, 0)
        sigma_train = np.std(X_train, 0)
        X_train = (X_train - mu_train) / sigma_train
        X_test = (X_test - mu_train) / sigma_train
        
               
        for count,Lambda in enumerate(lambdas):
            
            #print('\n Crossvalidation of {0} Lambda '.format(round(Lambda,5)))
            
            mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                           tol=1e-4, 
                                           penalty='l2', C=1/Lambda,
                                           max_iter=1000000)
            mdl.fit(X_train,y_train)
            
            y_train_est = mdl.predict(X_train)
        
            RLogR_5_train_error[f,count]=np.sum(y_train_est!=y_train) / len(y_train)
            
            
            y_test_est = mdl.predict(X_test)
        
            RLogR_5_test_error[f,count]=np.sum(y_test_est!=y_test) / len(y_test)
        
        
    
        
       
    print('\n  Calculating Error of Crossvalidation Fold') 
        
    RLogR_5_test_err_vs_lambda = np.mean(RLogR_5_test_error,axis=0)
    RLogR_5_opt_val_err = np.min(RLogR_5_test_err_vs_lambda)
    RLogR_5_opt_lambda = lambdas[np.argmin(RLogR_5_test_err_vs_lambda)]
    
    train_err_vs_lambda = np.mean(RLogR_5_train_error,axis=0)
    test_err_vs_lambda = np.mean(RLogR_5_test_error,axis=0)
    
    
       
    return RLogR_5_opt_lambda,RLogR_5_opt_val_err, train_err_vs_lambda,test_err_vs_lambda



#%%

# Type of Glass - the class we are trying predict
y = glass_type.squeeze()
#BinaryGlassType
#The elements' presence and refractive index
X = Y2

N, M = X.shape

# K1-fold crossvalidation
K1 = 5
# K2-fold crossvalidation
K2 = 5


# Fit multinomial logistic regression model
regularization_strength = np.linspace(0.0001,2.5,1000)

Opt_Lambda=np.empty((K1,1))

# Initialize variables
Error_train = np.empty((K1,1))
Error_test = np.empty((K1,1))
Error_train_rlr = np.empty((K1,1))
Error_test_rlr = np.empty((K1,1))
Error_train_nofeatures = np.empty((K1,1))
Error_test_nofeatures = np.empty((K1,1))
mu = np.empty((K1, M-1))
sigma = np.empty((K1, M-1))

CV = model_selection.KFold(K1, shuffle=True)
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    RLogR_5_opt_lambda,RLogR_5_opt_val_err, train_err_vs_lambda,test_err_vs_lambda = RLogR_5_validate(X,y,regularization_strength)
    
    Opt_Lambda[k]=RLogR_5_opt_lambda
    
    print('\n Evaluation of RLogR Outer_CV') 
   
    # Standardize the training and set set based on training set mean and std
    mu_train = np.mean(X_train, 0)
    sigma_train = np.std(X_train, 0)
    X_train = (X_train - mu_train) / sigma_train
    X_test = (X_test - mu_train) / sigma_train
    
    
    mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                   tol=1e-4, 
                                   penalty='l2', C=1/RLogR_5_opt_lambda)
    mdl.fit(X_train,y_train)
    y_train_est = mdl.predict(X_train)
    y_test_est = mdl.predict(X_test)
    
    # Compute mean squared error without regularization
    Error_train[k] = np.sum(y_train_est!=y_train) / len(y_train)
    Error_test[k] = np.sum(y_test_est!=y_test) / len(y_test)
    
    print('\n Evaluation of baseline model Outer_CV')
    
    class_count=np.array([0]*7)
    for element in y_train:
        case = {
            1: 'class_count[0]+=1',
            2: 'class_count[1]+=1',
            3: 'class_count[2]+=1',
            4: 'class_count[3]+=1',
            5: 'class_count[4]+=1',
            6: 'class_count[5]+=1',
            7: 'class_count[6]+=1',
        }
        statement = case[element]
        exec(statement)
    base_max=np.argmax(class_count)+1
    
    base_true =0
    base_false =0
    base_pred=np.array([base_max]*len(y_test))
    base_error=(base_pred!=y_test)    
    
    
    figure(k, figsize=(6,8))
    title('Optimal lambda: {0}'.format((Opt_Lambda[k])))
    loglog(regularization_strength,train_err_vs_lambda.T*100,'b.-',regularization_strength,test_err_vs_lambda.T*100,'r.-')
    xlabel('Regularization factor')
    ylabel('Squared Error')
    legend(['Train error','Validation error'])
    grid()

    
