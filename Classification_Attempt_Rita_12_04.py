# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:30:57 2023

@author: Rita
"""
import os
os.chdir('C:/Users/ritux/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DTU/1 6º Semester/1 3 02450 Machine Learning/Project 2/02450-Project-2')


#Importing data
from Loading_data import * 

#Importing packages
import numpy as np
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, rlr_validate
#from Suporting_Functions import RLogR_and_NB_validate





from sklearn.naive_bayes import MultinomialNB

def RLogR_and_NB_validate(X,y,lambdas,alphas,cvf=5,n_replicates = 3):
    
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    w = np.empty((M,cvf,len(lambdas)))
    
    NB_test_error = np.empty((cvf,len(alphas)))
    RLogR_test_error = np.empty((cvf,len(lambdas)))
    
    for (f, (train_index, test_index)) in enumerate(CV.split(X,y)):
        
        
        print('\nCrossvalidation of RLogR Inner_CV: {0}/{1}'.format(f+1,cvf))
        
        #JONAS
               
        for count,Lambda in enumerate(lambdas):
            #JONAS
            pass
        
        
        
        print('\nCrossvalidation of NB Inner_CV: {0}/{1}'.format(f+1,cvf))
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
        
        for count,alpha in enumerate(alphas):
            
            print('\n Crossvalidation of {0} Alpha '.format(alpha)) 
            
            nb_classifier = MultinomialNB(alpha=alpha,
                                          fit_prior=True)
            nb_classifier.fit(X_train, y_train)
            y_est_prob = nb_classifier.predict_proba(X_test)
            y_est = np.argmax(y_est_prob,1)

            NB_test_error[f,count]=np.sum(y_est!=y_test,dtype=float)/y_test.shape[0] # store error rate for current CV fold 
        
        
    print('\nCalculating Error of Crossvalidation Fold') 
        
    RLogR_test_err_vs_lambda = np.mean(RLogR_test_error,axis=0)
    RLogR_opt_val_err = np.min(RLogR_test_err_vs_lambda)
    RLogR_opt_lambda = lambdas[np.argmin(RLogR_test_err_vs_lambda)]
    RLogR_mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
        
    NB_test_err_vs_alpha = np.mean(NB_test_error,axis=1)
    NB_opt_val_err = np.min(NB_test_err_vs_alpha)
    NB_opt_alphas = alphas[np.argmin(NB_test_err_vs_alpha)]
        
    
    return RLogR_opt_val_err,RLogR_opt_lambda,RLogR_mean_w_vs_lambda,NB_opt_val_err, NB_opt_alphas






















# Refractive Index - the feature we are trying predict
y = glass_type.squeeze()
#The X data is the rest of the features
X = Y2

N, M = X.shape


# K1-fold crossvalidation
K1 = 5
# K2-fold crossvalidation
K2 = 5

# Necessary parameters for the methods

# Naive Bayes classifier parameters
alpha = 1.0 # pseudo-count, additive parameter (Laplace correction if 1.0 or Lidtstone smoothing otherwise)
fit_prior = True   # uniform prior (change to True to estimate prior from data)

###############
regul_lamdas = np.power(10.,np.arange(0,2,0.01))
###############


#for each outer fold contains for the three models
#the best tweaked paramter and its error
Table_Info= np.zeros((K1,2,2)) #outer_fold,model,[parameter,error]
Gen_Error_Table = np.zeros((K1,3)) #outer_fold,model


CV = model_selection.KFold(K1, shuffle=True)
for (k1, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    
    print('Crossvalidation fold: {0}/{1}'.format(k+1,K1))

    # Extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    print('\nCrossvalidation Inner Fold') 
    
    RLogR_opt_val_err,RLogR_opt_lambda,RLogR_mean_w_vs_lambda,NB_opt_val_err, NB_opt_alphas = RLogR_and_NB_validate(X,y,regul_lamdas,alphas,cvf=5,n_replicates = 3)
    Table_Info[k1,1,0]=RLogR_opt_lambda; Table_Info[k1,1,1]=RLogR_opt_val_err;
    Table_Info[k1,0,0]=NB_opt_alphas; Table_Info[k1,0,1]=NB_opt_val_err;