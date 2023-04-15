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


def RLR_and_ANN_validate(X,y,lambdas,h_units,cvf=5,n_replicates = 3,max_iter = 10000):
    
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    w = np.empty((M,cvf,len(lambdas)))
        
    ANN_test_error = np.empty((cvf,len(h_units)))
    
    RLR_test_error = np.empty((cvf,len(lambdas)))
    
    for (f, (train_index, test_index)) in enumerate(CV.split(X,y)):
                
        print('\nCrossvalidation of RLR Inner_CV: {0}/{1}'.format(f+1,cvf))
        
        X_train = X[train_index]
        y_train = y[train_index].squeeze()
        X_test = X[test_index]
        y_test = y[test_index].squeeze()
        
        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
               
        for count,Lambda in enumerate(lambdas):
            #print('Crossvalidation of {0} Regularization  Lambda'.format(round(Lambda,5)))
            
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = Lambda * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,f,count] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            RLR_test_error[f,count] = np.power(y_test-X_test @ w[:,f,count].T,2).mean(axis=0)
        
        
        
        print('\nCrossvalidation of ANN Inner_CV: {0}/{1}'.format(f+1,cvf))
        
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
            
            ANN_test_error[f,count]=mse # store error rate for current CV fold 
           
        
    print('\nCalculating Error of Crossvalidation Fold') 
        
    RLR_test_err_vs_lambda = np.mean(RLR_test_error,axis=0)
    RLR_opt_val_err = np.min(RLR_test_err_vs_lambda)
    RLR_opt_lambda = lambdas[np.argmin(RLR_test_err_vs_lambda)]
    RLR_mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    ANN_test_err_vs_hunits = np.mean(ANN_test_error,axis=1)
    ANN_opt_val_err = np.min(ANN_test_err_vs_hunits)
    ANN_opt_hunits = h_units[np.argmin(ANN_test_err_vs_hunits)]
    
    return RLR_opt_val_err,RLR_opt_lambda,RLR_mean_w_vs_lambda,ANN_opt_val_err, ANN_opt_hunits




import sklearn.linear_model as lm
from sklearn import tree



def RLogR_and_CT_validate(X,y,lambdas,treecomplex,cvf=5):
    
    CV = model_selection.KFold(cvf, shuffle=True)
    
    CT_test_error = np.empty((cvf,len(treecomplex)))
    RLogR_test_error = np.empty((cvf,len(lambdas)))
    
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
                                           tol=1e-4, random_state=1, 
                                           penalty='l2', C=1/Lambda)
            mdl.fit(X_train,y_train)
            y_test_est = mdl.predict(X_test)
        
            RLogR_test_error[f,count]=np.sum(y_test_est!=y_test) / len(y_test)
        
        print('    Classification Tree')
        
        for count, tc in enumerate(treecomplex):
            
            #print('\n Crossvalidation of {0} Tree Complexity '.format(round(alpha,5)))
            
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=tc)
            dtc = dtc.fit(X_train,y_train)
        
            # Evaluate classifier's misclassification rate over train/test data
            y_est_test = np.asarray(dtc.predict(X_test),dtype=int)
            
            CT_test_error[f,count]= sum(y_est_test != y_test) / y_est_test.shape[0] # store error rate for current CV fold 
        
    print('\n  Calculating Error of Crossvalidation Fold') 
        
    RLogR_test_err_vs_lambda = np.mean(RLogR_test_error,axis=0)
    RLogR_opt_val_err = np.min(RLogR_test_err_vs_lambda)
    RLogR_opt_lambda = lambdas[np.argmin(RLogR_test_err_vs_lambda)]
        
    CT_test_err_vs_TC = np.mean(CT_test_error,axis=1)
    CT_opt_val_err = np.min(CT_test_err_vs_TC)
    CT_opt_tc = treecomplex[np.argmin(CT_test_err_vs_TC)]
    
    return RLogR_opt_val_err,RLogR_opt_lambda,CT_opt_val_err, CT_opt_tc




#%%

#import sklearn.linear_model as lm
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder


def RLogR_and_NB_validate(X,y,lambdas,alphas,cvf=5):
    
    CV = model_selection.KFold(cvf, shuffle=True)
    
    NB_test_error = np.empty((cvf,len(alphas)))
    RLogR_test_error = np.empty((cvf,len(lambdas)))
    
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
                                           tol=1e-4, random_state=1, 
                                           penalty='l2', C=1/Lambda)
            mdl.fit(X_train,y_train)
            y_test_est = mdl.predict(X_test)
        
            RLogR_test_error[f,count]=np.sum(y_test_est!=y_test) / len(y_test)
        
        print('    Naive Bayes')
        
        XNB = OneHotEncoder().fit_transform(X=X)
        
        # extract training and test set for current CV fold
        XNB_train = XNB[train_index,:]
        yNB_train = y[train_index]
        XNB_test = XNB[test_index,:]
        yNB_test = y[test_index]
        
        for count,alpha in enumerate(alphas):
            
            #print('\n Crossvalidation of {0} Alpha '.format(round(alpha,5)))
            
            nb_classifier = MultinomialNB(alpha=alpha,
                                          fit_prior=True)
            nb_classifier.fit(XNB_train, yNB_train)
            yNB_est_prob = nb_classifier.predict_proba(XNB_test)
            yNB_est = np.argmax(yNB_est_prob,1)

            NB_test_error[f,count]=np.sum(yNB_est!=yNB_test,dtype=float)/yNB_test.shape[0] # store error rate for current CV fold 
        
    print('\n  Calculating Error of Crossvalidation Fold') 
        
    RLogR_test_err_vs_lambda = np.mean(RLogR_test_error,axis=0)
    RLogR_opt_val_err = np.min(RLogR_test_err_vs_lambda)
    RLogR_opt_lambda = lambdas[np.argmin(RLogR_test_err_vs_lambda)]
        
    NB_test_err_vs_alpha = np.mean(NB_test_error,axis=1)
    NB_opt_val_err = np.min(NB_test_err_vs_alpha)
    NB_opt_alphas = alphas[np.argmin(NB_test_err_vs_alpha)]
        
    
    return RLogR_opt_val_err,RLogR_opt_lambda,NB_opt_val_err, NB_opt_alphas

