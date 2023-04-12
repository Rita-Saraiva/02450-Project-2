# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:31:48 2023

@author: Rita and Mathias
"""


#import os
#os.chdir('C:/Users/ritux/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DTU/1 6º Semester/1 3 02450 Machine Learning/Project 2/02450-Project-2')


####

#Missing Base line
#Mathias is working on implementing the baseline model
#Baseline model is equivalent to computing the mean of the y_train data and the predicting all y_test as y_train.mean()


#exercise 7_3_1 for setup I (section 11.3):
####


#Importing data
from Loading_data import * 


# exercise 8.2.6
import numpy as np
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, rlr_validate
from Suporting_Functions import RLR_and_ANN_validate

# Refractive Index - the feature we are trying predict
y = Y2[:,[0]]
#The X data is the rest of the features
X = Y2[:,1:]

N, M = X.shape



# K1-fold crossvalidation
K1 = 5
# K2-fold crossvalidation
K2 = 5

# Necessary parameters for the methods
n_replicates=3
max_iter=10000
hidden_units = np.array([1, 2, 3, 4, 5])
regul_lamdas = np.power(10.,np.arange(0,2,0.01))
weight = np.zeros((K1,8))


#for each outer fold contains for the three models
#the best tweaked paramter and its error
Table_Info= np.zeros((K1,2,2)) #outer_fold,model,[parameter,error]
Gen_Error_Table = np.zeros((K1,3)) #outer_fold,model

CV = model_selection.KFold(K1, shuffle=True)
for (k1, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    
    print('\n Crossvalidation Outer Fold: {0}/{1}'.format(k1+1,K1))  
    
    # extract training and test set of the Outer CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index].squeeze()
    
    print('\nCrossvalidation Inner Fold') 
   
    RLR_opt_val_err,RLR_opt_lambda,RLR_mean_w_vs_lambda,ANN_opt_val_err, ANN_opt_hunits = RLR_and_ANN_validate(X,y,regul_lamdas,hidden_units)
    Table_Info[k1,1,0]=RLR_opt_lambda; Table_Info[k1,1,1]=RLR_opt_val_err;
    Table_Info[k1,0,0]=ANN_opt_hunits; Table_Info[k1,0,1]=ANN_opt_val_err;
    
    print('\n Evaluation of RLR Outer_CV')
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train    
    
    lambdaI = RLR_opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # remove bias regularization
    weight[k1,:] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
    # Evaluate training and test performance  
    Gen_Error_Table[k1,1]=np.power(y_test-X_test @ weight[k1,:].T,2).mean(axis=0)
    
    print('\n Evaluation of ANN Outer_CV')  

    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, ANN_opt_hunits), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(ANN_opt_hunits, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
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
    Gen_Error_Table[k1,0]=mse
    
    
    
    print('\n Evaluation of baseline model Outer_CV')
    #Computing  and Storing basline model error 
    Gen_Error_Table[k1,2] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
        
    
## %%    
from tabulate import tabulate
Table=np.zeros((5,6))
Table[:,0]=np.arange(1,5+1).T
Table[:,1]=Table_Info[:,0,0]
Table[:,2]=Gen_Error_Table[:,0]
Table[:,3]=Table_Info[:,1,0]
Table[:,4]=Gen_Error_Table[:,1]
Table[:,5]=Gen_Error_Table[:,2]


Top=np.array([["Outer fold","ANN","","Linear","Regression","baseline"],["i","*h_i","Test^E_i","*Lambda_i","Test^E_i","Test^E_i"]])
print(tabulate(Table, headers=Top, tablefmt="fancy_grid", showindex="always"))
                