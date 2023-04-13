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
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from Suporting_Functions import RLogR_and_NB_validate



# Refractive Index - the feature we are trying predict
y = glass_type.squeeze()
#BinaryGlassType
#The X data is the rest of the features
X = Y2

N, M = X.shape

# K1-fold crossvalidation
K1 = 5
# K2-fold crossvalidation
K2 = 5

# Necessary parameters for the methods

# Naive Bayes classifier parameters
alphas = np.linspace(0.01,1.2,500) # pseudo-count, additive parameter (Laplace correction if 1.0 or Lidtstone smoothing otherwise)
fit_prior = True   # uniform prior (change to True to estimate prior from data)

# Fit multinomial logistic regression model
regularization_strength = np.linspace(0.1,10,500)


#for each outer fold contains for the three models
#the best tweaked paramter and its error
Table_Info= np.zeros((K1,2,2)) #outer_fold,model,[parameter,error]
Gen_Error_Table = np.zeros((K1,3)) #outer_fold,model


CV = model_selection.KFold(K1, shuffle=True)
for (k1, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    
    print('\nCrossvalidation Outer Fold: {0}/{1}'.format(k1+1,K1))

    # Extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    print('\n Crossvalidation Inner Fold') 
    
    RLogR_opt_val_err,RLogR_opt_lambda,NB_opt_val_err,NB_opt_alphas = RLogR_and_NB_validate(X_train,y_train,regul_lambdas,alphas,cvf=K2)
    Table_Info[k1,1,0]=RLogR_opt_lambda; Table_Info[k1,1,1]=RLogR_opt_val_err;
    Table_Info[k1,0,0]=NB_opt_alphas; Table_Info[k1,0,1]=NB_opt_val_err;
    
    print('\n Evaluation of RLogR Outer_CV') 
   
    # Standardize the training and set set based on training set mean and std
    mu_train = np.mean(X_train, 0)
    sigma_train = np.std(X_train, 0)
    X_train = (X_train - mu_train) / sigma_train
    X_test = (X_test - mu_train) / sigma_train
    
    regularization_strength = RLogR_opt_lambda
    
    
    mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                   tol=1e-4, random_state=1, 
                                   penalty='l2', C=1/regularization_strength)
    mdl.fit(X_train,y_train)
    y_test_est = mdl.predict(X_test)


    Gen_Error_Table[k1,1] = np.sum(y_test_est!=y_test) / len(y_test)
    
    
    print('\n Evaluation of NB Outer_CV')  
    
    XNB = OneHotEncoder().fit_transform(X=X)
    
    # extract training and test set for current CV fold
    XNB_train = XNB[train_index,:]
    yNB_train = y[train_index]
    XNB_test = XNB[test_index,:]
    yNB_test = y[test_index]
    
    #print('\n Crossvalidation of {0} Alpha '.format(round(alpha,5)))
    nb_classifier = MultinomialNB(alpha=NB_opt_alphas,
                                  fit_prior=True)
    nb_classifier.fit(XNB_train, yNB_train)
    yNB_est_prob = nb_classifier.predict_proba(XNB_test)
    yNB_est = np.argmax(yNB_est_prob,1)
    
    Gen_Error_Table[k1,0] = np.sum(yNB_est!=yNB_test,dtype=float)/yNB_test.shape[0]
    
    
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
    
    #print(f'Error for baseline at Fold {k1+1} is {base_error.mean()} %')
    
    Gen_Error_Table[k1,2] = base_error.mean()

    

print('\nEnd of Cross-Validation') 


Table=np.zeros((K1,6))
Table[:,0]=np.arange(1,K1+1).T
Table[:,1]=Table_Info[:,0,0]
Table[:,2]=Gen_Error_Table[:,0]
Table[:,3]=Table_Info[:,1,0]
Table[:,4]=Gen_Error_Table[:,1]
Table[:,5]=Gen_Error_Table[:,2]