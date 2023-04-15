# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 13:47:14 2023

@author: Rita, Jonas and Mathias
"""

#import os
#os.chdir('C:/Users/ritux/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DTU/1 6º Semester/1 3 02450 Machine Learning/Project 2/02450-Project-2')


#Importing data
from Loading_data import * 

#Importing packages
import numpy as np
import sklearn.linear_model as lm
from sklearn import model_selection, tree
from Suporting_Functions import RLogR_and_CT_validate


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

# Necessary parameters for the methods

# Tree complexity parameter - constraint on maximum depth
treecomplexity = np.arange(2, 21, 1)
# Fit multinomial logistic regression model
regularization_strength = np.linspace(0.01,10)


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
    
    RLogR_opt_val_err,RLogR_opt_lambda,CT_opt_val_err, CT_opt_tc = RLogR_and_CT_validate(X,y,regularization_strength,treecomplexity,cvf=5)
    Table_Info[k1,1,0]=RLogR_opt_lambda; Table_Info[k1,1,1]=RLogR_opt_val_err;
    Table_Info[k1,0,0]=CT_opt_tc; Table_Info[k1,0,1]=CT_opt_val_err;
    
    print('\n Evaluation of RLogR Outer_CV') 
   
    # Standardize the training and set set based on training set mean and std
    mu_train = np.mean(X_train, 0)
    sigma_train = np.std(X_train, 0)
    X_train = (X_train - mu_train) / sigma_train
    X_test = (X_test - mu_train) / sigma_train
    
    
    mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                   tol=1e-2, 
                                   penalty='l2', C=1/RLogR_opt_lambda)
    mdl.fit(X_train,y_train)
    y_test_est = mdl.predict(X_test)


    Gen_Error_Table[k1,1] = np.sum(y_test_est!=y_test) / len(y_test)
    
    
    print('\n Evaluation of NB Outer_CV')  
    
    # Fit decision tree classifier, Gini split criterion, different pruning levels
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=CT_opt_tc)
    dtc = dtc.fit(X_train,y_train)

    # Evaluate classifier's misclassification rate over train/test data
    y_est_test = np.asarray(dtc.predict(X_test),dtype=int)
        
    Gen_Error_Table[k1,0] = sum(y_est_test != y_test) / y_est_test.shape[0]
    
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

Top=np.array([["Outer fold","Class Tree","","Logistic","Regression","baseline"],
              ["i        ","*h_i","Test^E_i","*Lambda_i ","Test^E_i ","Test^E_i"]])

Table=np.zeros((K1,6))
Table[:,0]=np.arange(1,K1+1).T
Table[:,1]=Table_Info[:,0,0]
Table[:,2]=Gen_Error_Table[:,0]
Table[:,3]=Table_Info[:,1,0]
Table[:,4]=Gen_Error_Table[:,1]
Table[:,5]=Gen_Error_Table[:,2]

print(Top[0],'\n',Top[1],'\n',Table)
