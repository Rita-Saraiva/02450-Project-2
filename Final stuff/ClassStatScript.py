# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 13:33:49 2023

@author: Jonas
"""



from toolbox_02450 import *
import pickle
from sklearn import *
from toolbox_02450 import mcnemar
import numpy as np

# Load Info_Table from file
with open('y_true_class.pickle', 'rb') as f:
    y_true_class = pickle.load(f)
with open('y_hat_class.pickle', 'rb') as f:
    y_hat_class = pickle.load(f)    

y_hat_class_rlog=[]
y_hat_class_tree=[]
y_hat_class_base=[]

y_true_fix=[]

for i in range(len(y_hat_class)):
    
    new_array=(y_hat_class[i])
    new_true=(y_true_class[i])
    
    y_hat_class_rlog=np.append(y_hat_class_rlog,new_array[0])
    y_hat_class_tree=np.append(y_hat_class_tree,new_array[1])
    y_hat_class_base=np.append(y_hat_class_base,new_array[2])
    
    y_true_fix=np.append(y_true_fix,new_true)
del (new_array, new_true, y_hat_class, y_true_class, i)



alpha = 0.05

print("\n \n Comparing Logistic Regression and Decision Tree\n")
[thetahat_rlog_tree, CI_rlog_tree, p_rlog_tree] = mcnemar(y_true_fix, y_hat_class_rlog, y_hat_class_tree, alpha=alpha)
print("theta_hat = "+str(thetahat_rlog_tree))

print("\n \n Comparing Logistic Regression and Baseline\n")
[thetahat_rlog_base, CI_rlog_base, p_rlog_base] = mcnemar(y_true_fix, y_hat_class_rlog, y_hat_class_base, alpha=alpha)
print("theta_hat = "+str(thetahat_rlog_base))

print("\n \n Comparing Decision Tree and Baseline\n")
[thetahat_tree_base, CI_tree_base, p_tree_base] = mcnemar(y_true_fix, y_hat_class_tree, y_hat_class_base, alpha=alpha)
print("theta_hat = "+str(thetahat_tree_base))

    