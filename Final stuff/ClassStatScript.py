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

# Flatten the arrays
y_hat_class_rlog = np.ravel(y_hat_class[0])
y_hat_class_tree = np.ravel(y_hat_class[1])
y_hat_class_base = np.ravel(y_hat_class[2])
y_true_flat = np.ravel(y_hat_class[2])


alpha = 0.05

print("\n \n Comparing Logistic Regression and Decision Tree\n")
[thetahat_rlog_tree, CI_rlog_tree, p_rlog_tree] = mcnemar(y_true_flat, y_hat_class_rlog, y_hat_class_tree, alpha=alpha)

print("\n \n Comparing Logistic Regression and Baseline\n")
[thetahat_rlog_base, CI_rlog_base, p_rlog_base] = mcnemar(y_true_flat, y_hat_class_rlog, y_hat_class_base, alpha=alpha)

print("\n \n Comparing Decision Tree and Baseline\n")
[thetahat_tree_base, CI_tree_base, p_tree_base] = mcnemar(y_true_flat, y_hat_class_tree, y_hat_class_base, alpha=alpha)


    