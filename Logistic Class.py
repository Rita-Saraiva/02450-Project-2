# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:12:42 2023

@author: Jonas
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
from toolbox_02450 import rocplot, confmatplot
import sklearn.linear_model as lm
from matplotlib.pyplot import figure, show, title
from Loading_data import *
from mpl_toolkits.mplot3d import Axes3D

# Preparing data

K = 10


# Split dataset into features and target vector
#y = ClassKMartix[:,:].squeeze()
C = len(ClassNames)
y = raw_data[:, 10].squeeze()
X_reg = X
X_reg

Test_SIZE = 1/K
X_train, X_test, y_train, y_test = train_test_split(X_reg, y, test_size=Test_SIZE, stratify=y)

# Standardize data based on training set
mu_train = np.mean(X_train, 0)
sigma_train = np.std(X_train, 0)
X_train = (X_train - mu_train) / sigma_train
X_test = (X_test - mu_train) / sigma_train

#%% Model fitting and prediction

# Fit multinomial logistic regression model
regularization_strength = 10000

mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                               tol=1e-4, random_state=1, 
                               penalty='l2', C=1/regularization_strength)
mdl.fit(X_train,y_train)
y_test_est = mdl.predict(X_test)

test_error_rate = np.sum(y_test_est!=y_test) / len(y_test)



# Number of miss-classifications
print('Error rate: \n\t {0} % out of {1}'.format(test_error_rate*100,len(y_test)))


plt.figure(2, figsize=(9,9))
plt.hist([y_train, y_test, y_test_est], color=['red','green','blue'], density=True)
plt.legend(['Training labels','Test labels','Estimated test labels'])


# predict = lambda x: np.argmax(mdl.predict_proba(x),1)
# plt.figure(2,figsize=(9,9))
# visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], AttributeNames, ClassNames)
# plt.title('LogReg decision boundaries')
# plt.show()







#%% Model based upon 8_3_2

logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, random_state=1)
logreg.fit(X_train,y_train)
print('Number of miss-classifications for Multinormal regression:\n\t {0} out of {1}'.format(np.sum(logreg.predict(X_test)!=y_test),len(y_test)))
print('Error rate: \n\t {0} % out of {1}'.format(test_error_rate*100,len(y_test)))


# predict = lambda x: np.argmax(logreg.predict_proba(x),1)
# figure(2,figsize=(9,9))
# visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], AttributeNames, ClassNames)
# title('LogReg decision boundaries')

show()

#%% Plots
















# lambda_interval = np.logspace(-8, 2, 50)
# train_error_rate = np.zeros(len(lambda_interval))
# test_error_rate = np.zeros(len(lambda_interval))
# coefficient_norm = np.zeros(len(lambda_interval))
# for k in range(0, len(lambda_interval)):
#     mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
    
#     mdl.fit(X_train, y_train)

#     y_train_est = mdl.predict(X_train).T
#     y_test_est = mdl.predict(X_test).T
    
#     train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
#     test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

#     w_est = mdl.coef_[0] 
#     coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

# min_error = np.min(test_error_rate)
# opt_lambda_idx = np.argmin(test_error_rate)
# opt_lambda = lambda_interval[opt_lambda_idx]



# attributeName, C, classNames, coefficient_norm, font_size, K, k, lambda_interval,
# M, mat_data, mdl, min_error, mu, N, opt_lambda, opt_lambda_idx, sigma, test_error_rate,
# train_error_rate, w_est, X, X_test, X_train, y, y_test, y_test_est, y_train, y_train_est



