# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Mathias)s
"""
#Decision tree classifier
#Based on 6.1.1
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show
from sklearn import model_selection, tree
import numpy as np

#Importing data
from Loading_data import * 

#The X data is the features
X = Y2[:,:]
#y = glass_type.squeeze() #All 7 glass types
y = BinaryGlassType.squeeze() #Binary class problem. 1 = Window glass 2 = Non windowd glass 
#Define number of K folds
K1 = 5

#Partionining data into a test and training set
CV = model_selection.KFold(K1, shuffle=True)

N, M = X.shape
C = len(ClassNames)+1
classNames = ClassNames
attributeNames = AttributeNames

#%% 

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

# Simple holdout-set crossvalidation
test_proportion = 0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)

# Initialize variables
Error_train = np.empty((len(tc),K1))
Error_test = np.empty((len(tc),K1))

for (k1, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    # Extract training and test set of the CV fold
    X_train = X[train_index]
    y_train = y[train_index].squeeze()
    X_test = X[test_index]
    y_test = y[test_index].squeeze()

    for i, t in enumerate(tc):
        
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train,y_train)
    
        # Evaluate classifier's misclassification rate over train/test data
        y_est_test = np.asarray(dtc.predict(X_test),dtype=int)
        y_est_train = np.asarray(dtc.predict(X_train), dtype=int)
        misclass_rate_test = sum(y_est_test != y_test) / float(len(y_est_test))
        misclass_rate_train = sum(y_est_train != y_train) / float(len(y_est_train))
        Error_test[i,k1], Error_train[i,k1] = misclass_rate_test, misclass_rate_train
        # Determine errors
        e = (y_est_test != y_test)
        print('Number of miss-classifications for Decisiontree of depth {2}:\n\t {0} out of {1}'.format(sum(e),len(e),t))

# =============================================================================
# f = figure()
# plot(tc, Error_train*100)
# plot(tc, Error_test*100)
# xlabel('Model complexity (max tree depth)')
# ylabel('Error (%)')
# legend(['Error_train','Error_test'])
#     
# show()    
# =============================================================================
Error_tresize_train = Error_train.mean(axis=1)
Error_tresize_test = Error_test.mean(axis=1)

print("The optimal treedepth is {0}".format(np.argmin(Error_tresize_test)+3))