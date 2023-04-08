# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:05:30 2023

@author: Rita
"""

#Importing data
from Loading_data import * 


from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection


# Maximum number of neighbors
L=40


#Therefore the Classification y data is the the glass_type
y = glass_type
#The X data is the all of the features
X = Y2


#Shape of X matrix
N, M = X.shape
C = len(ClassNames)

CV = model_selection.LeaveOneOut()
errors = np.zeros((N,L))
i=0
for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[i,l-1] = np.sum(y_est[0]!=y_test[0])

    i+=1
    
# Plot the classification error rate
figure()
plot(100*sum(errors,0)/N)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()
