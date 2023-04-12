# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:30:50 2023

@author: Jonas
"""

#Make some weights based upon the inverse of the occurance of each class
https://stackoverflow.com/questions/35236836/weighted-linear-regression-with-scikit-learn



import matplotlib.pyplot as plt
import sklearn.linear_model as lm

from Loading_data import *

# Split dataset into features and target vector
y = ClassKMatrix[:,:]


# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y


# find the index of the maximum value in each row of y_est
y_int = np.argmax(y_est, axis=1)

# add 1 to the index values to get the corresponding column numbers (1 to 7)
y_int = y_int + 1

# reshape y_int to a column vector
y_int = y_int.reshape(-1, 1)

# extract the 11th column of raw_data
col_11 = raw_data[:, 10]

# plot y_int against col_11
plt.plot(col_11, y_int, 'o')

xplot=np.arange(1,215)

# plot y_int and col_11 against the x-axis values
plt.plot(xplot, y_int, 'o', label='Linear Regression')
plt.plot(xplot, col_11, label='Glass Class')

# add labels and legend to the plot
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('y_int and 11th column of raw_data')
plt.legend(loc='upper right', bbox_to_anchor=(1.40, 1))
plt.show()


