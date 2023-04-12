# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:30:50 2023

@author: Jonas
"""

#Make some weights based upon the inverse of the occurance of each class
#https://stackoverflow.com/questions/35236836/weighted-linear-regression-with-scikit-learn



import matplotlib.pyplot as plt
import sklearn.linear_model as lm

from Loading_data import *

# Split dataset into features and target vector
y = ClassKMartix[:,:]


# extract the 11th column of raw_data
col_11 = raw_data[:, 10]


########################## The unweighted model ################################
# Fit ordinary least squares regression model
model_unweight = lm.LinearRegression()
model_unweight.fit(X,y)

# Predict class
y_est_unweight = model_unweight.predict(X)
residual_unweight = y_est_unweight-y

# find the index of the maximum value in each row of y_est
y_int_unweight = np.argmax(y_est_unweight, axis=1)

# add 1 to the index values to get the corresponding column numbers (1 to 7)
y_int_unweight = y_int_unweight + 1

# reshape y_int to a column vector
y_int_unweight = y_int_unweight.reshape(-1, 1)


########################## Adjusting the weights ################################

#Making logical weights based upon the dataset
#First
hit_miss_column=np.column_stack((col_11,y_int_unweight))

Confusion_matrix=np.zeros([7,4])
    
for i, element in enumerate(hit_miss_column):
    if element[1] == element[0]:
        Confusion_matrix[int(element[1]-1),0]+=1
    elif element[1] != element[0]:
        Confusion_matrix[int(element[1]-1),1]+=1
    Confusion_matrix[int(element[0]-1),2]+=1
    Confusion_matrix[int(element[1]-1),3]+=1

  

# weights = np.array([Confusion_matrix[0,3]/(Confusion_matrix[0,2]),
#                     Confusion_matrix[1,3]/(Confusion_matrix[1,2]),
#                     Confusion_matrix[2,3]/(Confusion_matrix[2,2]),
#                     Confusion_matrix[3,3]/(Confusion_matrix[3,2]),
#                     Confusion_matrix[4,3]/(Confusion_matrix[4,2]),
#                     Confusion_matrix[5,3]/(Confusion_matrix[5,2]),
#                     Confusion_matrix[6,3]/(Confusion_matrix[6,2])])

#weights = np.array([0.228571,0.276316,1,1,0.692308,0.888889,0.137931])
#weights = np.array([0.228571,0.276316,1,0,0.692308,0.88889,0.137931])
#weights = np.array([0.79375,0.792453,1,1,0.995238,0.995305,0.962963])
#weights = np.array([3.962963,3.890909,1,1,53.5,214,8.56])
#weights = np.array([143,147,231,214,218,220,186])+214
#weights = np.array([1,9,34,0,17,15,1])
#weights = np.array([15,15,51,0,26,23,5])
#weights = np.array([0.183908,0.238636,1,0,1.8,4,0.125])
#weights = np.array([0.62069,0.625,1.4,0,0.8,0.5,0.78125])
#weights = np.array([0.22857,0.27632,0.5,0,0.69231,0.88889,0.13793])
#weights = np.array([20,31,34,0,20,15,19])
#weights = np.array([1.296296,1.381818,5,0,3.25,9,1.16])  -Måske?
weights = np.array([1,9,34,0,17,15,1])


col_11_weighted=np.array([])

for i in col_11:
    case = {
        1: 'col_11_weighted=np.append(col_11_weighted,weights[0])',
        2: 'col_11_weighted=np.append(col_11_weighted,weights[1])',
        3: 'col_11_weighted=np.append(col_11_weighted,weights[2])',
        4: 'col_11_weighted=np.append(col_11_weighted,weights[3])',
        5: 'col_11_weighted=np.append(col_11_weighted,weights[4])',
        6: 'col_11_weighted=np.append(col_11_weighted,weights[5])',
        7: 'col_11_weighted=np.append(col_11_weighted,weights[6])',
    }
    exec(case.get(i))


########################## The weighted model ################################



# Fit ordinary least squares regression model
model_weighted = lm.LinearRegression()
model_weighted.fit(X,y,col_11_weighted)

# Predict class
y_est_weighted = model_weighted.predict(X)
residual_weighted = y_est_weighted-y

# find the index of the maximum value in each row of y_est
y_int_weighted = np.argmax(y_est_weighted, axis=1)

# add 1 to the index values to get the corresponding column numbers (1 to 7)
y_int_weighted = y_int_weighted + 1

# reshape y_int to a column vector
y_int_weighted = y_int_weighted.reshape(-1, 1)




xplot=np.arange(1,215)

# plot y_int and col_11 against the x-axis values
plt.plot(xplot, y_int_unweight, 'o', label='Unweighted Linear Regression')
plt.plot(xplot, col_11, label='Glass Class')

# add labels and legend to the plot
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('unweighted y_int and 11th column of raw_data')
plt.legend(loc='upper right', bbox_to_anchor=(1.40, 1))
plt.show()



# plot y_int and col_11 against the x-axis values
plt.plot(xplot, y_int_weighted, 'o', label='Weighted Linear Regression')
plt.plot(xplot, col_11, label='Glass Class')

# add labels and legend to the plot
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('weighted y_int and 11th column of raw_data')
plt.legend(loc='upper right', bbox_to_anchor=(1.40, 1))
plt.show()



