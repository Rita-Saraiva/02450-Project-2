# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Mathias, Jonas, Rita)s
"""

#import os
#os.chdir('C:/Users/ritux/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DTU/1 6º Semester/1 3 02450 Machine Learning/Project 2/02450-Project-2')


#Loading the data
import numpy as np

# We will load the data with pd
import pandas as pd

# We start by defining the path to the file that we're we need to load.
# Upon inspection, we saw that the messy_data.data was infact a file in the
# format of a CSV-file with a ".data" extention instead.  
#file_path = r'C:/Users/ritux/OneDrive - Danmarks Tekniske Universitet/Skrivebord/DTU/1 6º Semester/1 3 02450 Machine Learning/Project 2/glass.data'
# First of we simply read the file in using readtable, however, we need to
# tell the function that the file is tab-seperated. We also need to specify
# that the header is in the second row:
data = pd.read_csv("glass.data", sep=',', header=None)


""" Preparing the Data"""

#Raw data
raw_data = data.values

# The last column is an integer for the glass type (Classes)
#We decide to extract this in a variable for itself
glass_type = np.array(raw_data[:,10])

#Continuous data
D = np.array(raw_data[:,1:10],dtype=np.float64)
#Sutracting 1 from all values to a true ratio
RI_Ratio = np.array([ RI_Element-1 for RI_Element in D[:,0]])
D[:,0]=RI_Ratio

#Defining the data as X
X = D

#Shape of data
N, M = D.shape

#Standardizing and then doing the same plots and computations
# Subtract the mean from the data
Y1 = X - np.ones((N, 1))*X.mean(0)

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y2 = X - np.ones((N, 1))*X.mean(0)
Y2 = Y2*(1/np.std(Y2,0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions 
# of Y2

mu = np.mean(X[:, :], 0)
sigma = np.std(X[:, :], 0)


Y3 = (X[:, :] - mu ) / sigma 




""" Class Names and One-of-K Coding"""
#Class Names are found in the .names file and manually added
ClassNames = ['Building Windows - Float Processed', 'Building Windows - Non Float Processed',
              'Vehicle Windows - Float Processed','Vehicle Windows - Non Float Processed',
              'Containers', 'Tableware', 'Headlamps']

#Hereafter a dictionary is created according to the .names file
ClassDict = dict(zip(ClassNames,range(1,8)))
#Abreviations of Attributenames are also manually added
ShortClassNames=["bwfp","bwnfp","vwfp","vwnfp",
                 "conts","tware","hlamp"] 
SclassDict = dict(zip(ShortClassNames,range(1,8)))


#Attributenames are also manually added
AttributeNames = ["Refractive Index","Sodium",
                  "Magnesium","Aluminum","Silicon",
                  "Potassium","Calcium","Barium","Iron"]
#Abreviations of Attributenames are also manually added
ShortAttributeNames = ["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"]


#One-of-K Coding

#Class Based - One-of-7 Coding - Matrix
ClassKMatrix=np.zeros([N,len(ClassNames)])


#Window/NonWindow Based - One-of-2 Coding - Matrix
BinaryKMatrix=np.zeros([N,2])
BinaryGlassType=np.zeros([N])

for i in range(N):#glass type
    Type=int(glass_type[i])-1
    ClassKMatrix[i,Type]=1
    if Type<=4:
        BinaryKMatrix[i,0]=1
        BinaryGlassType[i]=1
    else:
        BinaryKMatrix[i,1]=1
        BinaryGlassType[i]=2
    


