# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Mathias)s
"""
import pickle
from toolbox_02450 import *

#In the following script on the data from our comparison of the regression models
# we do hypothesis testing by p value and confidence intervals

#This is based on script 7.3.1
#Loading results from regression comparison script
# Load Info_Table from file
with open('Reg_Table.pickle', 'rb') as f:
    Reg_Table = pickle.load(f)
    
hp_Table=Reg_Table

#%% 

alpha = 0.05

Models=["ANN","RLR","Baseline"]

Z=np.zeros((hp_Table.shape[0],3))
Z[:,0]=hp_Table[:,2]
Z[:,1]=hp_Table[:,4]
Z[:,2]=hp_Table[:,5]
#n = 214
n = 5

for i in range(3):
    for j in range(i):
        if i==j:
            pass
        else:
            print("\n  \nModel {0} and {1}".format(Models[i],Models[j]))
            z=Z[:,i]-Z[:,j]
            z_hat= np.mean(z)
            sigma= st.sem(z)
            
            CI_setupI = st.t.interval(1 - alpha, n - 1, loc=z_hat, scale=st.sem(z))  # Confidence interval
            p_setupI = 2*st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=n - 1)  # p-value

            print( "p="+str(p_setupI) )
            print("mean(z)="+str(z_hat) )
            print("CI=("+str(CI_setupI[0])+","+str(CI_setupI[1])+")")
