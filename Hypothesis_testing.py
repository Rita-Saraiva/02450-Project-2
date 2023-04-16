# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Mathias, Jonas)s
"""
import pickle
from toolbox_02450 import *
from sklearn import *
import numpy as np

#In the following script on the data from our comparison of the regression models
# we do hypothesis testing by p value and confidence intervals

#This is based on script 7.3.1
#Loading results from regression comparison script


# Load Info_Table from file

with open('Sq_loss_ANN.pickle', 'rb') as f:
    Sq_loss_ANN = pickle.load(f)
with open('Sq_loss_RLR.pickle', 'rb') as f:
    Sq_loss_RLR = pickle.load(f)
with open('Sq_loss_base.pickle', 'rb') as f:
    Sq_loss_base = pickle.load(f)

sq_loss_ann_fix=[]
sq_loss_rlr_fix=[]
sq_loss_base_fix=[]

for i in range(len(Sq_loss_ANN)):
    new_array=np.ravel(Sq_loss_ANN[i])
    #globals()["Sq_loss_ANN_"+str(i+1)]=new_array
    sq_loss_ann_fix=np.concatenate((sq_loss_ann_fix,new_array))
    
for i in range(len(Sq_loss_RLR)):
    new_array=np.ravel(Sq_loss_RLR[i])
    #globals()["Sq_loss_RLR_"+str(i+1)]=new_array
    sq_loss_rlr_fix=np.concatenate((sq_loss_rlr_fix,new_array))
    
for i in range(len(Sq_loss_base)):
    new_array=np.ravel(Sq_loss_base[i])
    #globals()["Sq_loss_base_"+str(i+1)]=new_array
    sq_loss_base_fix=np.concatenate((sq_loss_base_fix,new_array))


#%% 

alpha = 0.05

Models=["ANN","RLR","Baseline"]

Z=np.zeros((214,3))
Z[:,0]=sq_loss_ann_fix
Z[:,1]=sq_loss_rlr_fix
Z[:,2]=sq_loss_base_fix


for i in range(3):
    for j in range(i):
        if i==j:
            pass
        else:
            print("\n  \nModel {0} and {1}".format(Models[i],Models[j]))
            z=Z[:,i]-Z[:,j]
            z_hat= np.mean(z)
            sigma= st.sem(z)
            
            CI_setupI = st.t.interval(1 - alpha, len(z) - 1, loc=z_hat, scale=st.sem(z))  # Confidence interval
            p_setupI = 2*st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value

            print( "p="+str(p_setupI) )
            print("mean(z)="+str(z_hat) )
            print("CI=("+str(CI_setupI[0])+","+str(CI_setupI[1])+")")
                
            
            
