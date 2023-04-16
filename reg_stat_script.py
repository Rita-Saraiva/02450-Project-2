# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 19:01:47 2023

@author: Jonas
"""


from toolbox_02450 import *
import pickle
from sklearn import *
from toolbox_02450 import mcnemar
import numpy as np

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
    globals()["Sq_loss_ANN_"+str(i+1)]=new_array

for i in range(len(Sq_loss_RLR)):
    new_array=np.ravel(Sq_loss_RLR[i])
    globals()["Sq_loss_RLR_"+str(i+1)]=new_array

for i in range(len(Sq_loss_base)):
    new_array=np.ravel(Sq_loss_base[i])
    globals()["Sq_loss_base_"+str(i+1)]=new_array
    


 


