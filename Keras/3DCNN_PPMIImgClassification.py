# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 17:57:08 2017

@author: sara
"""

############################ 3D CNN With Keras for PPMI Data###################
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
import numpy as np
import os
FLOAT_PRECISION = np.float32
import glob
import nibabel as nib
import csv
import random
import matplotlib
import matplotlib.pyplot as plt



## subfunction
def data_balancing(labels):
    ## find the positive and negative index
    pos_indx = [i for i, j in enumerate(labels) if j =='PD']
    neg_indx = [i for i, j in enumerate(labels) if j =='Control']
    #print(pos_indx,neg_indx)
    ### PPMI Data
    ## select a number of positive samples because the # is more than neg one
    sel_pos_indx = random.sample(pos_indx, 86)
    ### Dr Ba DATA
    #sel_pos_indx = pos_indx
    #labels =['Control','Control','Control','Control','Control','PD','PD','PD','PD',]

    ##
    samp_indx = neg_indx
    samp_indx.extend( sel_pos_indx)
    return(samp_indx)
def get_clincal_info(SubjectId):
    with open('prepPPMIT1Sag_Ap5_6_19_2017.csv') as f6:
        infodata = csv.reader(f6)
        if infodata is not None: 
            for row in infodata:
                if row[1]==SubjectId:
                    #print('Hi')
                    #print(row[2],row[3],row[4])
                    return row[2],row[3],row[4]  
                    
def data_mpropsize(SubjData,maxDim,SubjSize):
    zpimglist=[]
    for k in range(len(SubjData)):
        #print(k)
        [w,h,d]=SubjData[k].shape
        
        zpdata=np.zeros([maxDim[0], maxDim[1], maxDim[2]], np.float)
        for i in range(w):
            curimg=SubjData[k][i]
            #zpdata.append(np.resize(curimg,[maxDim[1], maxDim[2]]))
            zpdata[i]=np.resize(curimg,[maxDim[1], maxDim[2]])
            #flattenData.append(zpdata[i].flatten())
        zpimglist.append(zpdata)
        del(zpdata)
    return(zpimglist)               
def load_data():
    data_dir='/home/sara/Documents/Thesis/DataSet/sara_PPMI_Aprilsix/PPMI/';
    subjects = os.listdir(data_dir)
    subjects = glob.glob(data_dir+'/*/')
    SubjData=[]
    SubjSize=[]
    SubjId=[]
    labels = []
    AgeVec=[]
    SexVec=[]
    print(subjects)
    cnt=0
    for sub in subjects:
        child_dir = []
        dirs = [x[0] for x in os.walk(os.path.join(data_dir, sub))]
        child_dir = dirs[3]
        nii_file = glob.glob(child_dir+'/*nii')[0]
        #print(nii_file)
        #if 'T2' not in nii_file:
        nii_data = nib.load(nii_file).get_data()
        ndata = np.asarray(nii_data)
        ndata = ndata.astype('float32')      
        SubjData.append(ndata)    
        SubjSize.append(ndata.shape)
 
        #SubjectId = filter(lambda x: x.isdigit(), sub)
        SubjectId = ''.join([x for x in sub if x.isdigit()])
        SubjId.append(SubjectId)
        label,sex,age = get_clincal_info(SubjectId) 
        AgeVec.append(age)
        SexVec.append(sex)  
        labels.append(label)
        print(cnt)
        ndata=[]
        cnt+=1
    #maxDim=np.max(np.asarray(SubjSize),axis=0)
    #print(SubjSize)
    nn=0
    for i in range(len(SubjSize)): 
        if SubjSize[i]==(176,240,256):
            nn+=1
    print(nn)
#    print(maxDim)   
#    zpimglist = data_mpropsize(SubjData,maxDim,SubjSize)
#    with open('res_PPMIData', 'wb') as f5:
#        pickle.dump(zpimglist, f5,protocol=2)
#    with open('res_PPMILabel', 'wb') as f6:
#        pickle.dump(labels, f6,protocol=2)
    return(SubjData,labels)
###################################### Main Part###############################
# image specification
img_rows,img_cols,img_depth=16,16,15
## define the train data           
samples,labels = load_data()