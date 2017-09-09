# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:46:55 2017

@author: sara
"""

############################# keras learning_ Example 1########################
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras as ks
import numpy as np
import os
FLOAT_PRECISION = np.float32
import glob
import nibabel as nib
import csv
import random

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
    #print(subjects)
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
#    print(maxDim)
    
    #zpimglist = data_mpropsize(SubjData,maxDim,SubjSize)
#    with open('res_PPMIData', 'wb') as f5:
#        pickle.dump(zpimglist, f5,protocol=2)
#    with open('res_PPMILabel', 'wb') as f6:
#        pickle.dump(labels, f6,protocol=2)
    return(SubjData,labels)
## create the simplest model
model =  Sequential()

## desing model
model.add(Dense(units = 64, input_dim = 100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

## compile
#model.compile(loss='categoriacal_crossentropy',optimizer='sgd',metric=['accuracy'])
model.compile(loss=ks.losses.binary_crossentropy,optimizer=ks.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
              
## iterate throuhg my training data
## define the train data           
samples,labels = load_data()

## data balancing
bal_Indx = data_balancing(labels)
asubj_num =  len(bal_Indx)
print('Finish loading the data and balance the PD and Conrol!')
## Shuffle the data
random.shuffle(bal_Indx)
# separate test and train set
train_Indx = bal_Indx[:10]
test_Indx =bal_Indx[10:15]
x_train=[]
x_test=[]
y_train=[]
y_test=[]
for w in train_Indx:
    #print(w)
    x_train.append(samples[w])
    y_train.append(labels[w])

for w2 in test_Indx:
    #print(w2)
    x_test.append(samples[w2])
    y_test.append(labels[w2])

# Training 
print('Shape of train and test data')
print(len(x_train),len(x_test))
#x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=5)

##Evaluate your performance in one line: Or generate predictions on new data:

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

classes = model.predict(x_test, batch_size=128)
