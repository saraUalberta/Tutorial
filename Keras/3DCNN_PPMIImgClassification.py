# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 17:57:08 2017

@author: sara
"""

############################ 3D CNN With Keras for PPMI Data###################
import numpy as np
import argparse
import os
FLOAT_PRECISION = np.float32
import glob
import nibabel as nib
import csv
import random
import matplotlib.pyplot as plt
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,                          MaxPooling3D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

## subfunction
def data_balancing(labels):
    ## find the positive and negative index
    pos_indx = [i for i, j in enumerate(labels) if j =='PD']
    
    neg_indx = [i for i, j in enumerate(labels) if j =='Control']
    #print(pos_indx,neg_indx)
    ### PPMI Data
    ## select a number of positive samples because the # is more than neg one
    sel_pos_indx = random.sample(pos_indx, 55)
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
        if ndata.shape == (176,240,256):
            SubjData.append(nii_data.tolist())    
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

def train_test_split(samples , orglabels,hatlabels, test_size=0.2, random_state=43):
    ## data balancing
    bal_Indx = data_balancing(orglabels)
    ## Shuffle the data
    random.shuffle(bal_Indx)
    # separate test and train set
    ntest = round( test_size * len(bal_Indx))
    ntrain =  len(bal_Indx) - ntest
    train_Indx = bal_Indx[:ntrain]
    test_Indx =bal_Indx[ntrain:]
    train_data=[]
    test_data=[]
    train_label=[]
    test_label=[]
    for w in train_Indx:
        #print(w)
        train_data.append(samples[w])
        train_label.append(hatlabels[w])
        
    for w2 in test_Indx:
        #print(w2)
        test_data.append(samples[w2])
        test_label.append(hatlabels[w2])
    X_train, X_test, Y_train, Y_test = train_data, test_data, train_label, test_label
    return X_train, X_test, Y_train, Y_test 
        
###################################### Main Part###############################    
def main():
#    parser = argparse.ArgumentParser(description='simple 3D convolution for PPMI Image recognition')
#    parser.add_argument('--batch', type=int, default=128)
#    parser.add_argument('--epoch', type=int, default=100)
#    parser.add_argument('--videos', type=str, default='UCF101',
#                        help='directory where videos are stored')
#    parser.add_argument('--nclass', type=int, default=101)
#    parser.add_argument('--output', type=str, required=True)
#    parser.add_argument('--color', type=bool, default=False)
#    parser.add_argument('--skip', type=bool, default=True)
#    parser.add_argument('--depth', type=int, default=10)
#    args = parser.parse_args()

    img_rows, img_cols, frames = 176, 240, 256
    channel = 1    
    nb_classes = 2
    orgsamples ,orglabels = load_data()
    epoch = 100
    batch = 11
    ## find the number of positive and negative samples
    pos_indx = [i for i, j in enumerate(orglabels) if j =='PD']
    neg_indx = [i for i, j in enumerate(orglabels) if j =='Control']
    print('The number of PD samples are:')    
    print(len(pos_indx))
    print('The number of Control samples are:')    
    print(len(neg_indx))
    labels= np.zeros(len(orglabels))
    for k in pos_indx:
       labels[k]=1
    
    samples =  np.array(orgsamples)    
    X = samples.reshape((samples.shape[0], img_rows, img_cols, frames, channel))
    Y = np_utils.to_categorical(labels, nb_classes)

   

    # Define model
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
        X.shape[1:]), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(), metrics=['accuracy'])
    model.summary()
#    plot_model(model, show_shapes=True,
#               to_file=os.path.join(args.output, 'model.png'))
  
    X_train, X_test, Y_train, Y_test = train_test_split(X ,orglabels, Y, test_size=0.2, random_state=43)

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch,
                        epochs=epoch, verbose=1, shuffle=True)
    model.evaluate(X_test, Y_test, verbose=0)
#    model_json = model.to_json()
#    if not os.path.isdir(args.output):
#        os.makedirs(args.output)
#    with open(os.path.join(args.output, 'ucf101_3dcnnmodel.json'), 'w') as json_file:
#        json_file.write(model_json)
#    model.save_weights(os.path.join(args.output, 'ucf101_3dcnnmodel.hd5'))

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
#    plot_history(history, args.output)
#    save_history(history, args.output)

main()
'''
img_rows, img_cols, frames = 176, 240, 256
channel = 1    
nb_classes = 2
samples ,orglabels = load_data()
## find the number of positive and negative samples
pos_indx = [i for i, j in enumerate(orglabels) if j =='PD']
neg_indx = [i for i, j in enumerate(orglabels) if j =='Control']
print('The number of PD samples are:')    
print(len(pos_indx))
print('The number of Control samples are:')    
print(len(neg_indx))
labels= np.zeros(len(orglabels))
for k in pos_indx:
    labels[k]=1
    
samples =  np.array(samples)    
X = samples.reshape((samples.shape[0], img_rows, img_cols, frames, channel))
Y = np_utils.to_categorical(labels, nb_classes)

X_train, X_test, Y_train, Y_test = train_test_split(samples, orglabels, Y ,test_size=0.2, random_state=43)
'''
