# -*- coding: utf-8 -*-
"""
Created on Wed May 23 18:30:50 2018

@author: Camilo Velasquez A

Functions for data processing, data adequacy and plot
"""
#Imports
import os
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.externals import joblib
import matplotlib.pyplot as plt

def preprocessingdata(X_pre,Y_pre,directory):
    """
    """
    n,h,w,d=X_pre.shape
    X_preprocessed=X_pre.reshape([n,-1])
    Y_preprocessed=inverse_ohe(directory,Y_pre)
    return X_preprocessed,Y_preprocessed

def inverse_ohe(directory,Y_pre):
    """
    """
    classes=joblib.load(os.path.split(directory)[0]+'\\train'+'\\classes.pkl')
    Y_preprocessed=np.matmul(Y_pre,classes.active_features_)
    return Y_preprocessed

def data_adequacy(directory,train=True):
    """
    """
    os.chdir(directory)
    data=list()
    files=os.listdir(directory)
    files=[file for file in files if file.endswith('.ppm')]
    for file in files:
        if file.endswith('.ppm'):
            im=cv2.imread(directory+'\\'+file)
            resize_im=cv2.resize(im, (32, 32))
            gray=cv2.cvtColor(resize_im,cv2.COLOR_BGR2GRAY)
            label=int(file[0:2])
            data.append([gray,label])
    df=pd.DataFrame(data)
    if train:
        ohe=OneHotEncoder()
        classes=ohe.fit(np.reshape(df[1].values,(-1,1)))
        joblib.dump(classes,directory+'\\classes.pkl')    
    else:
        classes= joblib.load(os.path.split(directory)[0]+'\\train\\classes.pkl')
    Y=classes.transform(np.reshape(df[1].values,(-1,1))).toarray()
    df[1]=Y.tolist()
    
    #X and Y
    X_data=df[0].values
    h,w=X_data[0].shape
    X_data=np.concatenate(X_data).reshape(len(X_data),h,w,1)
    Y_data=df[1].values
    labels=len(Y_data[0])
    Y_data=np.concatenate(Y_data).reshape(len(Y_data),labels)
    X_data=X_data.astype(np.float32)
    Y_data=Y_data.astype(np.int32)
    return X_data, Y_data

def infer_plot(X_infer,predict,Y_infer):
    """
    """
    images_and_labels = list(zip(X_infer, predict, Y_infer ))
    for index, (image, predict_i, correct) in enumerate(images_and_labels[:]):
        plt.figure(index)
        plt.axis('off')
        plt.imshow(image[:,:,0], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Prediction: %i - Correct: %i, Probability: %i' % (np.argmax(predict_i), correct, np.max(predict_i)*100)+'%')
    plt.show()
    
def evaluate(X,Y,X_eval, Y_eval, accuracy_operation, batch_size):
    num_examples = len(X_eval)
    total_accuracy = 0
    sess = tf.get_default_session()
    for i in range(0, num_examples, batch_size):
        batch_X, batch_Y = X_eval[i:i+batch_size], Y_eval[i:i+batch_size]
        batch_X=batch_X.astype(np.float32)
        batch_Y=batch_Y.astype(np.int32)
        accuracy = sess.run(accuracy_operation, feed_dict={X: batch_X, Y: batch_Y})
        total_accuracy += (accuracy * len(batch_X))
    return total_accuracy/num_examples