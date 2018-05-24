# -*- coding: utf-8 -*-
"""
@author: Camilo Velasquez A

Functions for German Traffic Signs classification using Logistic 
    Regresion in Scikit Learn
"""
#Imports
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression


def SK_LR_infer(X_infer,model_train_path):
    """
    """
    LR=joblib.load(model_train_path+'\\german_sk_lr.pkl')
    predict=LR.predict_proba(X_infer)
    return predict

def SK_LR_test(X_test,Y_test,model_train_path):
    """
    """
    LR=joblib.load(model_train_path+'\\german_sk_lr.pkl')
    accuracy = LR.score(X_test,Y_test)
    print("Test accuracy ... " + str(accuracy*100) + "%")
    pass

def SK_LR_train(X_train,Y_train,model_train_path):
    """
    Train data using sklearn with logistic regression 
    INPUT
    data: Training data, List of n samples and data[i] means X,Y (label)
    OUTPUT
    model: 
    """
    
    LR = LogisticRegression(C=1,max_iter=200)
    LR.fit(X_train,Y_train)
    accuracy = LR.score(X_train,Y_train)
    print("Training accuracy ... " + str(accuracy*100) + "%")
    joblib.dump(LR,model_train_path+'\\german_sk_lr.pkl')
    pass