# -*- coding: utf-8 -*-
"""
@author: Camilo Velasquez Agudelo

Functions for German Traffic Signs classification using LENET method in Tensor Flow
"""
#Imports
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from data_processing import evaluate

def TF_LENET_infer(X_infer,model_train_path):
    """
    """
    Graph_1=tf.Graph()
    with tf.Session(graph=Graph_1) as sess:
        loader=tf.train.import_meta_graph(model_train_path+'\\german_lenet.ckpt.meta')
        loader.restore(sess,model_train_path+'\\german_lenet.ckpt')
        X = tf.get_default_graph().get_tensor_by_name("X:0")
        prob=tf.get_default_graph().get_tensor_by_name("prob:0")
        predict=sess.run(prob,feed_dict={X:X_infer})   
    return predict


def TF_LENET_test(X_test,Y_test,model_train_path,batch_size=200):
    """
    X_test: 
    Y_test:
    """
    Graph_1=tf.Graph()
    with tf.Session(graph=Graph_1) as sess:
        loader=tf.train.import_meta_graph(model_train_path+'\\german_lenet.ckpt.meta')
        loader.restore(sess,model_train_path+'\\german_lenet.ckpt')
        X = tf.get_default_graph().get_tensor_by_name("X:0")
        Y = tf.get_default_graph().get_tensor_by_name("Y:0")
        accuracy_operation=tf.get_default_graph().get_tensor_by_name("accuracy_operation:0")
        accuracy=evaluate(X,Y,X_test, Y_test, accuracy_operation, batch_size)
        print("Accuracy: ", accuracy)
      

def TF_LENET_train(X_train,Y_train, model_train_path, epochs=60, batch_size=64,rate=0.0009):
    """
    LENET
    INPUT
    X_train: 
    Y_train: 
    """
    #Connect graph
    Graph_1=tf.Graph()
    with Graph_1.as_default():
        X = tf.placeholder(tf.float32, (None, 32, 32, 1),name="X")
        Y = tf.placeholder(tf.int32, (None, 43),name="Y")
        logits = LENET(X)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        training_operation = optimizer.minimize(loss_operation)
        tf.nn.softmax(logits=logits,name="prob")
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1),name="correct_prediction") #If equal 1
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy_operation") #Average of correct predictions
        saver = tf.train.Saver()
    #Sess run 
    with tf.Session(graph=Graph_1) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        num_examples = len(X_train)
        for i in range(epochs):
            X_train, Y_train = shuffle(X_train, Y_train)
            for j in range(0, num_examples, batch_size):
                end = j + batch_size
                batch_X, batch_Y = X_train[j:end], Y_train[j:end]
                sess.run(training_operation, feed_dict={X: batch_X, Y: batch_Y})
            #Could be changed to validation data
            validation_accuracy = evaluate(X,Y,X_train, Y_train, accuracy_operation, batch_size)
        print("EPOCH {} ... Validation Accuracy = {:.3f}".format(i+1,validation_accuracy))
        saver.save(sess, model_train_path+'\\german_lenet.ckpt')
    pass

def LENET(X_lenet,mu=0,sigma=0.1):
    """
    LENET description
    - 
    INPUT:
    X_lenet: A Tensor (tf.placeholder(tf.float32, (None, 32, 32, 1)))
    mu: The mean for the truncated normal distribution
    sigma: The standard deviation of the normal distribution, before truncation
    OUTPUT:
    logits: A tensor which is the result of LENET-5 convolutional neural networks
    """
    depth_image=1  
    #Convolution layer 1
    filter_size_1=5
    num_filters_1=6
    #Subsampling 1
    filter_size_s1=2
    #Convolution layer 2
    filter_size_2=5
    num_filters_2=16
    #Subsampling 2
    filter_size_s2=2
    #Fully connected 1
    fc_size_1=120
    #Fully connected 2
    fc_size_2=84
    #Last layer (Number of classes)
    last_size=43
    
    #C1
    C1_w = tf.Variable(tf.truncated_normal(shape = [filter_size_1,filter_size_1,depth_image,num_filters_1],mean = mu, stddev = sigma))
    C1_b = tf.Variable(tf.zeros(num_filters_1))
    C1 = tf.add(tf.nn.conv2d(X_lenet,C1_w, strides = [1,1,1,1], padding = 'VALID'),C1_b)
    C1 = tf.nn.relu(C1)
    
    #S1
    S1 = tf.nn.max_pool(C1,ksize = [1,filter_size_s1,filter_size_s1,1], strides = [1,2,2,1], padding = 'VALID')
    
    #C2
    C2_w = tf.Variable(tf.truncated_normal(shape = [filter_size_2,filter_size_2,num_filters_1,num_filters_2],mean = mu, stddev = sigma))
    C2_b = tf.Variable(tf.zeros(num_filters_2))
    C2 = tf.add(tf.nn.conv2d(S1,C2_w, strides = [1,1,1,1], padding = 'VALID'),C2_b)
    C2 = tf.nn.relu(C2)
    
    #S2
    S2 = tf.nn.max_pool(C2,ksize = [1,filter_size_s2,filter_size_s2,1], strides = [1,2,2,1], padding = 'VALID')
    
    #Fully connected 1
    #Flatten
    FC1 = flatten(S2)
    FC1_w = tf.Variable(tf.truncated_normal(shape = (filter_size_2*filter_size_2*num_filters_2,fc_size_1), mean = mu, stddev = sigma))
    FC1_b = tf.Variable(tf.zeros(fc_size_1))
    FC1 = tf.add(tf.matmul(FC1,FC1_w),FC1_b)
    FC1 = tf.nn.relu(FC1)
    
    #Fully connected 2
    FC2_w = tf.Variable(tf.truncated_normal(shape = (fc_size_1,fc_size_2), mean = mu, stddev = sigma))
    FC2_b = tf.Variable(tf.zeros(fc_size_2))
    FC2 = tf.add(tf.matmul(FC1,FC2_w),FC2_b)
    FC2 = tf.nn.relu(FC2,name="FC2")
    
    #Fully connected 3
    FC3_w = tf.Variable(tf.truncated_normal(shape = (fc_size_2,last_size), mean = mu , stddev = sigma))
    FC3_b = tf.Variable(tf.zeros(last_size))
    logits = tf.add(tf.matmul(FC2, FC3_w),FC3_b,name="logits")
    return logits

