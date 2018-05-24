# -*- coding: utf-8 -*-
"""
@author: Camilo Velasquez A

Functions for German Traffic Signs classification using Logistic Regression in Tensor Flow
"""
#Imports
from sklearn.utils import shuffle
import tensorflow as tf
from data_processing import evaluate

def TF_LR_infer(X_infer,model_train_path):
    """
    """
    Graph_1=tf.Graph()
    with tf.Session(graph=Graph_1) as sess:
        loader=tf.train.import_meta_graph(model_train_path+'\\german_lr.ckpt.meta')
        loader.restore(sess,model_train_path+'\\german_lr.ckpt')
        X = tf.get_default_graph().get_tensor_by_name("X:0")
        prob=tf.get_default_graph().get_tensor_by_name("prob:0")
        predict=sess.run(prob,feed_dict={X:X_infer})   
    return predict

def TF_LR_test(X_test,Y_test,model_train_path,batch_size=200):
    """
    X_test: 
    Y_test:
    """
    Graph_1=tf.Graph()
    with tf.Session(graph=Graph_1) as sess:
        loader=tf.train.import_meta_graph(model_train_path+'\\german_lr.ckpt.meta')
        loader.restore(sess,model_train_path+'\\german_lr.ckpt')
        X = tf.get_default_graph().get_tensor_by_name("X:0")
        Y = tf.get_default_graph().get_tensor_by_name("Y:0")
        accuracy_operation=tf.get_default_graph().get_tensor_by_name("accuracy_operation:0")
        accuracy=evaluate(X,Y,X_test, Y_test, accuracy_operation, batch_size)
        print("Accuracy: ", accuracy)

def TF_LR_train(X_train,Y_train,model_train_path,rate=0.0006, epochs=60, batch_size=150):
    """
    X_train:
    Y_train:
    """
    _,classes=Y_train.shape
    Graph_1=tf.Graph()
    with Graph_1.as_default():
        X = tf.placeholder(tf.float32, (None, 32, 32, 1),name="X")
        Y = tf.placeholder(tf.int32, (None, classes),name="Y")
        LR=TF_Logistic_Regression(X,classes,mu=0,sigma=0.1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=LR, labels=Y)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        tf.nn.softmax(LR,name="prob")
        training_operation = optimizer.minimize(loss_operation)
        correct_prediction = tf.equal(tf.argmax(LR, 1), tf.argmax(Y, 1),name="correct_prediction") #If equal 1
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy_operation") #Average of correct predictions
        saver = tf.train.Saver()
    #Run 
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
            validation_accuracy = evaluate(X,Y,X_train, Y_train, accuracy_operation, batch_size)
        print("EPOCH {} ... Validation Accuracy = {:.3f}".format(i+1,validation_accuracy))
        saver.save(sess, model_train_path+'\\german_lr.ckpt')
    pass

def TF_Logistic_Regression(X_lr,classes,mu=0,sigma=0.1):
    """
    X:
    mu:
    sigma:
    """
    _,h,w,d=X_lr.shape
    X_flat = tf.contrib.layers.flatten(X_lr)
    W = tf.Variable(tf.truncated_normal(shape = [h.value*w.value*d.value,classes],mean = mu, stddev = sigma))
    b = tf.Variable(tf.zeros([1,classes]))
    LR = tf.add(tf.matmul(X_flat,W),b,name="LR")
    return LR

