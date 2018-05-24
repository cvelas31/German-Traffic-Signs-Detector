# -*- coding: utf-8 -*-
"""
Created on Wed May 16 20:16:25 2018

@author: CAMILO VELASQUEZ AGUDELO
"""
#Imports
import click
import urllib
import zipfile
import os
import shutil
from LENET import TF_LENET_infer,TF_LENET_test,TF_LENET_train
from SK_LR import SK_LR_infer,SK_LR_test,SK_LR_train
from TF_LR import TF_LR_infer,TF_LR_test,TF_LR_train
from data_processing import preprocessingdata,inverse_ohe,data_adequacy,infer_plot
from sklearn.model_selection import train_test_split

@click.group()
def main():
    pass

@main.command("download")
@click.option('-u', '--url', default='http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip', help='URL where the image datset is going to be downloaded.')
def download(url):
    """
    Download images from the German Traffic Signs Dataset
    
    \b
    This command will download all data from the German Traffic Signs Dataset 
    (http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset).
    This will store the the data set inside the images folder.
    It will split the images between train and test, and will be saved in their correspondent folder"
    SORRY, I forgot to become this a short single function.
    \b
    INPUT:
    url: (OPTIONAL) URL where the image dataset is going to be downloaded. 
        predetermined url is http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip
    """
    #Current working directory path 
    path=os.getcwd()
    #Link: 'http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip'
    print('Downloading from '+ url)
    urllib.request.urlretrieve(url,
                               filename=path +'/German_traffic_sign_data.zip')
    #Extract files
    zip_ref = zipfile.ZipFile(path +'/German_traffic_sign_data.zip', 'r')
    zip_ref.extractall(path + '/images')
    zip_ref.close()
    
    #Delete files except ReadMe.txt
    files=os.listdir(path + '\\images\\FullIJCNN2013')
    os.remove(path + '\\images\\FullIJCNN2013\\'+'gt.txt')
    os.remove(path + '\\images\\FullIJCNN2013\\'+'ReadMe.txt')
    for file in files:
        if file.endswith('.ppm'):
            os.remove(path + '/images/FullIJCNN2013/'+file)
    os.remove(path +'/German_traffic_sign_data.zip')
    #Take images to train
    general_source = path + '\\images\\FullIJCNN2013'
    dest = path + '\\images\\train'
    files = os.listdir(general_source)
    for f in files:
        try: sub_files = os.listdir(general_source+'\\'+f)
        except:
            pass
        for sub_f in sub_files:
            if sub_f.endswith('.ppm'):
                os.rename(general_source+'\\'+f+'\\'+sub_f, dest+'\\'+f+'_'+sub_f)
    
    #Divide between train and test
    files=os.listdir(dest)
    labels=list()
    new_files=list()
    for file in files:
        if file.endswith('.ppm'):
            new_files.append(file)
            labels.append(int(file[0:2]))
    #Split images files
    train_dest=path + '\\images\\train'
    test_dest=path + '\\images\\test'
    train_images,test_images, train_labels,test_labels=train_test_split(new_files,labels,test_size=0.2, stratify=labels) #Split random images
    for test_im in test_images:
        os.rename(train_dest+'\\'+test_im,test_dest+'\\'+test_im) 
    shutil.rmtree(general_source)
            
@main.command("train")
@click.option('-m', '--model', help='Number of model or name to train. See description above')
@click.option('-d', '--directory', help='Directory where the train images are placed in format .ppm')
def train(model, directory):
    """
    Train model # from images in directory ...   
    
    \b
    INPUT:
    model: Number of model or name to train
        (0 - 'SK_LR' refers to SK logistic regression)
        (1 - 'TF_LR' refers to  TF Logistic regression)
        (2 - 'TF_LENET' refers to  TF LENET)
        
    directory: Directory where the train images are placed in format .ppm
    """
    os.chdir(directory)
    X_train, Y_train=data_adequacy(directory,train=True)
    os.chdir('..') 
    os.chdir('..')
    if model=='1' or model=='SK_LR':
        model_train_path=os.getcwd()+'\\models\\model1'
        X_train,Y_train=preprocessingdata(X_train,Y_train,directory)
        SK_LR_train(X_train,Y_train,model_train_path)
        pass
    elif model=='2' or model=='TF_LR':
        model_train_path=os.getcwd()+'\\models\\model2'
        TF_LR_train(X_train,Y_train,model_train_path,rate=0.0009, epochs=300, batch_size=64)
        pass
    elif model=='3' or model=='TF_LENET':
        model_train_path=os.getcwd()+'\\models\\model3'
        TF_LENET_train(X_train,Y_train,model_train_path,epochs=50,batch_size=64,rate=0.0009)
        pass
    else:
        print('Ingrese un numero de modelo que exista')
    
@main.command("test")
@click.option('-m', '--model', help='Number of model or name. See description above')
@click.option('-d', '--directory', help='Directory where the test images are placed in format .ppm')
def test(model, directory):
    """
    Test Model # from images in directory ...  
    
    \b
    INPUT:
    model: Number of model or name
      (0 or SK_LR refers to SK logistic regression)
      (1 or TF_LR refers to  TF Logistic regression)
      (2 or TF_LENET refers to  TF LENET)
        
    directory: Directory where the test images are placed in format .ppm
    """
    os.chdir(directory)
    X_test, Y_test=data_adequacy(directory,train=False)
    os.chdir('..') 
    os.chdir('..')
    if model=='1' or model=='SK_LR':
        model_train_path=os.getcwd()+'\\models\\model1'
        X_test, Y_test=preprocessingdata(X_test, Y_test, directory)
        SK_LR_test(X_test,Y_test,model_train_path)
        pass
    elif model=='2' or model=='TF_LR':
        model_train_path=os.getcwd()+'\\models\\model2'
        TF_LR_test(X_test,Y_test,model_train_path,batch_size=64)
        pass
    elif model=='3' or model=='TF_LENET':
        model_train_path=os.getcwd()+'\\models\\model3'
        TF_LENET_test(X_test,Y_test,model_train_path,batch_size=64)
        pass

@main.command("infering")
@click.option('-m', '--model', help='Number of model or name. See description above')
@click.option('-d', '--directory', help='Directory where the infer or user images are placed in format .ppm')
def infer(model, directory):
    """
    Infer from model # from images in directory  
    
    \b
    INPUT:
    model: Number of model or name
      (0 or SK_LR refers to SK logistic regression)
      (1 or TF_LR refers to  TF Logistic regression)
      (2 or TF_LENET refers to  TF LENET)
        
    directory: Directory where the infer or user images are placed in format .ppm
    """
    os.chdir(directory)
    X_infer, Y_infer=data_adequacy(directory,train=False)
    os.chdir('..') 
    os.chdir('..')
    if model=='1' or model=='SK_LR':
        model_train_path=os.getcwd()+'\\models\\model1'
        X_infer_pre, Y_infer_pre=preprocessingdata(X_infer, Y_infer, directory)
        predict=SK_LR_infer(X_infer_pre,model_train_path)
        infer_plot(X_infer,predict,Y_infer_pre)
        pass
    elif model=='2' or model=='TF_LR':
        model_train_path=os.getcwd()+'\\models\\model2'
        predict=TF_LR_infer(X_infer,model_train_path)
        Y_infer_pre=inverse_ohe(directory,Y_infer)
        infer_plot(X_infer,predict,Y_infer_pre)
        pass
    elif model=='3' or model=='TF_LENET':
        model_train_path=os.getcwd()+'\\models\\model3'
        predict=TF_LENET_infer(X_infer,model_train_path)
        Y_infer_pre=inverse_ohe(directory,Y_infer)
        infer_plot(X_infer,predict,Y_infer_pre)
        pass
    
if __name__ == '__main__':
    main(obj={})