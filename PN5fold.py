from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
keras = tf.keras
print('imported tensorflow and keras')
#from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
import requests
#from io import BytesIO
import os
import random
from PIL import Image, ImageFile, ImageOps
from tensorflow.keras.models import load_model
import seaborn as sns
from shutil import copyfile
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
NUMCORES=int(os.getenv("NSLOTS",1))
print("Using", NUMCORES, "core(s)" )
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUMCORES, inter_op_parallelism_threads=NUMCORES,
   allow_soft_placement=True, device_count = {'CPU': NUMCORES}))
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import sys
import shutil
import scipy.stats as ss
from scipy.stats import spearmanr
import math
########################################################################################
themodel='AllResults/TransferModels/FiveStages/models/model640.h5'
modeltype='UncropFiveStagesallt1'
BLR=0.001
stage='PN'
HU=640
bl=0.001
numofepochs=360
numiter=50
batchsize=128
testset='comparisontestset'
basefolder='AllResults/TransferModels/'+str(modeltype)+'/'+testset+'/'+str(batchsize)+'/'
class0='alldatacorrected/'+stage+'/1/data/A'
class1='alldatacorrected/'+stage+'/1/data/B'
val0_size=48
val1_size=25





def GetImageListForClass(theclass):
   # returns lists of the names of every image in each class."
    imagelist=[]
    directory = os.fsencode(theclass)
    for file in os.listdir(directory):
        filename = os.fsdecode(file) 
        imagelist.append(filename)

    return imagelist

def getlists(testset):
   # returns lists of the names of every image in each class."
    
   
    csvf= testset+'/twinsB.csv'
    sh = pd.read_csv(csvf)
    twinsB=[]
    for i in range(0,11):
        twinsB.append([sh['twin1'][i], sh['twin2'][i]])
    csvf= testset+'/twinsA.csv'
    sh = pd.read_csv(csvf)
    twinsA=[]
    for i in range(0,87):
        twinsA.append([sh['twin1'][i], sh['twin2'][i]])
    csvf= testset+'/twinsAp.csv'
    sh = pd.read_csv(csvf)
    twinsAp=[]
    for i in range(0,17):
        twinsAp.append([sh['twin1'][i], sh['twin2'][i]])

    with open(testset+"/singlesA.txt", 'r') as f:
        singlesA=[line.rstrip('\n') for line in f]
    with open(testset+"/singlesAp.txt", 'r') as f:
        singlesAp=[line.rstrip('\n') for line in f]
    with open(testset+"/singlesB.txt", 'r') as f:
        singlesB=[line.rstrip('\n') for line in f]
    with open(testset+"/singlesBp.txt", 'r') as f:
        singlesBp=[line.rstrip('\n') for line in f]


    return twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp;



def RandomExtraction(twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp,imagelist0,imagelist1,test0_size, test1_size):
    #get a random list of the images to be in the test set for each class. This is done by randomly extracting from imagelist0 and 1 and checking for overlaps
 

    vallist=[]
    for i in range(0,20):
        addition=random. choice(twinsA)
        twinsA.remove(addition)
        vallist.append(addition[0])
        vallist.append(addition[1])
    
    for i in range(0,4):
        addition=random. choice(twinsAp)
        twinsAp.remove(addition)
        vallist.append(addition[0])
        vallist.append(addition[1])
    
    
    for i in range(0,42):
        addition=random. choice(singlesA)
        singlesA.remove(addition)
        vallist.append(addition)

    for i in range(0,8):
        addition=random. choice(singlesAp)
        singlesAp.remove(addition)
        vallist.append(addition)
    
    for i in range(0,38):
        addition=random. choice(singlesB)
        singlesB.remove(addition)
        vallist.append(addition)

    for i in range(0,8):
        addition=random. choice(singlesBp)
        singlesBp.remove(addition)
        vallist.append(addition)
    
    for i in range(0,2):
        addition=random. choice(twinsB)
        twinsB.remove(addition)
        vallist.append(addition[0])
        vallist.append(addition[1]) 

   
    
    TestList_first_5 = set([l[0:5] for l in vallist])
    imageList0_5 = set([l[0:5] for l in imagelist0])
    imageList1_5 = set([l[0:5] for l in imagelist1])
    in_both0 = TestList_first_5 & imageList0_5
    in_both1 = TestList_first_5 & imageList1_5
    test0list = [l for l in imagelist0 if l[0:5] in in_both0]
    test1list = [l for l in imagelist1 if l[0:5] in in_both1]
    imageList0new = [l for l in imagelist0 if l[0:5] not in in_both0]
    imageList1new = [l for l in imagelist1 if l[0:5] not in in_both1]
       

    
    return test0list, test1list, imageList0new, imageList1new,twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp

def TestExtraction(imagelist0,imagelist1, newTestlist,twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp):
    #get a random list of the images to be in the test set for each class. This is done by randomly extracting from imagelist0 and 1 and checking for overlaps
 
#this has been edited so that only embryos that are in both classes are picked. images are selected from class 0 and then the same image is selected from class 1

#For images in TestList, if first five chars present in imageList0 then add file to test0list and remove from imageList0

    for ij in newTestlist:
        if ij in twinsA:
            twinsA.remove(ij)
        if ij in twinsAp:
            twinsAp.remove(ij)
        if ij in twinsB:
            twinsB.remove(ij)
        if ij in singlesA:
            singlesA.remove(ij)
        if ij in singlesAp:
            singlesAp.remove(ij)
        if ij in singlesB:
            singlesB.remove(ij)
        if ij in singlesBp:
            singlesBp.remove(ij)


    TestList_first_5 = set([l[0:5] for l in newTestlist])
    print(TestList_first_5)
    imageList0_5 = set([l[0:5] for l in imagelist0])
    imageList1_5 = set([l[0:5] for l in imagelist1])
    in_both0 = TestList_first_5 & imageList0_5
    in_both1 = TestList_first_5 & imageList1_5
    test0list = [l for l in imagelist0 if l[0:5] in in_both0]
    test1list = [l for l in imagelist1 if l[0:5] in in_both1]
    imageList0new = [l for l in imagelist0 if l[0:5] not in in_both0]
    imageList1new = [l for l in imagelist1 if l[0:5] not in in_both1]
    

        
  
   
   
    return test0list, test1list,imageList0new, imageList1new,twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp

def MakeDic(testlist0,testlist1):
    TestDic={}
    for i in testlist0:
        TestDic[i]=[7]
    for i in testlist1:
        TestDic[i]=[8]
    return TestDic

def Sortandprepareforbasemodel(list0, list1, class0, class1, outputfolder, a, group):
    #writes out the random classes and names of the images in each class to the outputfolder. Also created the numpy arrays for training and test X data
    the_path = outputfolder+str(a)+'/'+group
    os.mkdir(outputfolder+str(a)+'/'+group)
    os.mkdir(outputfolder+str(a)+'/'+group+'/0')
    os.mkdir(outputfolder+str(a)+'/'+group+'/1')

    directory = os.fsencode(class0)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename in list0:
            copyfile(class0+"/"+filename, the_path+'/0/'+filename)
            if group=='train':
                img=Image.open(the_path+'/0/'+filename)
                for rot in [90,180,270]:
                    rotim=img.rotate(rot)
                    rotim.save(the_path+'/0/'+str(rot)+filename)
                mirimage=ImageOps.mirror(img)#this is a horizintalflip
                mirimage.save(the_path+'/0/'+'mir'+filename)

    directory = os.fsencode(class1)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename in list1:
            copyfile(class1+"/"+filename, the_path+'/1/'+filename)
            if group=='train':
                img=Image.open(the_path+'/1/'+filename)
                for rot in [90,180,270]:
                    rotim=img.rotate(rot)
                    rotim.save(the_path+'/1/'+str(rot)+filename)
                mirimage=ImageOps.mirror(img)#this is a horizintalflip
                mirimage.save(the_path+'/1/'+'mir'+filename)

        
    if group=='test':
        the_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input) \
        .flow_from_directory(directory=the_path, target_size=(224,224), classes=['0', '1'], batch_size=10,shuffle=False )
        print('notshuffled')
    else:
        the_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input) \
        .flow_from_directory(directory=the_path, target_size=(224,224), classes=['0', '1'], batch_size=10, shuffle=False)
        print('shuffled')
    
    return the_batches



def Preparedatafortopmodel(imagesin):
    y=imagesin.classes
    y = tf.keras.utils.to_categorical(y, 2)
   # IMG_SHAPE = (224, 224, 3)
    mobile =  tf.keras.applications.MobileNetV2()
    
    x=mobile.layers[-2].output
    model = Model(inputs=mobile.input, outputs=x)
    convx2=model.predict(imagesin)
    
    topmodel=load_model(themodel)
    
    layer_name = 'dense'
    intermediate_layer_model = keras.Model(inputs=topmodel.input,
                                       outputs=topmodel.get_layer(layer_name).output)
    convx = intermediate_layer_model.predict(convx2)
    
    x=topmodel.layers[-1].output
    #newmodel = Model(inputs=topmodel.input, outputs=x)
    #print(newmodel.summary)
    #convx=topmodel.predict(convx2)
    #rint("Shape of convx: {}".format(convx.shape))
    #basemodel = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
    #global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    #convx = global_average_layer(basemodel(imagesin))
           
    return convx, y

def PreparedatafortopmodelNT(imagesin):
    y=imagesin.classes
    y = tf.keras.utils.to_categorical(y, 2)
   # IMG_SHAPE = (224, 224, 3)
    mobile =  tf.keras.applications.MobileNetV2()
    x=mobile.layers[-2].output
    model = Model(inputs=mobile.input, outputs=x)
    convx=model.predict(imagesin)
    #basemodel = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
    #global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    #convx = global_average_layer(basemodel(imagesin))
           
    return convx, y



def MakeandcompileFCModel(num_classes,BLR,drop,HU):
    prediction_layer = keras.layers.Dense(num_classes, activation='softmax', input_shape=(HU,))
    model = tf.keras.Sequential()
    if drop>0.001:
        model.add(keras.layers.Dropout(drop))
    model.add(prediction_layer)
    base_learning_rate = BLR
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def RunModelAndSaveResults2(all_val_auc,all_accum_losses, all_accum_val_losses, Xtrain, Ytrain, model, i,Xval,Yval, trainlst0, trainlst1,weight,EN,allvalprec, vallist0,vallist1,allvaloverfit,allpositive, allslope):
    val_loss_best=0.01    
    loss_accum=[]
    val_loss_accum=[]
    aucaccum=[]
    epochaccum=[]
    class_weight = {0: 1.,
                    1: weight}
    
              
    epochnum=0   
    valprec=0
    allP=0
    auc_score_best=0
    for j in range(0,numofepochs,10):
        
        
        train_history = model.fit(Xtrain, Ytrain, batch_size=batchsize, epochs=10,verbose=2, validation_data=(Xval, Yval), class_weight=class_weight)
        loss = train_history.history['accuracy']
        val_loss = train_history.history['val_accuracy'] 
        actual_val_loss=train_history.history['val_loss'] 
        loss_accum=loss_accum+loss
        
        val_loss_accum=val_loss_accum+val_loss
        y_val_cat_prob=model.predict_proba(Xval)[:, 1]
        y_val_cat=Prep_val_images.classes
        auc_score=roc_auc_score(y_val_cat,y_val_cat_prob)
        aucaccum.append(auc_score)
        epochaccum.append(j)
        if auc_score>=auc_score_best:
            val_loss_best=val_loss[9]
            auc_score_best=auc_score
            loss_best=loss[9]
            overfit=loss_best-val_loss_best
            model.save(outputfolder+model_name+str(i)+'.h5')
            epochnum=j+10
            
            #work out validation precision at point model was stopped
    model = load_model(outputfolder+model_name+str(i)+'.h5')
    p=model.predict(Xval)
    count=0
    TP=0
    TN=0
    FP=0
    FN=0
            
            #Find TN, TP, FN and FP so we can calculate the confusion matrix values
    for h in range(0,len(vallist0)):    
        if p[h][0]<0.5:
            FN=FN+1
        else:        
            TP=TP+1
    for h in range(len(vallist0),len(vallist0)+len(vallist1)):    
        if p[h][0]>0.5:
            FP=FP+1
        else:        
            TN=TN+1
        if (TP+FP)>0:
            prec=TP/(TP+FP)
        else:
            prec=0
        valprec=prec
        allP=TP+FP
            
    
    #make training graph for training run
    plt.plot(loss_accum)
    plt.plot(val_loss_accum)
    plt.legend(['Training accuracy ', 'Test accuracy'])
    plt.savefig(outputfolder+str(i)+individual_run_graph_name)
    plt.clf()
    
    plt.scatter(epochaccum, aucaccum) 
    plt.savefig(outputfolder+str(i)+'rocauc_vs_epoch.jpg')
    plt.clf()  
    
    yvals=val_loss_accum[5:8]
    xvals=list(range(5,8))
    slope, intercept = np.polyfit(xvals, yvals, 1)

    #Plot
    plt.figure()
    plt.scatter(xvals, yvals) 
    plt.plot(np.unique(xvals), np.poly1d(np.polyfit(xvals, yvals, 1))(np.unique(xvals)), color = 'k')
    plt.savefig(outputfolder+str(i)+'endslope.jpg')
    
    #add training and valisation time series to main list to be included in the averae graph later
    all_accum_losses.append(loss_accum)
    all_accum_val_losses.append(val_loss_accum)
    all_val_auc.append(aucaccum)
    EN.append(epochnum)
    allvalprec.append(valprec)
    allpositive.append(allP)
    allvaloverfit.append(overfit)
    allslope.append(slope)
    
    return all_accum_losses, all_accum_val_losses, loss_best, epochaccum

def RunModelAndSaveResults(all_val_auc,all_accum_losses, all_accum_val_losses, Xtrain, Ytrain, model, i,Xval,Yval, trainlst0, trainlst1,weight,EN,allvalprec, vallist0,vallist1,allvaloverfit,allpositive, allslope):
    val_loss_best=0.01    
    loss_accum=[]
    val_loss_accum=[]
    aucaccum=[]
    epochaccum=[]
    class_weight = {0: 1.,
                    1: weight}
    
              
    epochnum=0   
    valprec=0
    allP=0
    auc_score_best=0
    for j in range(0,numofepochs,10):
        
        
        train_history = model.fit(Xtrain, Ytrain, batch_size=batchsize, epochs=10,verbose=2, validation_data=(Xval, Yval), class_weight=class_weight)
        loss = train_history.history['accuracy']
        val_loss = train_history.history['val_accuracy'] 
        actual_val_loss=train_history.history['val_loss'] 
        loss_accum=loss_accum+loss
        
        val_loss_accum=val_loss_accum+val_loss
        y_val_cat_prob=model.predict_proba(Xval)[:, 1]
        y_val_cat=Prep_val_images.classes
        auc_score=roc_auc_score(y_val_cat,y_val_cat_prob)
        aucaccum.append(auc_score)
        epochaccum.append(j)
        #if auc_score>=auc_score_best:
        val_loss_best=val_loss[9]
        auc_score_best=auc_score
        loss_best=loss[9]
        overfit=loss_best-val_loss_best
        model.save(outputfolder+model_name+str(i)+'.h5')
        epochnum=j+10
            
            #work out validation precision at point model was stopped
    model = load_model(outputfolder+model_name+str(i)+'.h5')
    p=model.predict(Xval)
    count=0
    TP=0
    TN=0
    FP=0
    FN=0
            
            #Find TN, TP, FN and FP so we can calculate the confusion matrix values
    for h in range(0,len(vallist0)):    
        if p[h][0]<0.5:
            FN=FN+1
        else:        
            TP=TP+1
    for h in range(len(vallist0),len(vallist0)+len(vallist1)):    
        if p[h][0]>0.5:
            FP=FP+1
        else:        
            TN=TN+1
        if (TP+FP)>0:
            prec=TP/(TP+FP)
        else:
            prec=0
        valprec=prec
        allP=TP+FP
            
    
    #make training graph for training run
    plt.plot(loss_accum)
    plt.plot(val_loss_accum)
    plt.legend(['Training accuracy ', 'Test accuracy'])
    plt.savefig(outputfolder+str(i)+individual_run_graph_name)
    plt.clf()
    
    plt.scatter(epochaccum, aucaccum) 
    plt.savefig(outputfolder+str(i)+'rocauc_vs_epoch.jpg')
    plt.clf()  
    
    yvals=val_loss_accum[5:8]
    xvals=list(range(5,8))
    slope, intercept = np.polyfit(xvals, yvals, 1)

    #Plot
    plt.figure()
    plt.scatter(xvals, yvals) 
    plt.plot(np.unique(xvals), np.poly1d(np.polyfit(xvals, yvals, 1))(np.unique(xvals)), color = 'k')
    plt.savefig(outputfolder+str(i)+'endslope.jpg')
    
    #add training and valisation time series to main list to be included in the averae graph later
    all_accum_losses.append(loss_accum)
    all_accum_val_losses.append(val_loss_accum)
    all_val_auc.append(aucaccum)
    EN.append(epochnum)
    allvalprec.append(valprec)
    allpositive.append(allP)
    allvaloverfit.append(overfit)
    allslope.append(slope)
    
    return all_accum_losses, all_accum_val_losses, loss_best, epochaccum

def Outputtestresults(TestDictraw,valrho,testrho,TestDict,acc_overfit,precisionscores,NPVscores,sensitivityscores,selectivityscores,alltestacc,outputfolder,i,Xtest, testlist0, testlist1, Prep_test_images, Xval, vallist0, vallist1, allvalacc, trainacc, alltestoverfit,rocauc,prauc,F1,Xtrain,Prep_train_images,valrocauc,valprauc,valF1,valsensitivity,valselectivity,Prep_val_images):
    #produce numerical results of test set when model was optimal. Also produce images in the outout folder of the test set failures
    model = load_model(outputfolder+model_name+str(i)+'.h5')
    p=model.predict(Xtest)
    np.savetxt(outputfolder+str(i)+"/testresults", p)
    count=0
    TP=0
    TN=0
    FP=0
    FN=0
    testnames=[]
    for file in Prep_test_images.filenames:
        testnames.append(file[2:])
    
    #Find TN, TP, FN and FP so we can calculate the confusion matrix values
    for h in range(0,len(testlist0)):  
        TestDictraw[testnames[h]].append(p[h][1])  
        if p[h][0]<0.5:
            #copyfile(outputfolder+str(i)+'/test'+'/0/'+testnames[h], outputfolder +str(i)+ '/testsetfailures/'+str(h)+'class0.png')
            FN=FN+1
            TestDict[testnames[h]].append(1)
        else:        
            TP=TP+1
            TestDict[testnames[h]].append(0)
    for h in range(len(testlist0),len(testlist0)+len(testlist1)):   
        TestDictraw[testnames[h]].append(p[h][1])  
        if p[h][0]>0.5:
            #copyfile(outputfolder+str(i)+'/test'+'/1/'+testnames[h], outputfolder +str(i)+ '/testsetfailures/'+str(h)+'class1.png')
            FP=FP+1
            TestDict[testnames[h]].append(0)
        else:        
            TN=TN+1
            TestDict[testnames[h]].append(1)
            
    #calculate the confusion matrix variables and add to the list so we can work out an average later
    if (TP+FP)>0:
        precisionscores.append(TP/(TP+FP))
    else:
        precisionscores.append(0.5)
    if (TN+FN)>0:
        NPVscores.append(TN/(TN+FN))
    else:
        NPVscores.append(0.5)
    sensitivityscores.append(TP/(TP+FN))
    selectivityscores.append(TN/(TN+FP))
    alltestacc.append((TP+TN)/int(len(testlist0)+len(testlist1)))
    testacc=(TP+TN)/int(len(testlist0)+len(testlist1))
    if (trainacc-testacc)>0:
        acc_overfit.append(trainacc-testacc)
    else:
        acc_overfit.append(0)
    if (TP+0.5*(FP+FN))>0:
        f1score=TP/(TP+0.5*(FP+FN))
        F1.append(f1score)
    else:
        F1.append(0.5)

    ####################################################lift test
    y_hat=model.predict_proba(Xtest)[:, 1]
    y_hat_rank=ss.rankdata(y_hat)
    testsize=len(testlist0)+len(testlist1)
    bucketsize=testsize/10
    bucketfrequency=[0,0,0,0,0,0,0,0,0,0]
    buckettotal=[]
    for j in range(0,10):
        ii=j*bucketsize+1
    
        if float. is_integer(ii)==True and float. is_integer(ii+bucketsize)==True:
            buckettotal.append(bucketsize)
        if float. is_integer(ii)==True and float. is_integer(ii+bucketsize)==False:
            buckettotal.append(int(ii+bucketsize)-int(ii)+1)
        if float. is_integer(ii)==False and float. is_integer(ii+bucketsize)==True:
            buckettotal.append(int(ii+bucketsize)-int(ii)-1)
        if float. is_integer(ii)==False and float. is_integer(ii+bucketsize)==False:
            buckettotal.append(int(bucketsize+ii)-int(ii))
    
    for h in range(len(testlist0),len(testlist0)+len(testlist1)): 
        bucketnum=0
        for j in range(0,10):
            ii=j*bucketsize+1
            if y_hat_rank[h]>=ii and y_hat_rank[h]<ii+bucketsize:
                bucketfrequency[bucketnum]+=1
            bucketnum+=1
    bucketlift=[]
    bucketrate=[]
    for j in range(0,10):
        bucketrate.append((bucketfrequency[j]/buckettotal[j]))
        bucketlift.append((bucketfrequency[j]/buckettotal[j])/0.367)

    xlist=[1,2,3,4,5,6,7,8,9,10]
    rho, p = spearmanr(xlist,bucketrate)
    plt.clf() 
    plt.scatter(xlist, bucketrate) 
    plt.savefig(outputfolder+str(i)+'testrate_rho_is_'+str(rho)+'_p_is'+str(p)+'.jpg')
    plt.clf() 
    rho, p = spearmanr(xlist,bucketlift)
    plt.scatter(xlist, bucketlift) 
    plt.savefig(outputfolder+str(i)+'testlift_rho_is_'+str(rho)+'_p_is'+str(p)+'.jpg')
    plt.clf() 
    testrho.append(rho)


    ####################################################lift test
    y_hat=model.predict_proba(Xval)[:, 1]
    y_hat_rank=ss.rankdata(y_hat)
    testsize=len(vallist0)+len(vallist1)
    bucketsize=testsize/10
    bucketfrequency=[0,0,0,0,0,0,0,0,0,0]
    buckettotal=[]
    for j in range(0,10):
        ii=j*bucketsize+1
    
        if float. is_integer(ii)==True and float. is_integer(ii+bucketsize)==True:
            buckettotal.append(bucketsize)
        if float. is_integer(ii)==True and float. is_integer(ii+bucketsize)==False:
            buckettotal.append(int(ii+bucketsize)-int(ii)+1)
        if float. is_integer(ii)==False and float. is_integer(ii+bucketsize)==True:
            buckettotal.append(int(ii+bucketsize)-int(ii)-1)
        if float. is_integer(ii)==False and float. is_integer(ii+bucketsize)==False:
            buckettotal.append(int(bucketsize+ii)-int(ii))
    
    for h in range(len(vallist0),len(vallist0)+len(vallist1)): 
        bucketnum=0
        for j in range(0,10):
            ii=j*bucketsize+1
            if y_hat_rank[h]>=ii and y_hat_rank[h]<ii+bucketsize:
                bucketfrequency[bucketnum]+=1
            bucketnum+=1
    bucketlift=[]
    bucketrate=[]
    for j in range(0,10):
        bucketrate.append((bucketfrequency[j]/buckettotal[j]))
        bucketlift.append((bucketfrequency[j]/buckettotal[j])/0.367)

    xlist=[1,2,3,4,5,6,7,8,9,10]
    rho, p = spearmanr(xlist,bucketrate)
    plt.clf() 
    plt.scatter(xlist, bucketrate) 
    plt.savefig(outputfolder+str(i)+'valrate_rho_is_'+str(rho)+'_p_is'+str(p)+'.jpg')
    plt.clf() 
    rho, p = spearmanr(xlist,bucketlift)
    plt.scatter(xlist, bucketlift) 
    plt.savefig(outputfolder+str(i)+'vallift_rho_is_'+str(rho)+'_p_is'+str(p)+'.jpg')
    plt.clf() 
    valrho.append(rho)


    



        


    p=model.predict(Xval)    
    countval=0  
    TP=0
    TN=0
    FP=0
    FN=0
      
    for h in range(0,len(vallist0)):    
        if p[h][0]>0.5:
            countval=countval+1  
            TP=TP+1
        else:
            FN=FN+1    
    for h in range(len(vallist0),len(vallist0)+len(vallist1)):    
        if p[h][0]<0.5:
            countval=countval+1  
            TN=TN+1
        else:
            FP=FP+1
    allvalacc.append(countval/int(len(vallist0)+len(vallist1)))
    valsensitivity.append(TP/(TP+FN))
    valselectivity.append(TN/(TN+FP))
    if (TP+0.5*(FP+FN))>0:
        f1score=TP/(TP+0.5*(FP+FN))
        valF1.append(f1score)
    else:
        valF1.append(0.5)

    y_val_cat_prob=model.predict_proba(Xval)[:, 1]
    y_val_cat=Prep_val_images.classes
    auc_score=roc_auc_score(y_val_cat,y_val_cat_prob)
    lr_precision, lr_recall, _ = precision_recall_curve(y_val_cat,y_val_cat_prob)
    #lr_f1=f1_score(y_val_cat,y_val_cat_prob)
    pr_auc =auc(lr_recall, lr_precision)
    
    valrocauc.append(auc_score)
    valprauc.append(pr_auc)

    y_val_cat_prob=model.predict_proba(Xtrain)[:, 1]
    y_val_cat=Prep_train_images.classes
    auc_score_train=roc_auc_score(y_val_cat,y_val_cat_prob)


    
    y_val_cat_prob=model.predict_proba(Xtest)[:, 1]
    y_val_cat=Prep_test_images.classes
    auc_score=roc_auc_score(y_val_cat,y_val_cat_prob)
    lr_precision, lr_recall, _ = precision_recall_curve(y_val_cat,y_val_cat_prob)
    #lr_f1=f1_score(y_val_cat,y_val_cat_prob)
    pr_auc =auc(lr_recall, lr_precision)
    
    rocauc.append(auc_score)
    prauc.append(pr_auc)

    
    if auc_score<= auc_score_train:
        alltestoverfit.append(auc_score_train-auc_score)
    else:
        alltestoverfit.append(0)


    
    plt.clf()
    lr_fpr, lr_tpr, _ = roc_curve(y_val_cat,y_val_cat_prob)
    ns_probs = [0 for _ in range(len(y_val_cat))]
    ns_fpr, ns_tpr, _ = roc_curve(y_val_cat, ns_probs)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')    
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Model')
# axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
# show the legend
    plt.legend()
# show the plot
    plt.savefig(outputfolder+str(i)+'roc_curve.jpg')
    plt.clf()
    
                  
    return 

def SaveDataFromAllRunsAndMakeAverageraph(valrocauc,all_val_auc,epochaccum, precisionscores,NPVscores,sensitivityscores,selectivityscores,alltestacc, all_accum_val_losses,all_accum_losses, outputfolder, allvalacc,EN,allvalprec, allpositive, alltestoverfit, allvaloverfit, rocauc,prauc,F1,allslope):
    #Save the multi dimaensioanl array of all training and validation score at each eproch for each model to csv files
    #Also produce the final average graph
    
    #save all data
    b = np.asarray(all_accum_val_losses)
    np.savetxt(outputfolder+all_val_losses, b, delimiter=",")
    b = np.asarray(all_accum_losses)
    np.savetxt(outputfolder+all_training_losses, b, delimiter=",")

    #make final graph
    av_vals=[0,0]
    for x in range(0,numofepochs):
        av_val_loss=0
        for y in range(0,numiter):
            av_val_loss=av_val_loss+all_accum_val_losses[y][x]
        av_vals=av_vals+[(av_val_loss/numiter)]

    av_train_vals=[0,0]
    for x in range(0,numofepochs):
        av_train_val_loss=0
        for y in range(0,numiter):
            av_train_val_loss=av_train_val_loss+all_accum_losses[y][x]
        av_train_vals=av_train_vals+[(av_train_val_loss/numiter)]
            
    av_auc=[]
    epochaccum=[]
    bestauc=0
    bestE=0

    for x in range(0,int(numofepochs/10)):
        enum=x*10
        epochaccum.append(enum)
        av_train_val_loss=0
        for y in range(0,numiter):
            av_train_val_loss=av_train_val_loss+all_val_auc[y][x]
        av_auc.append((av_train_val_loss/numiter))
        if (av_train_val_loss/numiter)>bestauc:
            bestauc=(av_train_val_loss/numiter)
            bestE=enum


    


    plt.plot(av_train_vals[2:numofepochs], 'r')
    plt.plot(av_vals[2:numofepochs], 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training accuracy', 'Validation accuracy'])
    plt.savefig(outputfolder+average_runs_graph_name)
    plt.clf() 

    plt.scatter(epochaccum, av_auc) 
    plt.savefig(outputfolder+'rocauc_vs_epoch_av.jpg')
    plt.clf() 
    
    
    ax = sns.violinplot(x=alltestacc)
    plt.savefig(outputfolder+'Test_accuracies.jpg')
    plt.clf() 
    
    with open(outputfolder+'test_accuracies.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(alltestacc)
    
    with open(outputfolder+'test_precision.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(precisionscores)
    with open(outputfolder+'test_NPV.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(NPVscores)
    with open(outputfolder+'test_selectivity.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(selectivityscores)
    with open(outputfolder+'test_sensitivity.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(sensitivityscores)
        
    with open(outputfolder+'Allvalacc.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(allvalacc)
        
        
    with open(outputfolder+'AllEpochNum.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(EN)
    with open(outputfolder+'Allvallprec.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(allvalprec)
    
    with open(outputfolder+'Allvaloverfit.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(allvaloverfit)
        
        
    with open(outputfolder+'Alltestoverfit.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(alltestoverfit)
    with open(outputfolder+'Allvalpositive.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(allpositive)
    
    with open(outputfolder+'Alltestrocauc.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(rocauc)
    with open(outputfolder+'Alltestprauc.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(prauc)
    with open(outputfolder+'AlltestF1.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(F1)  
    with open(outputfolder+'Allendslope.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(allslope)  

    with open(outputfolder+'Allvalroc.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(valrocauc)
    return bestE



############################################################################2cell
    

############################################################################2cell
#variables that aren't normally changed
model_name= 'model'
individual_run_graph_name='trainingrun.jpg'
average_runs_graph_name='averageofruns.jpg'
all_training_losses='traininglosses.csv'
all_val_losses='vallosses.csv'
all_accum_losses=[]
all_accum_val_losses=[]
num_classes=2
imgsize=224
#hyperparameters
offset=[1280]
blrarray=[0.0001]
#offset=[2,2.25]
drop_out=0.5
iterationdetails=[]
iterationdetails.append(['stage', 'hidden units','blr', 'drop_out','BE','Average Test','av test error', 'Precision', 'pr_er', 'NPV','NPV_er','Sensitivity','sens_er', 'Selectivity', 'sel_er', 'Overfit', 'overfit_err', 'ROC AUC', 'ROCAUCerr', 'PR AUC','PRAUCerr','F1','F1err','av_end_grad', 'err_end_grad' ,'valrocauc','valrocauc_err','valprauc','valprauc_err','valF1','valF1_err','valsensitivity','valsensitivity_err','valselectivity','valselectivity_err','acc_overfit','acc_overfit_err','testrho','testrho_err','valrho','valrho_err'])

twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp=getlists(testset)
print(len(twinsA))
trainlst0=GetImageListForClass(class0)
trainlst1=GetImageListForClass(class1)

testlists=[]
for i in range(0,4):
    vallist0, vallist1, trainlst0, trainlst1,twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp =RandomExtraction(twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp,trainlst0,trainlst1, val0_size, val1_size)  
    print(vallist0)
    print(vallist1)
    print(len(twinsA))
    for j in vallist1:
        vallist0.append(j)
    testlists.append(vallist0)
for j in trainlst1:
    trainlst0.append(j)

testlists.append(trainlst0)
print('p')
print(testlists[0])

ns=math.sqrt(numiter)
runlength=[]
for g in range(0,len(testlists)):
    Testlist=testlists[g]
    print(Testlist)
    os.mkdir(basefolder+str(BLR)+'/'+stage+'/'+str(g))
    
    #input data information
    weight=2
    all_accum_losses=[]
    all_accum_val_losses=[]
    class0_size=len([f for f in os.listdir(os.fsencode(class0))])
    class1_size=len([f for f in os.listdir(os.fsencode(class1))])
    drop=drop_out
    outputfolder=basefolder+str(BLR)+'/'+stage+'/'+str(g)+'/'
    alltestacc=[]
    precisionscores=[]
    NPVscores=[]
    sensitivityscores=[]
    selectivityscores=[]
    allvalacc=[]
    EN=[]
    allvalprec=[]
    allvaloverfit=[]
    valprauc=[]
    valsensitivity=[]
    valF1=[]
    valselectivity=[]
    valrocauc=[]
    alltestoverfit=[]
    allpositive=[]
    rocauc=[]
    prauc=[]
    F1=[]
    allslope=[]
    acc_overfit=[]
    valrho=[]
    testrho=[]
    all_val_auc=[]
        



            #get lists of the names of the images in each class. this is used to make the list of randomly selected test names

    for a in range(0,numiter):
        os.mkdir(basefolder+str(BLR)+'/'+stage+'/'+str(g)+'/'+str(a))
        twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp=getlists(testset)
        imagelist0=GetImageListForClass(class0)
        imagelist1=GetImageListForClass(class1)
        print('k')
        print(Testlist)
        print(imagelist0)
        print(singlesA)
        testlist0, testlist1,trainlst0, trainlst1,twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp=TestExtraction(imagelist0,imagelist1,Testlist,twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp)
        print(testlist0)
        print(testlist1)
        
       
            #get lists of which image is to be in which class (only call it twice because the train list is the leftover and just gets overwritten if you call the 2nd line too  t)
        #testlist0, testlist1,imagelist0_2,imagelist1_2,twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp=RandomExtraction(twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp,imagelist0,imagelist1, val0_size, val1_size)        #get lists of which image is to be in which class (only call it twice because the train list is the leftover and just gets overwritten if you call the 2nd line too  t)
        #vallist0, vallist1, trainlst0, trainlst1,twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp =RandomExtraction(twinsA,twinsAp,twinsB,singlesA,singlesAp,singlesB,singlesBp,imagelist0_2,imagelist1_2, val0_size, val1_size)  
        vallist0=testlist0
        vallist1=testlist1
        if a<1:
            TestDict=MakeDic(testlist0,testlist1)
            TestDictraw=MakeDic(testlist0,testlist1)
        #remo0,remo1,trainlst0, trainlst1 =RandomExtraction(trainlst0,trainlst1, 60, 60)  
        #prepare the IMagegenerater objects for each dataset ready to go into the base model
        Prep_train_images=Sortandprepareforbasemodel(trainlst0, trainlst1, class0, class1, outputfolder, a, 'train')
        Prep_val_images=Sortandprepareforbasemodel(vallist0, vallist1, class0, class1, outputfolder, a, 'val')
        Prep_test_images=Sortandprepareforbasemodel(testlist0, testlist1, class0, class1, outputfolder, a, 'test')
        #Get arrays of the output of the basemodel that will be used to train the top model
        if HU<1200:
            Xtrain, Ytrain=Preparedatafortopmodel(Prep_train_images)
            Xval, Yval=Preparedatafortopmodel(Prep_val_images)
            Xtest, Ytest=Preparedatafortopmodel(Prep_test_images)
        else:
            
            Xval, Yval=PreparedatafortopmodelNT(Prep_val_images)
            Xtest, Ytest=PreparedatafortopmodelNT(Prep_test_images)
            Xtrain, Ytrain=PreparedatafortopmodelNT(Prep_train_images)

        #make, run and process results of the top model            
        model=MakeandcompileFCModel(num_classes,bl,drop,HU)
        all_accum_losses, all_accum_val_losses, trainacc,epochaccum=RunModelAndSaveResults(all_val_auc,all_accum_losses, all_accum_val_losses, Xtrain, Ytrain, model, a,Xval,Yval, trainlst0, trainlst1, weight,EN,allvalprec, vallist0,vallist1, allvaloverfit, allpositive, allslope)    
        Outputtestresults(TestDictraw,valrho,testrho,TestDict,acc_overfit,precisionscores,NPVscores,sensitivityscores,selectivityscores,alltestacc,outputfolder,a,Xtest, testlist0, testlist1, Prep_test_images, Xval, vallist0, vallist1, allvalacc,trainacc, alltestoverfit, rocauc,prauc,F1,Xtrain,Prep_train_images,valrocauc,valprauc,valF1,valsensitivity,valselectivity,Prep_val_images)
        tf.keras.backend.clear_session()


            #save all data and make the final average run graph
    BE=SaveDataFromAllRunsAndMakeAverageraph(valrocauc,all_val_auc,epochaccum,precisionscores,NPVscores,sensitivityscores,selectivityscores,alltestacc, all_accum_val_losses,all_accum_losses, outputfolder,allvalacc,EN,allvalprec, allpositive, alltestoverfit, allvaloverfit,rocauc,prauc,F1,allslope)
    iterationdetails.append([stage,str(g),bl,drop_out,BE,sum(alltestacc)/len(alltestacc),(np.std(alltestacc))/ns,sum(precisionscores)/len(precisionscores),(np.std(precisionscores))/ns,sum(NPVscores)/len(NPVscores),(np.std(NPVscores))/ns,sum(sensitivityscores)/len(sensitivityscores),(np.std(sensitivityscores))/ns,sum(selectivityscores)/len(selectivityscores),(np.std(selectivityscores))/ns, sum(alltestoverfit)/len(alltestoverfit),(np.std(alltestoverfit))/ns,sum(rocauc)/len(rocauc),(np.std(rocauc))/ns,sum(prauc)/len(prauc),(np.std(prauc))/ns,sum(F1)/len(F1),(np.std(F1))/ns,sum(allslope)/len(allslope),(np.std(allslope))/ns,sum(valrocauc)/len(valrocauc),(np.std(valrocauc))/ns,sum(valprauc)/len(valprauc),(np.std(valprauc))/ns,sum(valF1)/len(valF1),(np.std(valF1))/ns,sum(valsensitivity)/len(valsensitivity),(np.std(valsensitivity))/ns,sum(valselectivity)/len(valselectivity),(np.std(valselectivity))/ns,sum(acc_overfit)/len(acc_overfit),(np.std(acc_overfit))/ns,sum(testrho)/len(testrho),(np.std(testrho))/ns,sum(valrho)/len(valrho),(np.std(valrho))/ns])
    #Add a final 8 to the test dictionary to sandwich the test numbers and print out the dictionary to a csv
    runlength.append(BE)

    for i in testlist0:
        TestDict[i].append(8)
    for i in testlist1:
        TestDict[i].append(9)

    with open(basefolder+str(BLR)+'/'+stage+'/'+str(g)+'/testresults.csv', 'w') as f:
        for key in TestDict.keys():
            f.write("%s,%s\n"%(key,TestDict[key]))

    for i in testlist0:
        TestDictraw[i].append(8)
    for i in testlist1:
        TestDictraw[i].append(9)

    with open(basefolder+str(BLR)+'/'+stage+'/'+str(g)+'/testresultsraw.csv', 'w') as f:
        for key in TestDictraw.keys():
            f.write("%s,%s\n"%(key,TestDictraw[key]))

    
import csv

with open(basefolder+str(BLR)+'/'+stage+'/Summary_'+str(stage)+str(BLR)+'_drop05_3000E.csv','w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(iterationdetails)   
    
for g in range(0,len(testlists)):
    for a in range(0,50):
        mydir= basefolder+str(BLR)+'/'+stage+'/'+str(g)+'/'+str(a)
        try:
            shutil.rmtree(mydir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


