import tensorflow as tf
keras = tf.keras
print('imported tensorflow and keras')
print(tf. __version__)
#from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
import requests
#from io import BytesIO
#import numpy as np
import os
import random
from PIL import Image, ImageFile, ImageOps
from tensorflow.keras.models import load_model
import seaborn as sns
from shutil import copyfile
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import sys
import shutil

InputFolder='C:/eMLife/stages'

ImagesIn = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input) \
        .flow_from_directory(directory=InputFolder, target_size=(224,224), classes=['2cell'],batch_size=10,shuffle=False )

print(ImagesIn.filenames)
mobile =  tf.keras.applications.MobileNetV2()
x=mobile.layers[-2].output
model = Model(inputs=mobile.input, outputs=x)
X=model.predict(ImagesIn)
    
    
basefolder='C:/eMLife/Models/2cell'
directory = os.fsencode(basefolder)

thearray=[]

for file in os.listdir(directory):
    model_name = os.fsdecode(file)
    print(model_name)
    model = load_model(basefolder+'/'+model_name)
    pred = model.predict(X)
    thearray.append(pred)

print(thearray)    
average_score=[]