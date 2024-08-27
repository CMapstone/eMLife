import tensorflow as tf
keras = tf.keras
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model 
import csv

#set Stage to be required developmental stage in line below. Choices are: 'PN', '2cell', '4cell','9cell', 'blastocyst'
Stage='9cell'

def PrepareDataForModel(imagesin, Stage):
    #Uses MobileNetV2 and in some cases also a previously trained transfer model to prepare high level features to be used as inputs to the outcome predictor models.
    #The extra transfer learning step was found to be useful for the PN and 4 cell stages.

    mobile =  tf.keras.applications.MobileNetV2()    
    x=mobile.layers[-2].output
    model = Model(inputs=mobile.input, outputs=x)
    convx=model.predict(imagesin)
    X=convx
    
    if Stage=='PN' or Stage=='4cell':
        TransferModel='transfermodel/model640.h5'
        topmodel=load_model(TransferModel)
        layer_name = 'dense'
        intermediate_layer_model = keras.Model(inputs=topmodel.input,
                                           outputs=topmodel.get_layer(layer_name).output)
        X = intermediate_layer_model.predict(convx)
           
    return X

#Read in images from input folder
InputFolder='ImagesforPrediction'
ImagesIn = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input) \
        .flow_from_directory(directory=InputFolder, target_size=(224,224), classes=[Stage],batch_size=10,shuffle=False )

#convert images to comvolutional features (the format required for the live birth prediction models) using the PrepareDataForModel
X=PrepareDataForModel(ImagesIn, Stage)
    

#run all live birth prediction models on each image and record all scores. Produces a 2d array, each row is all the embryo scores for each model
basefolder='Models/'+Stage
directory = os.fsencode(basefolder)
all_scores_array=[]
for file in os.listdir(directory):
    model_name = os.fsdecode(file)
    model = load_model(basefolder+'/'+model_name)
    pred = model.predict(X)
    iterarray=[]
    for h in range(0,len(X)):    
        iterarray.append(pred[h][1])
    all_scores_array.append(iterarray)
    

#calculate the average score for each embryo by averaging over the columns in the array
average_score=[]
average_score.append(['Image file','Average model score'])
for h in range(0,len(X)):  
    s=0
    for i in range(0,50):
        s=s+all_scores_array[i][h]
    average_score.append([ImagesIn.filenames[h],(s/50)])


#produce an excel file reporting the average score for each embryo 
with open('Results/'+Stage+'/'+Stage+'_Model_Results.csv','w', newline='') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(average_score)  