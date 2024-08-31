import tensorflow as tf
keras = tf.keras
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model 
import csv


def PrepareDataForModel(images_in, stage):
    """
    Uses MobileNetV2 and in some cases also a previously trained transfer model to prepare high
    level features to be used as inputs to the outcome predictor models. The extra transfer 
    learning step was found to be useful for the PN and 4 cell stages.
    """
    # Build a model that extracts high level features to represent an image using the keras 
    # MobileNetV2 model. 
    mobile =  tf.keras.applications.MobileNetV2()    
    x = mobile.layers[-2].output
    model = Model(inputs = mobile.input, outputs = x)
    # Use the model to extract high level features for each of the input images
    convx = model.predict(images_in)
    features = convx

    # Get further high level features from transfer model if stage is PN or 4 cell (the stages
    # this was shown to be useful for).
    if stage == 'PN' or stage == '4cell':
        TransferModel = 'transfermodel/model640.h5'
        topmodel = load_model(TransferModel)
        layer_name = 'dense'
        intermediate_layer_model = keras.Model(inputs = topmodel.input,
                                           outputs = topmodel.get_layer(layer_name).output)
        features = intermediate_layer_model.predict(convx)
           
    return features


def predict_all_scores(features, stage):
    """
    Get a prediction of live birth for each image using each of the 50 provided models for this 
    developmental stage
    """
    base_folder  ='Models/'+stage
    directory = os.fsencode(base_folder)
    all_scores_array = []

    # Loop through each model and predict live birth for all the images using that model
    for file in os.listdir(directory):
        model_name = os.fsdecode(file)
        model = load_model(base_folder+'/'+model_name)
        pred = model.predict(features)
        # Record the model score for each embryo 
        iter_array = []
        for emb in range(0,len(features)):    
            iter_array.append(pred[emb][1])
        all_scores_array.append(iter_array)

    return all_scores_array


def calculate_average_scores(images_in, features, all_scores_array):
    """
    Calculate the average score across all models for each embryo.
    """
    average_score = []
    average_score.append(['Image file', 'Average model score'])
    # Iterate over every embryo in the dataset
    for emb in range(0, len(features)):  
        tot = 0
        # Iterate over every model score to get sum of all scores
        for mod in range(0, 50):
            tot = tot+all_scores_array[mod][emb]
        # Find average score by diving total by number of models (50), and then add this to a list 
        # of average scores
        average_score.append([images_in.filenames[emb], (tot/50)])
    
    return average_score


def save_as_csv(output_folder, stage, average_score):
    """
    Export list of average model scores for each embryo as a csv.
    """
    with open(output_folder+stage+'_Model_Results.csv', 'w', newline = '') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerows(average_score) 

    return


def getting_predictions_main(config):
    """
    Predict likelihood of live birth after IVF transfer for every embryo image in a folder and 
    export predictions to a csv. The developmental stage of the embryos should be specified so
    models trained on the correct stage are used.
    """
    # Read in input folder, output folder, and developmental stage from config folder
    stage = config['get_predictions']['stage']
    input_folder = config['get_predictions']['input_folder']
    output_folder = config['get_predictions']['output_folder']
    
    # Read in images from input folder
    images_in = ImageDataGenerator(preprocessing_function = tf.keras.applications.mobilenet.preprocess_input) \
            .flow_from_directory(directory = input_folder, target_size = (224,224), classes = [stage],batch_size = 10,shuffle = False )

    # Convert images to comvolutional features (the format required for the live birth prediction
    # models) using the PrepareDataForModel
    features = PrepareDataForModel(images_in, stage)
        
    # Run all live birth prediction models on each image and record all scores. Produces a 2d 
    # array, each row is all the embryo scores for each model
    all_scores_array = predict_all_scores(features, stage)
        
    # Calculate the average score for each embryo by averaging over the columns in the array
    average_score = calculate_average_scores(images_in, features, all_scores_array)

    # Produce an excel file reporting the average score for each embryo 
    save_as_csv(output_folder, stage, average_score)

    return