This repo contains code to train and deploy models for predicting live birth from embryo images. Instructions for using each script are given below.

Getting_predictions.py takes in images of embryos at a given stage and returns an output csv file containing a live birth prediction for each image.

Steps:
1. copy the input images you wish to use into the correct subfolder for the developmental stage they are at in the ImagesforPrediction folder. Each image should be a timelapse frame extracted at the pivotal moments in development we have identified; 1 hr before PNBD for the 'PN' stage, the first moment two cells cn be clearly seen for the '2cell stage' ... . For all stages except blastocyst these images should be cropped to 300x300 centered on the embryo. For assistance in preparing these images see ...
2. open Getting_predictions.py and edit line x so the Stage is set to the developmental stage of your images, choose from the options in the comment in the line below
3.run Getting_predictions.py
4.Navigate to the subfolder for your developmental stage in the Results folder to find the output csv that will give a live birth prediction for each input image.
