This repository contains the code accompanying the manuscript 'deep learning pipeline reveal key moments in human embryonic development predictive of live birth in IVF'.

Getting_predictions.py takes in images of embryos at a given stage and returns an output csv file containing a live birth prediction for each image. Follow the steps below: 

1. Copy the input images you wish to use into the correct subfolder for the developmental stage they are at in the ImagesforPrediction folder. Each image should be a timelapse frame extracted at the pivotal moments in development described in the manuscript. For all stages except blastocyst these images should be cropped to 300x300 centered on the embryo. 
2. Open Getting_predictions.py and edit line 10 so the Stage is set to the developmental stage of your images, choose from the options in the comment in the line above
3. Run Getting_predictions.py
4. Navigate to the subfolder for your developmental stage in the Results folder to find the output csv that will give a live birth prediction for each input image.