This repository contains the code accompanying the manuscript 'deep learning pipeline reveal key moments in human embryonic development predictive of live birth in IVF'.

Getting_predictions.py takes in images of embryos at a given stage and returns an output csv file containing a live birth prediction for each image. Follow the steps below: 

1. Copy the input images you wish to use into the correct subfolder for the developmental stage they are at in the ImagesforPrediction folder. Each image should be a timelapse frame extracted at the pivotal moments in development described in the manuscript. For all stages except blastocyst these images should be cropped to 300x300 centered on the embryo. 
2. Open Getting_predictions.py and edit line 10 so the Stage is set to the developmental stage of your images, choose from the options in the comment in the line above
3. Run Getting_predictions.py
4. Navigate to the subfolder for your developmental stage in the Results folder to find the output csv that will give a live birth prediction for each input image.

We have also included scripts used for pre-processing the input images, FindCroppingCoords.py and ExtractFrames.py. These can be used to extract and automatically crop frames from the timelapse videos at various offsets from our reference developmental stages. This can be done following the below steps:

1.Put all videos for which you wish to extract frames for into the input folder under Preprocessing/Input videos
2.Run FindCroppingCoords.py, this will find coordinates that can be used to automatically crop frames from the video based on the position of the embryo in the dish. These coordinates are written to an output file that will be saved under the Outputs folder called 'croppingcoords.csv'
3.Fill in Annotations.csv with the frame numbers of the reference developmental stages you want to use (currently this needs to be done manually by inspecting the videos), we reccomend using the reference moments outlined in the manuscript, and the cropping coordinates copied from croppingcoords.csv. A few rows have been given as examples.
4.Open ExtractFrames.py to set the variables defined just after the imports based on what developmental stage, offsets, and cropping option you want.
5.Run ExtractFrames.py, the extracted images will be saved in subfolders that will be created under Preprocessing/Outputs.

It should be noted that the final images should be checked before using as cropping coordinates are likely to need manual adjustment for around 5% of the videos, then the frames will need to be re-extracted by running ExtractFrames again. It should also be noted that while these preprocessing scripts should work for Embryoscope and EmbryoScopeplus timelapse videos, some adjustment is likely to be necessary if using videos from different timelapse systems.