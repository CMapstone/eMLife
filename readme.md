# eMLife

This repository contains code and models accompanying the research article 'Deep learning pipeline reveal key moments in human embryonic development predictive of live birth in IVF'. We provide code for deploying our pre-trained models to predict live birth from embryo timelapse images in [Getting_predictions.py](Getting_predictions.py). We also provide two preprocessing tools to prepare input images; [FindCroppingCoords.py](FindCroppingCoords.py) and [ExtractFrames.py](ExtractFrames.py). These can all be run from the [main.py](main.py) script by setting the required tasks to 'True' (see comment in main.py) and then running main.py. Before running any of these, the inputs, outputs, and other user-defined variables will need to specified in the [config.yaml](config.yaml) file. Instructions for specifying these variables and a description of the tasks performed is given below for each module. 

## Finding coordinates for cropping

FindCroppingCoords.py finds coordinates that can be used to automatically crop frames from the timelapse video based on the position of the embryo in the dish. These coordinates are written to an output csv file. Follow the steps below:

1. Put all videos for which you wish to find cropping coordinates into the input folder under Preprocessing/Input videos, or set input_folder in config.yaml (line 20) to your folder of videos. 
2. Set output_folder in config.yaml (line 22) to the location that you wish the output csv to be saved in.
3. Set find_cropping_coords to equal true in main.py and run main.py. The output csv should appear in your output folder.

## Extracting frames from timelapse video

ExtractFrames.py can be used to extract and automatically crop frames from timelapse videos at various offsets from our reference developmental stages. This can be done following the below steps:

1. Put all videos for which you wish to extract frames for into the input folder under Preprocessing/Input videos, , or set input_folder in config.yaml (line 7) to your folder of videos. 
2. Set output_folder in config.yaml (line 9) to the location that you wish the output images to be saved in. Subfolders will automatically be created when you run the code.
3. Prepare an input csv file using Annotations.csv in the preprocessing folder as a template. This should contain frame numbers of the reference time-points for the developmental stages you want to use (currently this needs to be done manually by inspecting the videos), we reccomend using the reference time-points outlined in our article. If you also wish to crop the images, the cropping coordinates  can be copied from the output file from the previous step (after filling in any coordinates that could not be found by the tool). Set annotations in config.yaml (line 5) to the path for your input csv file.  
4. Set stage in config.yaml (line 12) to the developmental stage of your images. This should match a folder heading in your input csv file.
5. Set cropping in config.yaml (line 14) to 'Y' or 'N' depending on whether you want the images to be cropped. We reccomend cropping for all stages apart from blastocyst.
6. Set offset in config.yaml (line 16) to a list of the time intervals offset from the reference frame that you want to extract images for. See our manuscript for suggested offset values.
7. Set extract_frames to equal true in main.py and run main.py. The output imges should appear in subfolders within your specified output folder.

It should be noted that the final images should be checked before using as model inputs, as cropping coordinates are likely to need manual adjustment for around 5% of the videos. After editing the coordinates in the input csv file, frames in affected time-lapse videos will need to be re-extracted by running ExtractFrames again. It should also be noted that while these preprocessing scripts should work for Embryoscope and EmbryoScopeplus timelapse videos, some adjustment is likely to be necessary if using videos from different timelapse systems.

## Predicting live birth

Getting_predictions.py takes in images of embryos at a given stage and returns an output csv file containing a live birth prediction for each image. Follow the steps below: 

1. Copy the input images you wish to use into the correct subfolder for their developmental stage in the ImagesforPrediction folder. Alternatively, set input_folder in config.yaml (line 26) to your own input directory (which might be the output of the previous step if you used our preprocessing tools), however this should match the structure and subfolder names in ImagesforPrediction. Each image should be a timelapse frame extracted at the pivotal moments in development described in the manuscript. For all stages except blastocyst these images should be cropped to 300x300 centered on the embryo. 
2. Set stage in config.yaml (line 29) to the developmental stage of your images.
3. Set output_folder in config.yaml (line 31) to the location where you would like the output csv to be saved.
4. Set get_predictions to equal true in main.py and run main.py. The output csv that will give a live birth prediction for each input image should appear in your output folder.