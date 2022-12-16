eMLife

The eMLife pipeline produced models that predict the likelihood of an embryo resulting in live birth after IVF transfer. Models have been trained for various stages identified as being key for live birth prediction; 1 hour before NEBD, first frame with 2 cells, 14 hours after first frame with 4 cells, 21 hours after initiation of 8-16 cell round, and last frame of the video (blastocyst stage).

Requirements

-Python version 3.10.9

-Tensorflow version 2.11.0

Instructions
1. Put embryo images (extracted from a timelapse video) to be analysed in the appropriate folders under eMLife/stages (stages described above). Or, if just testing the models, use the demo images already in each folder.
2. Run eMLife.py
3. Result files for each stage can now be found in the Results folder. A number between 0 and 1 will be assigned to each image, 1 corresponds to live birth prediction, 0 corresponds to no pregnancy prediction.
