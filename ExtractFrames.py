import pandas as pd
import os
import cv2
from PIL import Image, ImageFile
import tensorflow as tf
keras = tf.keras
import numpy as np
from tensorflow.keras.models import load_model

def read_digit(digit_coords, imgsize, frame, model):
    """
    Read the digit in a passed patch of an image using a pre-trained CNN.
    """
    digit = frame[digit_coords[0]:digit_coords[1],digit_coords[2]:digit_coords[3]] 
    digit = Image.fromarray(digit)
    digit = digit.resize((imgsize, imgsize, ))
    digit = digit.convert('L')
    digit = np. array(digit)
    if digit[0,0]>100:
        digit = digit.reshape(1, imgsize, imgsize, 1)
        digit = digit/ 255
        prediction = model.predict(digit)
        Dig=prediction.argmax()
    else:
        Dig = 0.0

    return Dig


def gettime(frame,model):
    """
    This function reads the timestamp. It should work for EmbryoScope and Embryoscope 
    plus videos, it will need to be adjusted for videos from other timelapse systems.
    """
    imgsize=28
    
    # The coordinates of the timestamp vary depending on whether the frame is from an Embryoscope or EmbryoScope plus video
    # EmbryoScope videos are 500x500 , EmbryoscopePlus videos are 800x800 pixels
    if len(frame) == 500:

        digit_coords = [478, 487, 449, 456]
        digit1 = read_digit(digit_coords, imgsize, frame, model)
        digit_coords = [478, 487, 456, 463]
        digit2 = read_digit(digit_coords, imgsize, frame, model)
        digit_coords = [478, 487, 463, 470]
        digit3 = read_digit(digit_coords, imgsize, frame, model)
        digit_coords = [478, 487, 473, 480]
        digit4 = read_digit(digit_coords, imgsize, frame, model)
    
    if len(frame) == 800:

        digit_coords = [772, 786, 726, 737]
        digit1 = read_digit(digit_coords, imgsize, frame, model)
        digit_coords = [772, 786, 737, 748]
        digit2 = read_digit(digit_coords, imgsize, frame, model)
        digit_coords = [772, 786, 748, 759]
        digit3 = read_digit(digit_coords, imgsize, frame, model)
        digit_coords = [772, 786, 764, 775]
        digit4 = read_digit(digit_coords, imgsize, frame, model)

    # Combine digits to get the time on the timestamp
    time = digit1*100+digit2*10+digit3+digit4*0.1

    return time
       

def extract_frames_main(config):

    #load the model we have trained to read the timestamp. It should work for EmbryoScope and Embryoscope plus videos, another model may need to be trained for videos from other timelapse systems
    model = load_model("Preprocessing/dig2.h5")
    #make output folders if they do not already exist
    stage = config['extract_frames']['stage']
    AllNotes = config['extract_frames']['AllNotes']
    offset = config['extract_frames']['offset']
    cropping = config['extract_frames']['cropping']
    inputfolder = config['extract_frames']['inputfolder']
    if not os.path.exists('Preprocessing/Outputs/'+stage):
        os.makedirs('Preprocessing/Outputs/'+stage)
        for i in offset:
            os.makedirs('Preprocessing/Outputs/'+stage+'/'+str(i))

    #extract information from Annotations csv
    d = {}
    Xstart={}
    Xend={}
    Ystart={}
    Yend={}
    sh = pd.read_csv(AllNotes)
    for i in range(0,len(sh)):
        cell_value_class = sh['video name'][i]
        d[cell_value_class]=sh[stage][i] 
        if cropping=='Y':
            Xstart[cell_value_class] = sh['Xstart'][i] 
            Xend[cell_value_class]=sh['Xend'][i] 
            Ystart[cell_value_class]=sh['Ystart'][i] 
            Yend[cell_value_class]=sh['Yend'][i] 

    #loop through every video in the input folder extracting all the output frames for this stage
    for file in os.listdir(os.fsencode(inputfolder)):
        filename = os.fsdecode(file)
        frameNum= d[filename]   #need to make sure column 1 matches the actual name of the video file
        if int(frameNum)>0:   #check frame number is non zero as when the key timepoint could not be determined a zero is entered into the csv
            
            #get the time of the key moment frame
            cap = cv2.VideoCapture(inputfolder+"/"+filename)
            cap.set(1,int(frameNum)-1); 
            ret, frame = cap.read() 
            time_of_frame=gettime(frame, model)
            keyeventtime=time_of_frame

            #loop through all offsets
            for i in offset:
                timewanted=keyeventtime+i  

                if i>0:
                    #find the first frame after the offset time 
                    while time_of_frame< timewanted:
                        frameNum=int(frameNum)+1
                        cap.set(1,int(frameNum)-1); # Where frame_no is the frame you want. having a -1 gives correct frame. change this to get an offset.
                        ret, frame = cap.read() # Read the frame
                        time_of_frame=gettime(frame, model)
                    #determine whether this frame or frame before/after is closest to the timepoint we want
                    cap.set(1,int(frameNum-2)); # Where frame_no is the frame you want. having a -1 gives correct frame. change this to get an offset.
                    ret, frame = cap.read() # Read the frame
                    time_other_frame=gettime(frame, model)
                    if time_of_frame-timewanted<timewanted-time_other_frame:
                        cap.set(1,int(frameNum)-1); # Where frame_no is the frame you want. having a -1 gives correct frame. change this to get an offset.
                    else:
                        cap.set(1,int(frameNum-2)); # Where frame_no is the frame you want. having a -1 gives correct frame. change this to get an offset.
                
                if i<0:
                    #find the first frame before the offset time 
                    while time_of_frame> timewanted:
                        frameNum=int(frameNum)-1
                        cap.set(1,int(frameNum)-1); # Where frame_no is the frame you want. having a -1 gives correct frame. change this to get an offset.
                        ret, frame = cap.read() # Read the frame
                        time_of_frame=gettime(frame, model)
                    #determine whether this frame or frame before/after is closest to the timepoint we want
                    cap.set(1,int(frameNum)); # Where frame_no is the frame you want. having a -1 gives correct frame. change this to get an offset.
                    ret, frame = cap.read() # Read the frame
                    time_other_frame=gettime(frame, model)
                    if timewanted-time_of_frame<time_other_frame-timewanted:
                        cap.set(1,int(frameNum)-1); # Where frame_no is the frame you want. having a -1 gives correct frame. change this to get an offset.
                    else:
                        cap.set(1,int(frameNum)); # Where frame_no is the frame you want. having a -1 gives correct frame. change this to get an offset.
                
                #save frame as image
                ret, frame = cap.read() 
                np_im = np. array(frame)
                if cropping=='Y':
                    img = Image.fromarray(np_im[int(Ystart[filename]):int(Yend[filename]),int(Xstart[filename]):int(Xend[filename])])
                if cropping=='N':
                    img=Image.fromarray(np_im)
                img.save('Preprocessing/Outputs/'+stage+'/'+str(i)+"/"+str(filename)+'_'+str(frameNum)+ '.png') 
