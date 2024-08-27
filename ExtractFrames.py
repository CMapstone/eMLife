import pandas as pd
import os
import cv2
from PIL import Image, ImageFile
import tensorflow as tf
keras = tf.keras
import numpy as np
from tensorflow.keras.models import load_model

#Variables to be set
stage='PN' #The developmental stage you are interested in, make sure it matches a column heading in the Annotations csv file
cropping='Y' #'Y' if you want cropped images, 'N' if you do not want images to be cropped
offset=[0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10] # The list of required offsets


#load the model we have trained to read the timestamp. It should work for EmbryoScope and Embryoscope plus videos, another model may need to be trained for videos from other timelapse systems
model = load_model("Preprocessing/dig2.h5")
#this function reads the timestamp. It should work for EmbryoScope and Embryoscope plus videos, it will need to be adjusted for videos from other timelapse systems
def gettime(frame,model):
    imgsize=28
    
    if len(frame)<601:
        
        digit1=frame[478:487,449:456] 
        digit1 = Image.fromarray(digit1)
        digit1=digit1.resize((imgsize, imgsize, ))
        digit1= digit1.convert('L')
        digit1 = np. array(digit1)
        if digit1[0,0]>100:
            digit1 = digit1.reshape(1, imgsize, imgsize, 1)
            digit1 =digit1/ 255
            prediction = model.predict(digit1)
            Dig1=prediction.argmax()
        else:
            Dig1=0.0

        digit2=frame[478:487,456:463]
        digit2 = Image.fromarray(digit2)
        digit2=digit2.resize((imgsize, imgsize, ))
        digit2= digit2.convert('L')
        digit2 = np. array(digit2) 
        if digit2[0,0]>100:
            digit2 = digit2.reshape(1, imgsize, imgsize, 1)
            digit2 =digit2/ 255
            prediction = model.predict(digit2)
            Dig2=prediction.argmax()
        else:
            Dig2=0.0

        digit3=frame[478:487,463:470]
        digit3 = Image.fromarray(digit3)
        digit3=digit3.resize((imgsize, imgsize, ))
        digit3= digit3.convert('L')
        digit3 = np. array(digit3) 
        digit3 = digit3.reshape(1, imgsize, imgsize, 1)
        digit3 =digit3/ 255
        prediction = model.predict(digit3)
        Dig3=prediction.argmax()

        digit4=frame[478:487,473:480]
        digit4 = Image.fromarray(digit4)
        digit4=digit4.resize((imgsize, imgsize, ))
        digit4= digit4.convert('L')
        digit4 = np. array(digit4) 
        digit4 = digit4.reshape(1, imgsize, imgsize, 1)
        digit4 =digit4/ 255
        prediction = model.predict(digit4)
        Dig4=prediction.argmax()


        time=Dig1*100+Dig2*10+Dig3+Dig4*0.1
       
        
    if len(frame)>601:
                
        digit1=frame[772:786,726:737] 
        digit1 = Image.fromarray(digit1)
        digit1=digit1.resize((imgsize, imgsize, ))
        digit1= digit1.convert('L')
        digit1 = np. array(digit1)
        if digit1[0,0]>100:
            digit1 = digit1.reshape(1, imgsize, imgsize, 1)
            digit1 =digit1/ 255
            prediction = model.predict(digit1)
            Dig1=prediction.argmax()
        else:
            Dig1=0.0

        digit2=frame[772:786,737:748]
        digit2 = Image.fromarray(digit2)
        digit2=digit2.resize((imgsize, imgsize, ))
        digit2= digit2.convert('L')
        digit2 = np. array(digit2) 
        if digit2[0,0]>100:
            digit2 = digit2.reshape(1, imgsize, imgsize, 1)
            digit2 =digit2/ 255
            prediction = model.predict(digit2)
            Dig2=prediction.argmax()
        else:
            Dig2=0.0

        digit3=frame[772:786,748:759]
        digit3 = Image.fromarray(digit3)
        digit3=digit3.resize((imgsize, imgsize, ))
        digit3= digit3.convert('L')
        digit3 = np. array(digit3) 
        digit3 = digit3.reshape(1, imgsize, imgsize, 1)
        digit3 =digit3/ 255
        prediction = model.predict(digit3)
        Dig3=prediction.argmax()

        digit4=frame[772:786,764:775]
        digit4 = Image.fromarray(digit4)
        digit4=digit4.resize((imgsize, imgsize, ))
        digit4= digit4.convert('L')
        digit4 = np. array(digit4) 
        digit4 = digit4.reshape(1, imgsize, imgsize, 1)
        digit4 =digit4/ 255
        prediction = model.predict(digit4)
        Dig4=prediction.argmax()


        time=Dig1*100+Dig2*10+Dig3+Dig4*0.1
       
    return time

AllNotes= 'Preprocessing/Annotations.csv' #Link to the csv file with video names,key developmental timepoints and co-ordinates for cropping (optional), see template
inputfolder= 'Preprocessing/Input videos' #folder of input videos

#make output folders if they do not already exist
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
    cell_value_id = sh[stage][i] 
    d[cell_value_class]=cell_value_id
    if cropping=='Y':
        Xs = sh['Xstart'][i] 
        Xe = sh['Xend'][i] 
        Ys = sh['Ystart'][i] 
        Ye = sh['Yend'][i] 
        Xstart[cell_value_class] = Xs
        Xend[cell_value_class]=Xe
        Ystart[cell_value_class]=Ys
        Yend[cell_value_class]=Ye


def get_key_time(frameNum, filename):
    """
    Get the time of the key moment frame
    """
    cap = cv2.VideoCapture(inputfolder+"/"+filename)
    cap.set(1,int(frameNum)-1); 
    ret, frame = cap.read() 
    time_of_frame = gettime(frame, model)
    keyeventtime = time_of_frame
    return keyeventtime

def process_frame(filename, frameNum):

    keyeventtime = get_key_time(frameNum, filename)

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

         

        
    

def iter_input_folder():
    """
    Loop through every video in the input folder extracting all the output frames for this stage
    """
    for file in os.listdir(os.fsencode(inputfolder)):
        # Read in file
        filename = os.fsdecode(file)
        frameNum= d[filename]   # Need to make sure column 1 matches the actual name of the video file
        if int(frameNum)>0:   # Check frame number is non zero as when the key timepoint could not be determined a zero is entered into the csv
            process_frame(filename, frameNum)

def extract_frames_main():
    gettime()
