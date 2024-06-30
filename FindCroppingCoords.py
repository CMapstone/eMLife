import numpy as np
from PIL import Image
import cv2 
import os
import csv
import math

inputfolder='Preprocessing/Input videos'
directory = os.fsencode(inputfolder)

#Create array to store coordinates in for each video
iterationdetails=[]
iterationdetails.append(['Video name','Yend','Ystart','Xend','Xstart'])

#Loop through each video in Input videos folder
for file in os.listdir(directory):
    #extract the 100th frame from the video to use to get cropping co-ordinates. 
    filename = os.fsdecode(file)  
    cap = cv2.VideoCapture(inputfolder+"/"+filename)
    cap.set(1,100);
    ret, frame = cap.read() 
    np_im = np. array(frame)
    img=Image.fromarray(np_im)
    img.save('Preprocessing/temp/temp.png') 

    #Set parameters for circle finding algorithm based on size of frame
    if len(np_im)>500:
        a=640
        b=80
        c=48
        d=160
        e=240
        r=240
        edge=800     
    else:
        a=400
        b=50
        c=30
        d=100
        e=150
        r=150
        edge=500

    #Find the embryo position
    img = cv.imread('Preprocessing/temp/temp.png',0) 
    img = cv.medianBlur(img,5)
    cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,a,
                            param1=b,param2=c,minRadius=d,maxRadius=e)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
        cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    #if above was successful write co-ordinates for cropping to iterationsdetails array
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        X = circles

        #corrections for if circle goes outside of frame
        Xstart=X[0][0]-r
        Xend=X[0][0]+r
        Ystart=X[0][1]-r
        Yend=X[0][1]+r
        if Xstart<0:
            Xstart=0
            Xend=2*r
        if Ystart<0:
            Ystart=0
            Yend=2*r
        if Xend>edge:
            Xend=edge
            Xstart=edge-(2*r)     
        if Yend>edge:
            Yend=edge
            Ystart=edge-(2*r)
        iterationdetails.append([filename,Yend,Ystart,Xend,Xstart])
    #if unsuccessful write 'failed' to iterationsdetails array
    else:
        iterationdetails.append([filename,'failed','failed','failed','failed'])
        
#Export iterations array to csv
with open('Preprocessing/Outputs/croppingcoords.csv','w' ,newline='') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(iterationdetails)   