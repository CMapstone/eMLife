import numpy as np
from PIL import Image
import cv2 
import os
import csv
import math

# TODO check it still works
# TODO add typehinting, e.g: def iter_videos(directory: str) -> str
# TODO add spaces on equals and commas
# TODO change all variables and function names to snake case

def export_coords(circles,iterationdetails,circle_params,filename):
    """
    #TODO add docstring
    """
    circles = np.round(circles[0, :]).astype("int")
    X = circles
    #corrections for if circle goes outside of frame
    Xstart=X[0][0]-circle_params[r]
    Xend=X[0][0]+circle_params[r]
    Ystart=X[0][1]-circle_params[r]
    Yend=X[0][1]+circle_params[r]
    if Xstart<0:
        Xstart=0
        Xend=2*circle_params[r]
    if Ystart<0:
        Ystart=0
        Yend=2*circle_params[r]
    if Xend>circle_params[edge]:
        Xend=circle_params[edge]
        Xstart=circle_params[edge]-(2*circle_params[r])     
    if Yend>circle_params[edge]:
        Yend=circle_params[edge]
        Ystart=circle_params[edge]-(2*circle_params[r])
    iterationdetails.append([filename,Yend,Ystart,Xend,Xstart])

def draw_circles(circle_params,img_path):
    """
    Finds circles that describe the position of a circular object in an image
    """
    # Find the embryo position
    img = cv2.imread(img_path,0) 
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,circle_params[a],
                            param1=circle_params[b],param2=circle_params[c],minRadius=circle_params[d],maxRadius=circle_params[e])
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # Draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # Draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        
    return circles


def extract_100th_frame(file,cropping_input_folder):
    """
    #TODO add docstring
    """
    filename = os.fsdecode(file)  
    cap = cv2.VideoCapture(cropping_input_folder + "/" + filename)
    cap.set(1,100)
    ret, frame = cap.read() 
    np_im = np.array(frame)
    img = Image.fromarray(np_im)
        #TODO save to memory instead of disk
    temp_path='Preprocessing/temp/temp.png'
    img.save(temp_path) 

    return filename, np_im,temp_path

def get_circle_params(np_im):
    """
    Parameters for cropping algorithm are set depending on whether video is EmbryoScope (500x500) or EmbryoScopePlus (800x800)
    a-e are cv2.HoughCircles parameters, r is half the size of the desired output cropped image, based on observed embryo size,
    edge is size of frame. These parameters will need changing if you are using a different timelapse system.
    """   
    if len(np_im)>500:
        circle_params = {
            'a': 640,
            'b': 80,
            'c': 48,
            'd': 160,
            'e': 240,
            'r': 240,
            'edge': len(np_im)
        }
    else:
        circle_params = {
            'a': 400,
            'b': 50,
            'c': 30,
            'd': 100,
            'e': 150,
            'r': 150,
            'edge': len(np_im)
        }
    return circle_params

def array_to_csv(csvpath: str, array_name: list):
    """
    #TODO add docstring
    """
    with open(csvpath,'w' ,newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerows(array_name)   
    return

def iter_videos(iterationdetails: list, cropping_input_folder: str) -> list:
    """
    #TODO add docstring
    """

    directory = os.fsencode(cropping_input_folder)
    # Loop through each video in Input videos folder
    for file in os.listdir(directory):
        
        filename, np_im,temp_path = extract_100th_frame(file,cropping_input_folder)

        # Get parameters for embryo locating tasks depending on size of frame
        circle_params=get_circle_params(np_im)

        #find embryo location using draw_circles function    
        circles = draw_circles(circle_params, temp_path)

        # If above was successful write co-ordinates for cropping to iterationsdetails array
        if circles is not None:
            export_coords(circles,iterationdetails,circle_params,filename)
        # If unsuccessful write 'failed' to iterationsdetails array
        else:
            iterationdetails.append([filename, 'failed', 'failed', 'failed', 'failed'])
        
    return iterationdetails

def main(config):
    """
    #TODO add docstring
    """
    # Set directory as input folder in config file
    cropping_input_folder = config['find_cropping_coords']['cropping_input_folder']

    # Create array to store coordinates in for each video
    # List of lists to allow for more lists to be appended
    iterationdetails = [['Video name', 'Yend', 'Ystart', 'Xend', 'Xstart']]

    # Loop through each video in Input videos folder
    iterationdetails = iter_videos(iterationdetails, cropping_input_folder)
            
    #Export iterations array to csv
    array_to_csv('Preprocessing/Outputs/croppingcoords.csv', iterationdetails)
    
    return