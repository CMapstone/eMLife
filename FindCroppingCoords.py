import numpy as np
from PIL import Image
import cv2 
import os
import csv
import math

# TODO add typehinting, e.g: def iter_videos(directory: str) -> str
#TODO save to memory instead of disk

def extract_100th_frame(file, cropping_input_folder):
    """
    Extracts the filename and 100th frame in the video as a np array 
    """
    file_name = os.fsdecode(file)  
    cap = cv2.VideoCapture(cropping_input_folder + "/" + file_name)
    cap.set(1, 100)
    ret, frame = cap.read() 
    np_im = np.array(frame)
    img = Image.fromarray(np_im)
    temp_path = 'temp.png'
    img.save(temp_path) 

    return file_name, np_im, temp_path


def get_circle_params(np_im):
    """
    Parameters for cropping algorithm are set depending on whether video is EmbryoScope (500x500) or EmbryoScopePlus (800x800)
    a-e are cv2.HoughCircles parameters, r is half the size of the desired output cropped image based on observed embryo size,
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


def export_coords(circles, iteration_details, circle_params, file_name):
    """
    Extract embryo centre co-ordinates from circles and then use these to work out the coordinates for cropping the embryo
    Corrections are made for if the embryo is close to the edge of the image
    The coordinates are then added to the list of cropping coordintes for each video (iteration_details) 
    """
    circles = np.round(circles[0, :]).astype("int")
    X = circles
    #corrections for if circle goes outside of frame
    x_start = X[0][0]-circle_params['r']
    x_end = X[0][0]+circle_params['r']
    y_start = X[0][1]-circle_params['r']
    y_end = X[0][1]+circle_params['r']
    if x_start < 0:
        x_start = 0
        x_end = 2*circle_params['r']
    if y_start < 0:
        y_start = 0
        y_end = 2*circle_params['r']
    if x_end > circle_params['edge']:
        x_end = circle_params['edge']
        x_start = circle_params['edge']-(2*circle_params['r'])     
    if y_end > circle_params['edge']:
        y_end = circle_params['edge']
        y_start = circle_params['edge']-(2*circle_params['r'])
    iteration_details.append([file_name, y_end, y_start, x_end, x_start])

    return


def draw_circles(circle_params, img_path):
    """
    Finds circles that describe the position of a circular object in an image
    """
    # Find the embryo position
    img = cv2.imread(img_path,0) 
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,circle_params['a'],
                            param1=circle_params['b'], param2=circle_params['c'], minRadius=circle_params['d'], maxRadius=circle_params['e'])
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        
    return circles


def array_to_csv(csv_path: str, array_name: list):
    """
    Write a 2d array to a csv file.
    """
    with open(csv_path, 'w' ,newline = '') as result_file:
        wr = csv.writer(result_file, dialect = 'excel')
        wr.writerows(array_name)   
    return


def iter_videos(iteration_details: list, cropping_input_folder: str) -> list:
    """
    Iterate over every video in the folder, extract co-ordinates for cropping frames centered on the embryo, append these 
    to iteration_details (a list of the cropping coordintes for each video)
    """
    directory = os.fsencode(cropping_input_folder)
    # Loop through each video in Input videos folder
    for file in os.listdir(directory):
        
        file_name, np_im,temp_path = extract_100th_frame(file, cropping_input_folder)

        # Get parameters for embryo locating tasks depending on size of frame
        circle_params = get_circle_params(np_im)

        #find embryo location using draw_circles function    
        circles = draw_circles(circle_params, temp_path)

        # If above was successful write co-ordinates for cropping to iteration_details array
        if circles is not None:
            export_coords(circles, iteration_details, circle_params, file_name)
        # If unsuccessful write 'failed' to iteration_details array
        else:
            iteration_details.append([file_name, 'failed', 'failed', 'failed', 'failed'])
        
    return iteration_details


def find_cropping_coords_main(config):
    """
    Finds co-ordinates for cropping images centered on the embryo for every video in the input folder
    These co-ordinates are exported to an output CSV that can be used (after manual checking step) by ExtractFrames.py
    """
    # Set directory as input folder in config file
    cropping_input_folder = config['find_cropping_coords']['cropping_input_folder']

    # Create array to store coordinates in for each video
    # List of lists to allow for more lists to be appended
    iteration_details = [['Video name', 'Yend', 'Ystart', 'Xend', 'Xstart']]

    # Loop through each video in Input videos folder
    iteration_details = iter_videos(iteration_details, cropping_input_folder)
            
    #Export iteration_details array to csv
    array_to_csv('Preprocessing/Outputs/croppingcoords.csv', iteration_details)
    
    return