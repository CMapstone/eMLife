import pandas as pd
import os
import cv2
from PIL import Image, ImageFile
import tensorflow as tf
keras = tf.keras
import numpy as np
from tensorflow.keras.models import load_model


def create_output_subfolders(config_dict):
    """
    Check whether subfolders for the output images exist, if not then create the subfolders.
    """
    if not os.path.exists(config_dict['output_folder']+config_dict['stage']):
        os.makedirs(config_dict['output_folder']+config_dict['stage'])
        for i in config_dict['offset']:
            os.makedirs(config_dict['output_folder']+config_dict['stage']+'/'+str(i))
    
    return


def read_in_annotations(config_dict):
    """
    Read frame numbers and cropping coordinates into dictionaries from annotations csv file.
    """
    # Create a dictionary of dictionaries for reference frame number, x start, y start, x end, 
    # y end. 
    frames_dict = {
    'ref_frame' : {},
    'x_start' : {},
    'x_end' : {},
    'y_start' : {},
    'y_end' : {},
    }
    
    # Read in annotations csv.
    sh = pd.read_csv(config_dict['annotations'])
    # Iterate over each embryo, adding reference frame and coordinates to dictionaries with video
    # name as key.
    for i in range(0,len(sh)):
        cell_value_class = sh['video name'][i]
        frames_dict['ref_frame'][cell_value_class] = sh[config_dict['stage']][i] 
        if config_dict['cropping'] == 'Y':
            frames_dict['x_start'][cell_value_class] = sh['Xstart'][i] 
            frames_dict['x_end'][cell_value_class] = sh['Xend'][i] 
            frames_dict['y_start'][cell_value_class] = sh['Ystart'][i] 
            frames_dict['y_end'][cell_value_class] = sh['Yend'][i] 
    
    return frames_dict


def calculate_key_moment_time(input_folder, file_name, frame_num, model):
    """
    Use the get_time function to read the timestamp at the reference key moment.
    """
    cap = cv2.VideoCapture(input_folder+"/"+file_name)
    cap.set(1,int(frame_num)-1); 
    ret, frame = cap.read() 
    time_of_frame = get_time(frame, model)
    key_event_time = time_of_frame

    return key_event_time, time_of_frame, cap


def read_digit(digit_coords, img_size, frame, model):
    """
    Read the digit in a passed patch of an image using a pre-trained CNN.
    This may need modifying if you are using a timelapse system other than the EmbryoScope or 
    EmbryoScope plus.
    """
    digit = frame[digit_coords[0]:digit_coords[1],digit_coords[2]:digit_coords[3]] 
    digit = Image.fromarray(digit)
    digit = digit.resize((img_size, img_size, ))
    digit = digit.convert('L')
    digit = np. array(digit)
    # Only use model to prediction digit value if there is a digit in the patch (determined by 
    # value of corner pixel that will always have a value greater than 100 if there is a digit 
    # present). This step will have to be checked if using a different timelapse system.
    if digit[0,0]>100:
        digit = digit.reshape(1, img_size, img_size, 1)
        digit = digit/ 255
        prediction = model.predict(digit)
        dig=prediction.argmax()
    else:
        dig = 0.0

    return dig


def get_time(frame, model):
    """
    This function reads the timestamp. It should work for EmbryoScope and Embryoscope plus videos,
    it will need to be adjusted for videos from other timelapse systems.
    """
    img_size = 28
    
    # The coordinates of the timestamp vary depending on whether the frame is from an Embryoscope 
    # or EmbryoScope plus video EmbryoScope videos are 500x500 , EmbryoscopePlus videos are 800x800
    # pixels.
    if len(frame) == 500:

        digit_coords = [478, 487, 449, 456]
        digit1 = read_digit(digit_coords, img_size, frame, model)
        digit_coords = [478, 487, 456, 463]
        digit2 = read_digit(digit_coords, img_size, frame, model)
        digit_coords = [478, 487, 463, 470]
        digit3 = read_digit(digit_coords, img_size, frame, model)
        digit_coords = [478, 487, 473, 480]
        digit4 = read_digit(digit_coords, img_size, frame, model)
    
    if len(frame) == 800:

        digit_coords = [772, 786, 726, 737]
        digit1 = read_digit(digit_coords, img_size, frame, model)
        digit_coords = [772, 786, 737, 748]
        digit2 = read_digit(digit_coords, img_size, frame, model)
        digit_coords = [772, 786, 748, 759]
        digit3 = read_digit(digit_coords, img_size, frame, model)
        digit_coords = [772, 786, 764, 775]
        digit4 = read_digit(digit_coords, img_size, frame, model)

    # Combine digits to get the time on the timestamp.
    time = digit1*100+digit2*10+digit3+digit4*0.1

    return time
       
def find_offset_going_forward(time_of_frame, time_wanted, frame_num, cap, model):
    """
    Iterate over each frame, going forward in time, until the frame with the correct timestamp is
    found.
    """
    while time_of_frame< time_wanted:
        frame_num = int(frame_num)+1
        # Extract the frame with number frame_num. 
        cap.set(1, int(frame_num)-1) 
        # Read the frame
        ret, frame = cap.read() 
        time_of_frame = get_time(frame, model)

    # Determine whether this frame or frame before is closest to the desired timepoint.
    cap.set(1, int(frame_num-2)) 
    ret, frame = cap.read() 
    time_other_frame = get_time(frame, model)
    if time_of_frame-time_wanted < time_wanted-time_other_frame:
        cap.set(1, int(frame_num)-1) 
    else:
        cap.set(1, int(frame_num-2))

    return cap, frame_num


def find_offset_going_backward(time_of_frame, time_wanted, frame_num, cap, model):
    """
    Iterate over each frame, going backwards in time, until the frame with the correct timestamp 
    is found.
    """
    while time_of_frame > time_wanted:
        # Extract the frame with number frame_num.
        frame_num = int(frame_num)-1
        cap.set(1, int(frame_num)-1)
        # Read the frame.
        ret, frame = cap.read() 
        time_of_frame = get_time(frame, model)

    # Determine whether this frame or frame after is closest to the desired timepoint.
    cap.set(1, int(frame_num)) 
    ret, frame = cap.read() 
    time_other_frame = get_time(frame, model)
    if time_wanted-time_of_frame < time_other_frame-time_wanted:
        cap.set(1, int(frame_num)-1)
    else:
        cap.set(1, int(frame_num))

    return cap, frame_num


def save_frame(cap, frame_num, config_dict, i, file_name, frames_dict):
    """
    Save the frame as a png to the correct output folder.
    """
    ret, frame = cap.read() 
    np_im = np. array(frame)
    if config_dict['cropping'] == 'Y':
        img = Image.fromarray(np_im[int(frames_dict['y_start'][file_name]):int(frames_dict['y_end'][file_name])
                                    ,int(frames_dict['x_start'][file_name]):int(frames_dict['x_end'][file_name])])
    if config_dict['cropping'] == 'N':
        img = Image.fromarray(np_im)
    img.save(config_dict['output_folder']+config_dict['stage']+'/'+str(i)+"/"+str(file_name)+'_'+str(frame_num)+ '.png') 

    return


def iter_offsets(cap, frame_num, config_dict, key_event_time, time_of_frame, model, file_name, frames_dict):
    """
    Iterate through every desired timepoint offset from the reference timepoint and save an image 
    at each timepoint.
    """
    for i in config_dict['offset']:

        # Calculate the time that should be showing on the timestamp for this offset.
        time_wanted = key_event_time+i  

        if i > 0:
        # Find the first frame after the offset time.
            cap, frame_num = find_offset_going_forward(time_of_frame, time_wanted, frame_num, cap, model)
                
        if i < 0:
            # Find the first frame before the offset time. 
            cap, frame_num = find_offset_going_backward(time_of_frame, time_wanted, frame_num, cap, model)
                    
        # Save frame as image.
        save_frame(cap, frame_num, config_dict, i, file_name, frames_dict)

    return


def iter_embryos(model, config_dict, frames_dict):
    """
    Iterate through every embryo in the dataset to extract frames at all the desired offsets from 
    the reference point.
    """
    for file in os.listdir(os.fsencode(config_dict['input_folder'])):
        file_name = os.fsdecode(file)
        frame_num = frames_dict['ref_frame'][file_name] 

        # Check frame number is non zero as when the key timepoint could not be determined a zero 
        # is entered into the csv.
        if int(frame_num)>0:  
            
            # Get the time of the key moment frame.
            key_event_time, time_of_frame, cap = calculate_key_moment_time(config_dict['input_folder'],
                                                                            file_name, frame_num, model)

            # Iterate through all offsets.
            iter_offsets(cap, frame_num, config_dict, key_event_time, time_of_frame, model, file_name, frames_dict)
            
    return


def extract_frames_main(config):
    """
    Extract frames as png files for all desired offsets from a reference time point for a dataset 
    of embryo timelapse videos.
    """
    # Read in all input data, output location, and user defined variables from config file.
    config_dict = {
    'stage' : config['extract_frames']['stage'],
    'annotations' : config['extract_frames']['annotations'],
    'offset' : config['extract_frames']['offset'],
    'cropping' : config['extract_frames']['cropping'],
    'input_folder' : config['extract_frames']['input_folder'],
    'output_folder' : config['extract_frames']['output_folder']
    }

    # Load the model we have trained to read the timestamp. It should work for EmbryoScope and
    # Embryoscope plus videos, another model may need to be trained for videos from other 
    # timelapse systems.
    model = load_model("Preprocessing/dig2.h5")

    # Extract information from Annotations csv.
    frames_dict = read_in_annotations(config_dict)

    # Create subfolders for the output images.
    create_output_subfolders(config_dict)

    # Loop through every video in the input folder extracting all the required frames for this 
    # stage and save them in the output folder.
    iter_embryos(model, config_dict, frames_dict)
    
    return