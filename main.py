from nbformat import read
import yaml

from ExtractFrames import extract_frames_main
from GettingPredictions import getting_predictions_main
from FindCroppingCoords import find_cropping_coords_main


def read_config():

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    return config


def run_main(extract_frames, get_predictions, find_cropping_coords, config):

    if extract_frames:
        extract_frames_main(config)
    
    if get_predictions:
        getting_predictions_main(config)
    
    if find_cropping_coords:
        find_cropping_coords_main(config)
    
    return


if __name__=='__main__':

    config = read_config()
    # Set the tasks you want to run to 'True' and leave the others as 'False'
    find_cropping_coords = False
    extract_frames = True
    get_predictions= False

    run_main(extract_frames, get_predictions, find_cropping_coords, config)

