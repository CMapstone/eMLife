from nbformat import read
import yaml

from ExtractFrames import extract_frames_main
#from GettingPredictions import PrepareDataForModel
from FindCroppingCoords import find_cropping_coords_main


def read_config():

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    return config

def run_main(extract_frames, get_predictions, find_cropping_coords, config):


    if extract_frames:
        extract_frames_main(config)
    
   # if get_predictions:
    #    PrepareDataForModel()
    
    if find_cropping_coords:
        find_cropping_coords_main(config)





if __name__=='__main__':

    config = read_config()

    find_cropping_coords = False
    extract_frames = True
    get_predictions= False

    run_main(extract_frames, get_predictions, find_cropping_coords, config)

