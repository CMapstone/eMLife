from nbformat import read
import yaml

from ExtractFrames import extract_frames_main
from GettingPredictions import PrepareDataForModel


def read_config():

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    return config

def run_main(extract_frames, get_predictions, config):

    extract_frames = '2'

    if extract_frames:
        extract_frames_main(config)
    
    if get_predictions:
        PrepareDataForModel()



if __name__=='__main__':

    config = read_config()

    extract_frames = True
    get_predictions= False
    run_main(extract_frames, get_predictions, config)

