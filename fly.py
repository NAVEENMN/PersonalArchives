
import argparse
import os
import numpy as np
import utils.game_manager as gm

#load our saved model
from keras.models import load_model

#helper class
import utils
model = None
prev_image_array = None
IMAGE_WIDTH           = 320
IMAGE_HEIGHT          = 240
IMAGE_CHANNEL         = 1  
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    #load model
    model = load_model(args.model)
    screen = gm.grab_screen(region=(0, 40 , 1920, 1080))
    screen = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    action = float(model.predict(screen, batch_size=1))
    gm.Actions(action)
   