
import argparse
import os
import numpy as np
import utils.game_manager as gm
import cv2
from firebase import firebase
firebase = firebase.FirebaseApplication('https://sdrone-9ae10.firebaseio.com/', None)

#load our saved model
from keras.models import load_model

#helper class
import utils
model = None
prev_image_array = None
IMAGE_WIDTH           = 320
IMAGE_HEIGHT          = 240
IMAGE_CHANNELS         = 1  
 

def fly_drone(args):
    #load model
    model = load_model(args.model)
    for x in range(0, 5):
    	screen = gm.grab_screen(region=(0, 40 , 1920, 1080))
    	screen = cv2.resize(screen, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    	screen = cv2.cvtColor( screen, cv2.COLOR_RGB2GRAY )
    	screen = np.reshape(screen, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)) 
    	print(screen.shape)
    	action = model.predict(screen, batch_size=1)
    	print(action[0])
    	take_action = [0, 0, 0, 0]
    	ind = np.argmax(action[0])
    	take_action[ind] = 1
    	#gm.Actions(take_action)
    	new_user = 'Ozgur Vatansever'
    	data = dict()
    	data['ra'] = 1;
    	result = firebase.put('/actions', "action", data)
    	print(result)

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
    fly_drone(args)
   