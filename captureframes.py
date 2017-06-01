import os
import web
import cv2
import pickle
import os.path
import numpy as np
import utils.game_manager as gm

urls = (
	'/', 'index'
	)

DIR_PATH = 'D:\\map_data\\'
IMAGE_WIDTH           = 320
IMAGE_HEIGHT          = 240
IMAGE_CHANNELS        = 1 

def save_screenshot(PATH, res):
	PATH = PATH+"\\"
	if not os.path.exists(PATH):
		os.makedirs(PATH)
	cv2.imwrite(PATH+screen_name, res)

def save_pickel(file_path, screen_name, data):
	res = np.reshape(data, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
	screens = dict()
	if os.path.exists(file_path):
		screens = pickle.load( open( file_path, "rb" ) )	
		screens[screen_name] = res
	with open(file_path, 'wb') as pfile:
		pickle.dump(screens, pfile, protocol=pickle.HIGHEST_PROTOCOL)

class index:
	def POST(self):
		screenshot = gm.grab_screen(region=(100, 40 , 1910, 1050))
		res = gm.preprocess(screenshot)
		data = str(web.data())
		data = data.split("&")
		screen_part = data[0]
		action_part = data[1]
		game_part = data[2]
		screen_name = screen_part.split("=")[1]
		action = action_part.split("=")[1]
		game_id =  game_part.split("=")[1]
		PATH = DIR_PATH + game_id
		save_pickel(PATH, screen_name, res)
		#save_screenshot(PATH, res)
		return True

if __name__ == "__main__":
	app = web.application(urls, globals())
	app.run()