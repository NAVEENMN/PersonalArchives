import os
import web
import cv2
import pickle
import os.path
import signal
import sys
import numpy as np
import utils.game_manager as gm

urls = (
	'/', 'index'
	)

DIR_PATH = 'D:\\map_data\\'
IMAGE_WIDTH           = 320
IMAGE_HEIGHT          = 240
IMAGE_CHANNELS        = 1 

'''
global screens
screens = dict()
global pkl
if os.path.exists(DIR_PATH+"game"):
	pkl = open( DIR_PATH+"game", "rb" )
	screens = pickle.load(pkl)
else:
	with open(DIR_PATH+"game", 'wb') as pkl:
		pickle.dump(screens, pkl, protocol=pickle.HIGHEST_PROTOCOL)
'''
def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        pkl.close()
        sys.exit(0)

def save_screenshot(PATH, res):
	PATH = PATH+"\\"
	if not os.path.exists(PATH):
		os.makedirs(PATH)
	cv2.imwrite(PATH+screen_name, res)

def save_pickel(dir_path, file_path, screen_name, data):
	res = np.reshape(data, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
	fcount = len(os.listdir(dir_path+"\\"))-1
	if fcount <0:
		file_path = file_path+"-"+str(0)
	else:
		file_path = file_path+"-"+str(fcount)
	screens = dict()
	try:
		if os.path.exists(file_path):
			with open(file_path, "rb") as pkl:
				screens = pickle.load(pkl)
		with open(file_path, "wb") as pkl:
			screens[screen_name] = res
			pickle.dump(screens, pkl, protocol=pickle.HIGHEST_PROTOCOL)
	except Exception:
		print(dir_path)
		parts = file_path.split("-")
		fcount = fcount + 1
		file_path = parts[0]+"-"+str(fcount)
		with open(file_path, "wb") as pkl:
			screens[screen_name] = res
			pickle.dump(screens, pkl, protocol=pickle.HIGHEST_PROTOCOL)
			
	pkl.close()

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
		save_pickel(DIR_PATH, PATH, screen_name, res)
		#save_screenshot(PATH, res)
		return True

if __name__ == "__main__":
	app = web.application(urls, globals())
	app.run()