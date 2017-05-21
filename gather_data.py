import time
import cv2
import os
import shutil
import numpy as np
import datetime
import utils.game_manager as gm

IMAGE_WIDTH           = 320
IMAGE_HEIGHT          = 240
IMAGE_CHANNEL         = 1 
 
PATH = 'D:\\map_data\\'

def get_a_frame(file_count):
	action = str("0000")
	screen = gm.grab_screen(region=(0, 40 , 1910, 1050))
	screen = cv2.resize(screen, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
	screen = cv2.cvtColor( screen, cv2.COLOR_RGB2GRAY )
	r_a = gm.get_pressed_key()
	#print(r_a)
	if r_a[0] == 1:
		action=str("1000")
	if r_a[1] == 1:
		action=str("0100")
	if r_a[2] == 1:
		action=str("0010")
	if r_a[3] == 1:
		action=str("0001")
	timestr = time.strftime("%Y%m%d-%H%M%S")
	file_name = str(timestr)+'_'+action
	cv2.imwrite(PATH+file_name+'.png',screen)
	if np.random.rand() < 0.5:
		screen = cv2.flip(screen, 1)
		if r_a[0] == 1:
			action=str("1000")
		if r_a[1] == 1:
			action=str("0100")
		if r_a[2] == 1:
			action=str("0001")
		if r_a[3] == 1:
			action=str("0010")
		file_count = file_count + 1
		timestr = time.strftime("%Y%m%d-%H%M%S")
		file_name = str(timestr)+'_'+action
		cv2.imwrite(PATH+file_name+'.png',screen)
	return screen, file_count

def main():
	#shutil.rmtree(PATH)
	#os.makedirs(PATH)
	time.sleep(5)
	file_count = 0
	try:
		path, dirs, files = os.walk(PATH).next()
		file_count = len(files)
	except Exception:
		file_count = 0

	endTime = datetime.datetime.now() + datetime.timedelta(minutes=10)
	while True:
		if datetime.datetime.now() >= endTime:
			break
		screen, file_count = get_a_frame(file_count+1)

if __name__ == "__main__":
	main()