import os
import time
import pyautogui
#import pyscreenshot as ImageGrab

def get_screen_section():
	print "place your mouse cursor"
	loc = None
	try:
		count = 0
		while count < 5:
			x1, y1 = pyautogui.position()
			time.sleep(1)
			x2, y2 = pyautogui.position()
			if x1-x2 == 0 and y2-y1 == 0:
				count +=1
			else:
				count = 0
		loc = [x1, y1]
	except KeyboardInterrupt:
		print "\n"
	return loc

def main():
	path = "/Users/naveenmysore/Documents/QL/FBRl/data/negative/training/"
	x = 118#222#0
	y = 536#121
	x1 =258#548
	y1 = 572#788
	# part of the screen
	x, y = get_screen_section()
	print x, y
	x1, y1 = get_screen_section()
	print x1, y1
	width = x1 - x
	height = y1 - y
	##print x, y
	screenshot=ImageGrab.grab(bbox=(x,y,width,height))
	path, dirs, files = os.walk(path).next()
	#file_id  = len(files)
	#file_name = "fail_"+str(file_id)+".png"
	#file_name = path+file_name
	screenshot.save("fail_frame.png")

if __name__ == "__main__":
	main()
