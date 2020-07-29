import glob
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from os import system
import os.path
import multiprocessing
from scipy import spatial
import sys
import pyscreenshot as ImageGrab
import time

roll = 10
path = os.getcwd()
x_loc = 0
y_loc = 121
width = 548-x_loc
height = 761-y_loc

f_x_loc = 118
f_y_loc = 536
f_width = 258 - f_x_loc
f_height= 572 - f_y_loc

def reset():
    tf.reset_default_graph()

def press_space():
	space()
        #p = multiprocessing.Process(target=space)
        #p.start()

def space():
        keys = "space"
        system('osascript -e \'tell application "System Events" to keystroke ' + keys + "'")
        #time.sleep(0.12)
        #system('osascript -e \'tell application "System Events" to keystroke ' + keys + "'")

def get_frame():
	resiz = 0.1
	os.system('screencapture screen.png')
	screen = Image.open("screen.png").convert('L')
	gamesec = screen.crop((1, 240, 1100, 1520))
	failsec = screen.crop((235, 1070, 520, 1150))
	gamesec = gamesec.resize( [int(resiz * s) for s in gamesec.size] )
	gamesec.save("frame.png")
	failsec.save("cfail_frame.png")
	arr_gamesec  = np.asarray(gamesec)
	pixel_count = gamesec.size[0] * gamesec.size[1]
	img = arr_gamesec.reshape(1, pixel_count)
	img = np.append([0.0, 0.0], img)
	arr_gamesec = np.ndarray((1, pixel_count), buffer=np.array(img), offset=np.float_().itemsize, dtype=float)
	arr_failsec = np.asarray(failsec)
	pixel_count = failsec.size[0] * failsec.size[1]
	img = arr_failsec.reshape(1, pixel_count)
	img = np.append([0.0, 0.0], img)
	arr_failsec = np.ndarray((1, pixel_count), buffer=np.array(img), offset=np.float_().itemsize, dtype=float)
	return arr_gamesec, arr_failsec, gamesec

def load_frames():
	resiz = 0.1
        fail_trigger, fn = load_image("fail_frame.png")
	os.system('screencapture screen.png')
	screen = Image.open("screen.png").convert('L')
	#screen.crop((1, 240, 1100, 1520)).save("fra.png")
	#screen.crop((235, 1045, w-2038, h-475)).save("fra.png")
	screen.crop((x_loc, 140, width, height)).save("fra.png")
        screenshot=ImageGrab.grab(bbox=(x_loc,y_loc,width,height))#game frame
        fail_frame = ImageGrab.grab(bbox=(f_x_loc,f_y_loc,f_width,f_height))#game frame
        #screenshot = 
	screenshot = screenshot.resize( [int(half * s) for s in screenshot.size] )
	print screenshot.size
	screenshot.save("frame.png")
        fail_frame.save("cfail_frame.png")
        in_data, n = load_image("frame.png")
        check_fail, cn = load_image("cfail_frame.png")
        return in_data, screenshot, cn, fn, n

def save_an_episode(screenshot, mode):
	if not (os.path.isdir("data/rollouts/positive/")):
		os.system("mkdir data/rollouts/positive/")
	if not (os.path.isdir("data/rollouts/current/")):
		os.system("mkdir data/rollouts/current/")
	if not (os.path.isdir("data/rollouts/negative/")):
		os.system("mkdir data/rollouts/negative/")

        if mode == "positive" or mode == "negative":
                if mode == "positive":
                        path = "data/rollouts/positive/"
                else:
                        path = "data/rollouts/negative/"
                try:
                        path1, dirs, files = os.walk(path).next()
                        eps_id = len(dirs)
                except:
                        eps_id = 0
                os.system("mkdir "+path+"episode"+str(eps_id))
                cmd = "sudo mv data/rollouts/current/* "+path+"episode"+str(eps_id)
                os.system(cmd)
        if mode == "current":
                path = "data/rollouts/current/"
                try:
                        path1, dirs, files = os.walk(path).next()
                        frame_id = len(files)
			file_name = path+"frame_"+str(frame_id)+".png"
			screenshot.save(file_name)
                except:
			e = sys.exc_info()[0]
			print e
                        frame_id = 0
			file_name = path+"frame_"+str(frame_id)+".png"
			screenshot.save(file_name)
	return
def play():
        prob = random.uniform(0, 1)
        if prob > 0.5:
                press_space()
        return prob

def new_game():
	press_space()
	#press_space()

def clean_rollouts():
        os.system("sudo rm -rf data/rollouts/current/*")
        os.system("sudo rm -rf data/rollouts/negative/*")
        os.system("sudo rm -rf data/rollouts/positive/*")

def load_image(path):
        img  = np.asarray(Image.open(path).convert('L'))
        pixel_count = img.shape[0] * img.shape[1]
        img = img.reshape(1, pixel_count)
        img = np.append([0.0, 0.0], img)
        img1 = np.ndarray((1, pixel_count), buffer=np.array(img), offset=np.float_().itemsize, dtype=float)
        img = tf.cast(img1, tf.float32)
        return img, img1

def load_batches():
	nfolder = glob.glob("data/rollouts/negative/*")
	pfolder = glob.glob("data/rollouts/positive/*")
        nepisodes = len(nfolder)
	pepisodes = len(pfolder)
	if nepisodes == 0 and pepisodes == 0:
		return [], []
        IMGS = []
        total_samples = 0
        for x in range(0, nepisodes):
                path = "data/rollouts/negative/episode"+str(x)+"/*.png"
                imgs = list()
                for filename in glob.glob(path):
                        c, img = load_image(filename)
                        imgs.append(img)
                for i,k in zip(imgs[0::2], imgs[1::2]):
                        diff = np.subtract(i, k)
                        if len(IMGS) == 0:
                                IMGS = diff
                        else:
                                IMGS = np.append(IMGS, diff, axis=0)
	total_nsamples = IMGS.size[0]
	print total_nsamples
        for x in range(0, pepisodes):
                path = "data/rollouts/positive/episode"+str(x)+"/*.png"
                imgs = list()
                for filename in glob.glob(path):
                        c, img = load_image(filename)
                        imgs.append(img)
                for i,k in zip(imgs[0::2], imgs[1::2]):
                        diff = np.subtract(i, k)
                        if len(IMGS) == 0:
                                IMGS = diff
                        else:
                                IMGS = np.append(IMGS, diff, axis=0)

        total_psamples = IMGS.shape[0]-total_nsamples
        #IMGS = tf.cast(IMGS, tf.float32)
        true_class = list()
        for x in range(0, total_nsamples):
                	true_class.append([-1.0])
	for x in range(0, total_psamples):
			true_class.append([1.0])
	true_class = np.array(true_class)
	#true_class = np.transpose(true_class)
	print IMGS.shape, true_class.shape
        return IMGS, true_class

