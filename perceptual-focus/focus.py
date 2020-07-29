'''
Perceptual Focus vision
'''

import cv2
import math
import time
import numpy as np
import urllib
from win32api import GetSystemMetrics
from scipy.interpolate import interp1d

global f

class foucs:
	def __init__(self, image):
		self.image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
		self.width = self.image.shape[0]
		self.height = self.image.shape[1]
		self.blank_image = np.zeros((self.width,self.height), np.uint8)
		self.pixels = self.width * self.height
		self.crops = list()

	def window(self, name, image):
		width = GetSystemMetrics(0)
		height = GetSystemMetrics(1)
		scale_width = 640 / self.width
		scale_height = 480 / self.height
		scale = min(scale_width, scale_height)
		window_width = int(self.width * scale)
		window_height = int(self.height * scale)
		cv2.namedWindow(name, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(name, window_width, window_height)
		cv2.setMouseCallback(name, mouse_callback)
		cv2.imshow(name, image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def check_boundry(self, col, row):
		end = True
		print(col, row)
		if col[1] >= self.width:
			col[1] = self.width
		if col[0] <= 0:
			col[0] = 0
		if row[1] >= self.height:
			row[1] = self.height
		if row[0] <= 0:
			row[0] = 0
		if (row[0]+col[0] == 0) and (row[1]*col[1] == self.pixels):
			end = False
			print(end)
		return col, row, end

	def get_normal_dist(self,):
		wd = np.random.normal(20, 5, 1000).astype(int)
		wd = np.sort(wd).tolist()
		dist = [s for s in wd if s >= 2]
		return dist

	def blur(self, px, py):
		window = 2
		end = True
		col = [px-window, px+window]
		row = [py-window, py+window]
		m = interp1d([1,self.width],[1,self.width/16])
		wd = self.get_normal_dist() #window sizes
		while end:
			try:
				k = int(float(m(window)))
			except:
				k = int(self.width/16)
			if k%2==0 :
				k=k+1
			img = cv2.GaussianBlur(self.image,(k,k),k/2)
			crop = img[col[0]:col[1], row[0]:row[1]]
			self.crops.append([crop,[col,row]])
			#if len(wd)>0:
			#	window= wd.pop(0)
			#else:
			window = (window+2)
			print(window)
			col = [px-window, px+window]
			row = [py-window, py+window]
			col, row, end = self.check_boundry(col, row)
		i = 0
		for seg in self.crops[::-1]:
			crop = seg[0]
			col, row = seg[1]
			self.blank_image[col[0]:col[1], row[0]:row[1]] = crop
			cv2.imwrite("images\\"+str(i)+'.png',self.blank_image)
			i =i+1
			cv2.imshow('original', self.blank_image)
			cv2.waitKey(1)
		print("done")	

def mouse_callback(event, x, y, flags, params):
	global f
	if event == 1:
		print(x, y)
		f.blur(x, y)

def main():
	global f
	source = cv2.imread("obama.jpg",0)
	f = foucs(source)
	f.window("original", source)
	f.blur()

if __name__ == "__main__":
	main()