import os
import cv2
import h5py
import numpy as np
from collections import deque

base_path = "D:\workspace\Projects\Flying_v1"
test_image = base_path+"\data\image1.png"
IMAGE_HEIGHT      = 160
IMAGE_WIDTH       = 320
IMAGE_CHANNELS    = 3
IMAGE_SHAPE = [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]

class batch_generator():
	def __init__(self):
		self.skip = 300
		self.batch_size = 1000
		self.data_path = "D:\workspace\dataset\drive\\"
		self.cam_list, self.log_list = [], []
		memory_camera = os.listdir(self.data_path+"camera")
		self.memory_size = len(memory_camera)
		for i in range(0, len(memory_camera)):
			memory = memory_camera[i]
			cam_path = self.data_path + "camera\\" + memory
			log_path = self.data_path + "log\\" + memory
			self.cam_list.append(h5py.File(cam_path))
			self.log_list.append(h5py.File(log_path))
	
	def visulize_a_image(self, cam_feed, log_feed):
		image = self.preprocess(cam_feed)
		cv2.imshow("recap", image)
		print("brake, gas, steering, accel")
		print(log_feed)
		print(" ")
		cv2.waitKey(1)
		
	def visulize_an_episode(self):
		cam = self.cam_list[1]
		log = self.log_list[1]
		for i in range(self.skip * 100, log['times'].shape[0]):
			cam_img = cam['X'][log['cam1_ptr'][i]].swapaxes(0, 2).swapaxes(0, 1)
			break_user = log['brake_user'][i]
			gas_user = log['gas'][i]
			angle_steers = log['steering_angle'][i]
			car_accel = log['car_accel'][i]
			log_feed = [break_user, gas_user, angle_steers, car_accel]
			self.visulize_a_image(cam_img, log_feed)
		
	def preprocess(self, image):
		#image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
		image = np.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
		return image
	def next_batch(self):
		cam_feed_train = np.empty([self.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
		cam_feed_test = np.empty([self.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
		data_feed_train = np.empty([self.batch_size, 4])
		data_feed_test = np.empty([self.batch_size, 4])
		while True:
			random_train_episode = np.random.randint(0, self.memory_size-2)
			test_episode = self.memory_size-1
			cam_train = self.cam_list[random_train_episode]
			log_train = self.log_list[random_train_episode]
			cam_test = self.cam_list[test_episode]
			log_test = self.log_list[test_episode]
			counter = 0
			for index in np.random.permutation(len(cam_train['X'])):
				if counter == self.batch_size:
					break
				cam_data = cam_train['X'][log_train['cam1_ptr'][index]].swapaxes(0, 2).swapaxes(0, 1)
				cam_data = self.preprocess(cam_data)
				break_user = log_train['brake_user'][index]
				gas_user = log_train['gas'][index]
				angle_steers = log_train['steering_angle'][index]
				car_accel = log_train['car_accel'][index]
				cam_feed_train[counter] = cam_data
				data_feed_train[counter] = [break_user, gas_user, angle_steers, car_accel]
				counter +=1
			counter = 0
			for index in np.random.permutation(len(cam_test['X'])):
				if counter == 500:
					break
				cam_data = cam_test['X'][log_test['cam1_ptr'][index]].swapaxes(0, 2).swapaxes(0, 1)
				cam_data = self.preprocess(cam_data)
				break_user = log_test['brake_user'][index]
				gas_user = log_test['gas'][index]
				angle_steers = log_test['steering_angle'][index]
				car_accel = log_test['car_accel'][index]
				cam_feed_test[counter] = cam_data
				data_feed_test[counter] = [break_user, gas_user, angle_steers, car_accel]
				counter +=1
			yield [[cam_feed_train, data_feed_train],[cam_feed_test, data_feed_test]]

def get_state():
	image = np.asarray(cv2.imread(test_image))
	return image

def merge_summary(sess, merged, feed_dict):
	summary =  sess.run(merged, feed_dict)
	return summary
