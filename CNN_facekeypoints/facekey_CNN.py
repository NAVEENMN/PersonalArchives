#!/usr/bin/env python
from __future__ import print_function

import cv2
import random
import os.path
import numpy as np
import PIL.ImageOps
import tensorflow as tf
from collections import deque
from sklearn.utils import shuffle
from PIL import Image, ImageChops
from pandas.io.parsers import read_csv

TRAIN_DATA_PATH = "training\\training.csv"
TEST_DATA_PATH  = "test\\test.csv"
IMAGE_WIDTH     = 96
IMAGE_HEIGHT    = 96
IMAGE_CHANNEL   = 1

#Network Parameters
INPUT_SHAPE   = 9216
OUTPUT_SHAPE  = 30
EPOCH         = 3000
learning_rate = 0.001
#======== Network Hyper Parameter
PADDING               = "SAME"
INPUT_SIZE_WIDTH      = 640
INPUT_SIZE_HEIGHT     = 160
LAYER_1_FILTER_SIZE   = 8 # 8x8
LAYER_1_FILTER_COUNT  = 32
LAYER_1_FRAMES_COUNT  = 2140
LAYER_1_FILTER_STRIDE = 4
LAYER_2_FILTER_SIZE   = 4 # 4x4
LAYER_2_FILTER_COUNT  = 64
LAYER_2_FRAMES_COUNT  = 32
LAYER_2_FILTER_STRIDE = 2
LAYER_3_FILTER_SIZE   = 3 # 3x3
LAYER_3_FILTER_COUNT  = 64
LAYER_3_FRAMES_COUNT  = 64
LAYER_3_FILTER_STRIDE = 1
FC_LAYER_1_SIZE = 256 # input to fully connected layer
FC_LAYER_2_SIZE = 256 # fully connected layer
OUT_LAYER_SIZE  = OUTPUT_SHAPE
# tf Graph input

#======== Network
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def create_network():
	x = tf.placeholder("float", [None, IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_CHANNEL])
	W_conv1 = weight_variable([LAYER_1_FILTER_SIZE, LAYER_1_FILTER_SIZE, IMAGE_CHANNEL, LAYER_1_FILTER_COUNT])
	b_conv1 = bias_variable([LAYER_1_FILTER_COUNT])
	W_conv2 = weight_variable([LAYER_2_FILTER_SIZE, LAYER_2_FILTER_SIZE, LAYER_1_FILTER_COUNT, LAYER_2_FILTER_COUNT])
	b_conv2 = bias_variable([LAYER_2_FILTER_COUNT])
	W_conv3 = weight_variable([LAYER_3_FILTER_SIZE, LAYER_3_FILTER_SIZE, LAYER_2_FILTER_COUNT, LAYER_3_FILTER_COUNT])
	b_conv3 = bias_variable([LAYER_3_FILTER_COUNT])

	W_fc1 = weight_variable([FC_LAYER_1_SIZE, FC_LAYER_2_SIZE])
	b_fc1 = bias_variable([FC_LAYER_2_SIZE])
	
	W_fc2 = weight_variable([FC_LAYER_2_SIZE, OUT_LAYER_SIZE])
	b_fc2 = bias_variable([OUT_LAYER_SIZE])
	# hidden layers
	h_conv1 = tf.nn.relu(conv2d(x, W_conv1, 4) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
	h_pool3 = max_pool_2x2(h_conv3)
	h_conv3_flat = tf.reshape(h_pool3, [-1, 256])
	h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
	readout = tf.matmul(h_fc1, W_fc2) + b_fc2
	return x, readout

def trainNetwork(sess, x, readout, X, Y):
	y = tf.placeholder("float", [None, OUTPUT_SHAPE])
	cost = tf.reduce_mean(tf.square(readout - y))
	train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
	sess.run(tf.global_variables_initializer())
	for ep in range(0, EPOCH):
		_, loss = sess.run([train_step, cost], feed_dict={x: X, y:Y})
		print(ep, loss)

def testNetwork(sess,x, T, readout):
	readout_t = readout.eval(feed_dict={x : T})
	return readout_t

def visulize_a_image(image):
	#image = np.reshape(image_array, (IMAGE_HEIGHT, IMAGE_WIDTH))
	cv2.imshow("image", image)
	print(image)
	cv2.waitKey()


def construct_image(image, facepoints):
	xp, yp = facepoints[0::2] * 48 + 48, facepoints[1::2] * 48 + 48
	for pixel in range(0, len(xp)):
		cv2.circle(image,(int(xp[pixel]),int(yp[pixel])), 1, (0,0,255), -1)
	cv2.imshow("image", image)
	cv2.waitKey()
 
def load_data():
	train_df = read_csv(os.path.expanduser(TRAIN_DATA_PATH))  # load pandas dataframe
	test_df = read_csv(os.path.expanduser(TEST_DATA_PATH))  # load pandas dataframe
	train_df['Image'] = train_df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
	test_df['Image'] = test_df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
	train_df = train_df.dropna()  # drop all rows that have missing values in them
	test_df = test_df.dropna()
	X = np.vstack(train_df['Image'].values) / 255.
	X = np.reshape(X, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
	X = X.astype(np.float32) # (2140 x 9216) (samples x pixels)
	T = np.vstack(test_df['Image'].values) / 255.
	print(T.shape)
	T = np.reshape(T, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
	T = T.astype(np.float32)
	#visulize_a_image(X[0])
	y = train_df[train_df.columns[:-1]].values
	# to train the CNN scale y in range -1 to 1
	y = (y - 48) / 48  # scale target coordinates to [-1, 1]
	y = y.astype(np.float32)
	X, y = shuffle(X, y, random_state=42)  # shuffle train data
	return X, y, T

def main():
	sess = tf.InteractiveSession()
	X, y, T = load_data()
	x, readout = create_network()
	print(X.shape)
	print(y.shape)
	trainNetwork(sess, x, readout, X, y)
	facepoints = testNetwork(sess, x, T, readout)
	construct_image(T[0], facepoints[0])


if __name__ == "__main__":
	main()