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


#Network Parameters
INPUT_SHAPE   = 9216
OUTPUT_SHAPE  = 30
EPOCH         = 30
learning_rate = 0.001
FC_LAYER_1_SIZE = INPUT_SHAPE # input to fully connected layer
FC_LAYER_2_SIZE = 512 # fully connected layer
OUT_LAYER_SIZE  = OUTPUT_SHAPE
# tf Graph input

#======== Network
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def create_network():
	x = tf.placeholder("float", [None, INPUT_SHAPE])

	W_fc1 = weight_variable([FC_LAYER_1_SIZE, FC_LAYER_2_SIZE])
	b_fc1 = bias_variable([FC_LAYER_2_SIZE])
	
	W_fc2 = weight_variable([FC_LAYER_2_SIZE, OUT_LAYER_SIZE])
	b_fc2 = bias_variable([OUT_LAYER_SIZE])
	
	h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
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

def testNetwork(sess,x, Y, readout):
	readout_t = readout.eval(feed_dict={x : Y})
	return readout_t

def visulize_a_image(image_array):
	image = np.reshape(image_array, (IMAGE_HEIGHT, IMAGE_WIDTH))
	cv2.imshow("image", image)
	print(image)
	cv2.waitKey()


def construct_image(image_array, facepoints):
	image = np.reshape(image_array, (IMAGE_HEIGHT, IMAGE_WIDTH))
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
	X = X.astype(np.float32) # (2140 x 9216) (samples x pixels)
	T = np.vstack(test_df['Image'].values) / 255.
	T = T.astype(np.float32)
	# visulize_a_image(X[10])
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