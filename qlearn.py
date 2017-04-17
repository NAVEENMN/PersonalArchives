#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import cv2
import time
import signal
import random
import numpy as np
import PIL.ImageOps
import cPickle as pickle
import tensorflow as tf
from collections import deque
from PIL import Image, ImageChops

#======== Handle Interrept
def sigint_handler(signum, frame):
    print("Exiting..")
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler)

#======== Game Settings
save_frames           = True
create_fail_reference = False
GAME                  = 'DINO' # the name of the game being played for log files
ACTIONS               = 2 # number of valid actions
GAMMA                 = 0.99 # decay rate of past observations
OBSERVE               = 100. # timesteps to observe before training
REPLAY_MEMORY         = 5000 # number of previous transitions to remember
BATCH                 = 32 # size of minibatch
model_path            = "saved_model/"
observation_data_path = "observations/"
#======== Network Hyper Parameter
PADDING               = "SAME"
INPUT_SIZE_WIDTH      = 320
INPUT_SIZE_HEIGHT     = 80
LAYER_1_FILTER_SIZE   = 8 # 8x8
LAYER_1_FILTER_COUNT  = 4
LAYER_1_FILTER_STRIDE = 4
LAYER_2_FILTER_SIZE   = 4 # 4x4
LAYER_2_FILTER_COUNT  = 64
LAYER_2_FILTER_STRIDE = 2
LAYER_3_FILTER_SIZE   = 3 # 3x3
LAYER_3_FILTER_COUNT  = 64
LAYER_3_FILTER_STRIDE = 1
FC_LAYER_1_SIZE       = [1, 640] # input to fully connected layer
OUT_LAYER_SIZE        = [1, ACTIONS] 
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
def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([640, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 320, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    h_conv3_flat = tf.reshape(h_pool3, [-1, 640])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess, mode):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(model_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    stateid = 0
    x_t, r_0, terminal,stateid = get_a_frame(do_nothing, stateid, None)
    x_t1 = x_t
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # start training
    epsilon = 0.2
    t = 0
    if mode == 1:
        while t <= OBSERVE:
            start_time = time.time()
	    readout_t = readout.eval(feed_dict={s : [s_t]})[0]
	    a_t = np.zeros([ACTIONS])
	    action_index = 0
	    if random.random() <= epsilon:
	    	action_index = random.randrange(ACTIONS)
		a_t[random.randrange(ACTIONS)] = 1
	    else:
		action_index = np.argmax(readout_t)
		a_t[action_index] = 1

	    # run the selected action and observe next state and reward
	    x_t1, r_t, terminal, stateid  = get_a_frame(a_t, stateid, x_t1)
	    x_t1 = np.reshape(x_t1, (80, 320, 1))
	    s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
	    # store the transition in D
	    D.append((s_t, a_t, r_t, s_t1, terminal))
	    if len(D) > REPLAY_MEMORY:
	  	D.popleft()
	    # update the old values
	    s_t = s_t1
	    t += 1
            print("TIMESTEP", t, "/ STATE", mode, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
	# check number of observations
	path, dirs, files = os.walk(observation_data_path).next()
	number_of_obser = len(files)
	filename = observation_data_path+"observation_"+str(number_of_obser+1)
	if number_of_obser > 20:
	    files = sorted(os.listdir(observation_data_path), key=os.path.getctime)
	    os.system("rm "+files[0])#remove oldest file
        with open(filename,'wb') as fp:
            pickle.dump(D,fp)

    if mode == 2:
        # load observations
	T = deque()
	path, dirs, memories = os.walk(observation_data_path).next()
	number_of_obser = len(memories)
        if number_of_obser > 0:
	    for memory in memories:
		memory = observation_data_path+memory
	        with open(memory,'rb') as fp:
			instance = pickle.load(fp)
	        T.extend(instance)
            # sample a minibatch to train on
            minibatch = random.sample(T, 1)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]
            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )
            saver.save(sess, model_path + GAME + '-dqn', global_step = 1)
	    print("cost")
        
#========= Image processing
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def did_game_end(fail_frame):
    fail_reference = cv2.imread("temp/fail_reference.png")
    fail_reference = cv2.cvtColor(fail_reference, cv2.COLOR_BGR2GRAY)
    diff = mse(fail_frame, fail_reference)
    game_status = False
    if diff < 0.1:
        game_status = True
    else:
        game_status = False
    return game_status 

def process_a_frame(frame):
    h, w, c = frame.shape
    gray_screen = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cropping
    game_section = gray_screen[(h/4)-80:(h/4)+150,(w/2)-610:(w/2)+630]
    fail_section = gray_screen[(h/4)+20:(h/4)+76,(w/2)-30:(w/2)+30]
    game_section = cv2.resize(game_section, (320, 80))
    ret, game_section = cv2.threshold(game_section, 128, 255, cv2.THRESH_BINARY)
    return game_section, fail_section

#========= Game Manager
def jump():
    keys = "space"
    os.system('osascript -e \'tell application "System Events" to keystroke '+keys+"'")

def take_action(action):
    if action[1] == 1:
        jump()

def save_frames(frame, action, stateid):
   ac = "_a0_"
   if action[1] == 1:
       ac = "_a1_"
   frame_name = "frame_"+ac+str(stateid)+".png"
   cv2.imwrite("frames/"+frame_name, frame)
   if (stateid % 100) == 0:
       os.system("rm -rf frames/*")

def get_a_frame(action, stateid, prev_frame):
    reward = 0.1
    cmd = "screencapture -x -m temp/screenshot.png"
    take_action(action)
    os.system(cmd)
    new_frame  = cv2.imread("temp/screenshot.png")
    gamesec, failsec = process_a_frame(new_frame)
    backup_frame = new_frame
    terminal = did_game_end(failsec)
    if terminal:
        reward = -1.0
        if stateid == 0:
            prev_frame = gamesec
        gamesec = prev_frame
        jump()# reset
    if save_frames:
        save_frames(prev_frame, action, stateid)
    return gamesec, reward, terminal, stateid+1

def generate_game_fail_reference():
    print("make sure the game is in fail state!!")
    ok = raw_input("press return key")
    cmd = "screencapture -x -m temp/screenshot.png"
    os.system(cmd)
    screen = cv2.imread("temp/screenshot.png")
    gamesec, failsec = process_a_frame(screen)
    cv2.imwrite("temp/fail_reference.png",failsec)
    time.sleep(1)
#======================

#========= Setup
def set_directories():
    if not (os.path.isdir("temp/")):
       os.system("mkdir temp/")
    if not (os.path.isdir("frames/")):
        os.system("mkdir frames/")
    if not (os.path.isdir(model_path)):
	os.system("mkdir "+model_path)
    if not (os.path.isdir(observation_data_path)):
        os.system("mkdir "+observation_data_path)
    os.system("rm -rf frames/*")
    if create_fail_reference:
        generate_game_fail_reference()

def main():
    mode = int(sys.argv[1])
    print(mode)
    set_directories()
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess, mode)

if __name__ == "__main__":
	main()
