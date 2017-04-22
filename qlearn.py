#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import cv2
import time
import signal
import random
import os.path
import datetime
import subprocess
import numpy as np
import PIL.ImageOps
import multiprocessing
import tensorflow as tf
from threading import Thread
from collections import deque
from subprocess import PIPE, run
from PIL import Image, ImageChops

#======== Handle Interrept and CLI
def sigint_handler(signum, frame):
    print("Exiting..")
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler)

def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=False)
    #cmd = command.split(" ")
    #ls_output=subprocess.Popen(result, stdout=subprocess.PIPE)
    return result

#======== Game Settings
save_frames           = False
create_fail_reference = False
GAME                  = 'DINO' # the name of the game being played for log files
ACTIONS               = 2 # number of valid actions
GAMMA                 = 0.99 # decay rate of past observations
FINAL_EPSILON         = 0.0001 # final value of epsilon
INITIAL_EPSILON       = 0.001 # starting value of epsilon
OBSERVE               = 100000. # timesteps to observe before training
EXPLORE               = 2000000.
REPLAY_MEMORY         = 5000 # number of previous transitions to remember
BATCH                 = 40 # size of minibatch
model_path            = "saved_networks\\"
observation_data_path = "observations\\"
EPOCHS                = 30
frame_per_sec         = 40
game_frame_x1         = "500" #pixel locations
game_frame_y1         = "140"
game_frame_x2         = "1100"
game_frame_y2         = "280"
game_fail_x1          = "783"
game_fail_y1          = "180"
game_fail_x2          = "816"
game_fail_y2          = "206"
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

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # store the previous observations in replay memory
    PD = deque()
    ND = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    stateid = 0
    x_t, r_0, terminal, stateid = get_a_frame(do_nothing, stateid, None)
    x_t1 = x_t
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    state = ""
    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(model_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    start_time = datetime.datetime.now()
    time_per_frame = frame_per_sec/1000000
    time_per_frame = 964188 # converted time_per_frame to microsec
    while True:
        now = datetime.datetime.now()
        elapsed = (now-start_time).microseconds
        # choose an action epsilon greedily
        if (t % frame_per_sec) == 0:
            readout_t = readout.eval(feed_dict={s : [s_t]})[0]
            a_t = np.zeros([ACTIONS])
            action_index = 0
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1

            # scale down epsilon
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # run the selected action and observe next state and reward
            x_t1, r_t, terminal, stateid = get_a_frame(a_t, stateid, x_t1)
            x_t1 = np.reshape(x_t1, (80, 320, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

            # store the transition in D
            if terminal:
                ND.append((s_t, a_t, r_t, s_t1, terminal))
            else:
                PD.append((s_t, a_t, r_t, s_t1, terminal))
            if len(PD) > REPLAY_MEMORY:
                PD.popleft()
            if len(ND) > REPLAY_MEMORY:
                ND.popleft()

            # only train if done observing
            if t > OBSERVE:
                # sample a minibatch to train on
                if min(len(PD), len(ND)) < BATCH:
                    B = min(len(PD), len(ND))
                else:
                    B = BATCH
                minibatch = random.sample(PD, B)
                Nminibatch = random.sample(ND, B)
                minibatch.extend(Nminibatch)
                minibatch = random.sample(minibatch, B)
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
            print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
            # update the old values
            s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess,  model_path + GAME + '-dqn', global_step = t)

        # print info
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        
       

#========= Image processing
def crop():
    section = game_frame_x1+','+game_frame_y1+','+game_frame_x2+','+game_frame_y2
    cmd = "D:\\boxcut\\boxcutter -c "+section+" temp\\game_reference.png"
    result = out(cmd)
    section = game_fail_x1+','+game_fail_y1+','+game_fail_x2+','+game_fail_y2
    cmd = "D:\\boxcut\\boxcutter -c "+section+" temp\\fail_reference.png"
    result = out(cmd)
    while not os.path.exists("temp\\game_reference.png"):
        i = 0 # wait till file created
    game_frame  = cv2.imread("temp\\game_reference.png")
    
    while not os.path.exists("temp\\fail_reference.png"):
        i = 0 
    fail_frame  = cv2.imread("temp\\fail_reference.png")
    return game_frame, fail_frame

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def did_game_end(fail_frame):
    fail_reference = cv2.imread("temp\\base_fail_reference.png")
    fail_reference = cv2.cvtColor(fail_reference, cv2.COLOR_BGR2GRAY)
    diff = mse(fail_frame, fail_reference)
    game_status = False
    if diff < 0.3:
        game_status = True
    else:
        game_status = False
    return game_status 

def process_a_frame(gamesec, fail_section):
    gray_screen = cv2.cvtColor(gamesec, cv2.COLOR_BGR2GRAY)
    fail_section = cv2.cvtColor(fail_section, cv2.COLOR_BGR2GRAY)
    game_section = cv2.resize(gray_screen, (320, 80))
    ret, game_section = cv2.threshold(game_section, 128, 255, cv2.THRESH_BINARY)
    return game_section, fail_section

#========= Game Manager
def jump():
    #subprocess.call([sys.executable, 'space.ahk'])
    os.system("START /B space.ahk")

def take_action(action):
    jump()
    time.sleep(0.2)#syncing time in space.ahk
    #if action[1] == 1:
    #    p = multiprocessing.Process(target=jump)
    #    p.start()

def save_frames(frame, action, stateid, ra):
   ac = "_a0_"
   if action[1] == 1:
       ac = "_a1_"
   frame_name = "frame_"+ra+ac+str(stateid)+".png"
   cv2.imwrite("frames/"+frame_name, frame)
   #if (stateid % 10000) == 0:
   #   os.system("del frames")

def get_a_frame(action, stateid, prev_frame):
    reward = 0.1
    ra = "_p_"
    take_action(action)
    gamesec, failsec = crop()
    gamesec, failsec = process_a_frame(gamesec, failsec)
    terminal = did_game_end(failsec)
    if terminal:
        reward = -1.0
        ra = "_n_"
        if stateid == 0:
            prev_frame = gamesec
        gamesec = prev_frame
        time.sleep(1)
        jump()# reset
    #if save_frames:
    #save_frames(prev_frame, action, stateid, ra)
    return gamesec, reward, terminal, stateid+1

#======================

#========= Setup
def set_directories():
    if not (os.path.isdir("temp\\")):
        os.system("mkdir temp\\")
    if not (os.path.isdir("frames\\")):
        os.system("mkdir frames\\")
    if not (os.path.isdir(model_path)):
        os.system("mkdir "+model_path)
    if not (os.path.isdir(observation_data_path)):
        os.system("mkdir "+observation_data_path)
    #os.system("del frames\\")
    if create_fail_reference:
        section = game_fail_x1+','+game_fail_y1+','+game_fail_x2+','+game_fail_y2
        cmd = "D:\\boxcut\\boxcutter -c "+section+" temp\\base_fail_reference.png"
        result = out(cmd)
        time.sleep(1)

def main():
    set_directories()
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

if __name__ == "__main__":
	main()
