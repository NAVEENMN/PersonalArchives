from __future__ import print_function

import tensorflow as tf
import fmg as fr
import numpy as np
from scipy import spatial
import os
import glob
import time
import datetime as dt
import math
import random

#game setup
roll = 3 # number of games
trial = 1
model_path = "rintmp1/model.ckpt"

# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 100
display_step = 1


# Network Parameters
n_hidden_1 = 10 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 13952#  data input (img shape: 1096 x 2080)
n_classes = 1 #  probablity of pressing space

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

def explore():
	fr.press_space()
	time.sleep(0.2)
	fr.press_space()
	#time.sleep(0.2)
	#fr.press_space()
	#time.sleep(0.5)
	#fr.press_space()
	#time.sleep(0.1)
	#fr.press_space()

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with Sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with Sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Output layer with RELU activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = tf.nn.relu(out_layer)
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
def batch_rolling(batch_x, batch_y):
	saver = tf.train.Saver()
	with tf.Session() as sess:
    		sess.run(init)
		saver.restore(sess, model_path)
    		# Training cycle
    		for epoch in range(training_epochs):
			#print("epoch: %d"%epoch)
			#batch_x = [[1.0]*1402880]
			#batch_y = [[1.0]]
            		# Run optimization op (backprop) and cost op (to get loss value)
            		_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
  		print("Optimization Finished!")
		save_path = saver.save(sess, model_path)

def print_roll_session(mode, prob, fitness, roll, runtime):
	print("======")
	print("roll %d"%roll)
	print("out %d"%prob)
	print("Prob: %d"%sigmoid(prob))
	print("Fit: %d"%fitness)
	print("time: %d"%runtime)
	print(mode)
	print("======")		

def get_action(frame_diff):
	prediction = 0.0
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		saver.restore(sess, model_path)
		prediction = sess.run(pred, feed_dict={x: frame_diff})
		prediction = float(prediction[0][0])
		return prediction

def generate_batch():
	prob_press = 0.0
	saver = tf.train.Saver()
	fr.clean_rollouts()
	fitness = 1
	fitness_threshold = 4
        rewards = 0
	fail_trigger, fail_ref = fr.load_image("fail_ref.png")
	IMGS = []
	true_class = list()
	with tf.Session() as sess:
		sess.run(init)
		saver.restore(sess, model_path)
		for r in range(0, roll):
			game_going = True
			print("--game: %r"%r)
			os.system("rm frame.png")
			os.system("rm cfail_frame.png")
			game_frame, fail_frame, game_frame_img = fr.get_frame()
			result = 1 - spatial.distance.cosine(fail_frame, fail_ref)
			if result > 0.95:
				fr.new_game()#reset
			start_time = time.time()
			prev_game_arr = []
			frame_count = 0
			while game_going:
				game_frame1, fail_frame, game_frame = fr.get_frame()
				if len(prev_game_arr) == 0:
					prev_game_arr = game_frame1
					continue
				fr.save_an_episode(game_frame, "current")
				frame_diff = np.subtract(game_frame1, prev_game_arr)
				frame_count = frame_count + 1
				print(frame_diff.size)
				if len(IMGS) == 0:
					IMGS = frame_diff
				else:
					IMGS = np.append(IMGS, frame_diff, axis=0)
				s_t = time.time()
				prediction = sess.run(pred, feed_dict={x: frame_diff})
				prediction = float(prediction[0][0])
				prediction_time = time.time() - s_t
				prob_press =  prediction
				if(prob_press >= 0):
					explore()
					fr.press_space()
					fitness = fitness + 1
				result = 1 - spatial.distance.cosine(fail_frame, fail_ref)
				print (result)
				if result > 0.95:#game ends
					elapsed_time = time.time() - start_time
					elapsed_time = elapsed_time - prediction_time
					if fitness >= fitness_threshold:
						rewards = rewards + 2
						print_roll_session("positive", prob_press, fitness, r, elapsed_time)
						fr.save_an_episode(None, "positive")
					else:
						rewards = rewards - 1
						print_roll_session("negative", prob_press, fitness, r, elapsed_time)
						fr.save_an_episode(None, "negative")
					fitness = 0
					game_going = False
				else:#good game
					game_going = True
					fail_count = 0
					fitness = fitness + 1
					rewards = rewards + 1
			for _ in range(0, frame_count):
				true_class.append([rewards])
		print (IMGS.shape)
		print (true_class)
	return IMGS, true_class
		
def main():
	for t in range(0, trial):
		print (t)
		print("--Roll outs--")
		IMGS, true_class = generate_batch()
		time.sleep(2)
		print("--Training--")
		batch_rolling(IMGS, true_class)
		#time.sleep(5)
	
if __name__ == "__main__":
	main()
