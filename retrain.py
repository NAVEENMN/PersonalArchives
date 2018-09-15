import numpy as np
import tensorflow as tf
from utils import *
from PIL import Image

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
IMAGE_CHANNEL = 3
LATENT_DIM = 2048
BATCH_SIZE = 30
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL
SHAPE_IM = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
LATENT_SHAPE = [None, LATENT_DIM]
CLASS_SIZE = 4
SHAPE_CLS = [None, CLASS_SIZE] # set the shape based on number of classes
SHAPE_PICK = [None, 1]
import random
from bGraph import *
from utils import *

SAVED = "saved_model/"
LOG = "tensor_board/"
DATA_PATH = "data/"
BOTTLE_NECK_GRAPH_PATH = "pb_models/classify_image_graph_def.pb"

class embed():
	def __init__(self, name):
		self.name = name
		self.opt = tf.train.AdamOptimizer(0.001)
		
	def latent(self, image, reuse=False):
		n = self.name
		with tf.variable_scope("layers"):
			conv1 = tf.layers.conv2d(image, 32, 10, activation=tf.nn.relu, name=n + "conv1")
			conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name=n + "conv1_pool")
			conv2 = tf.layers.conv2d(conv1, 64, 8, activation=tf.nn.relu, name=n + "conv2")
			conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name=n + "conv2_pool")
			conv3 = tf.layers.conv2d(conv2, 128, 5, activation=tf.nn.relu, name=n + "conv3")
			conv3 = tf.layers.max_pooling2d(conv3, 2, 2, name=n + "conv3_pool")
			conv3 = tf.layers.conv2d(conv3, 128, 3, activation=tf.nn.relu, name=n + "conv4")
			conv4 = tf.layers.max_pooling2d(conv3, 2, 2, name=n + "conv4_pool")
			h_conv4_flat = tf.contrib.layers.flatten(conv4)
			fc1 = tf.layers.dense(h_conv4_flat, 1024, activation=tf.nn.sigmoid, name=n + "E_fc1")
			z = tf.layers.dense(fc1, LATENT_DIM, activation=tf.nn.sigmoid, name=n + "E_fc2")
		return z
	
	def get_loss(self, prediction, target):
		loss = tf.losses.mean_squared_error(labels=target, predictions=prediction)
		cost = tf.reduce_mean(loss)
		return cost
	
	def train_step(self, cost):
		tvars = tf.trainable_variables()
		vars = [var for var in tvars if self.name in var.name]
		return self.opt.minimize(cost, var_list=vars)

class classifer():
	def __init__(self, name):
		self.name = name
		self.opt = tf.train.AdamOptimizer(0.0001)
	
	def classify(self, latent, reuse=False):
		n = self.name
		with tf.variable_scope('FC_layers', reuse=reuse):
			fc1 = tf.layers.dense(latent, 1200, activation=tf.nn.relu, name=n+"FC_1")
			fc6 = tf.layers.dense(latent, 800, activation=tf.nn.relu, name=n+"FC_6")
			
			fc2 = tf.layers.dense(fc1, 800, activation=tf.nn.tanh, name=n+"FC_2")
			fc3 = tf.layers.dense(fc2, 500, activation=tf.nn.tanh, name=n+"FC_3")
			
			class_pred = tf.layers.dense(fc3, CLASS_SIZE, activation=tf.nn.softmax, name="pred_out")
			prob_pick = tf.layers.dense(fc6, 1, activation=tf.nn.sigmoid, name="pick_out")
		return class_pred, prob_pick
	
	def get_loss(self, prediction, target):
		class_pred, prob_pick = prediction
		target_clss, target_pick = target
		class_pred_loss = tf.losses.softmax_cross_entropy(onehot_labels=target_clss,
														  logits=class_pred)
		prob_pick_loss = tf.losses.log_loss(labels=target_pick,
											predictions=prob_pick)
		total_loss = tf.reduce_mean(class_pred_loss) + tf.reduce_mean(prob_pick_loss)
		return class_pred_loss, prob_pick_loss, total_loss
	
	def get_accuracy(self, prediction, target):
		accuracy = tf.contrib.metrics.accuracy(predictions=tf.argmax(prediction, 1),
									           labels=tf.argmax(target, 1))
		return accuracy
		
	def train_step(self, cost):
		tvars = tf.trainable_variables()
		vars = [var for var in tvars if self.name in var.name]
		return self.opt.minimize(cost, var_list=vars)

class network():
	def __init__(self, sess):
		self.sess = sess
		with tf.name_scope("Inputs"):
			self.input_ph = tf.placeholder("float", SHAPE_IM, name="input_ph")
			self.latent_target_ph = tf.placeholder("float", LATENT_SHAPE, name="latent_target_ph")
			self.target_ph = tf.placeholder("float", SHAPE_CLS, name="target_ph")
			self.target_pick = tf.placeholder("float", SHAPE_PICK, name="target_pick_ph")
		
		with tf.name_scope("Embed"):
			emb = embed("embed_")
			self.latent_vec = emb.latent(self.input_ph)
			self.embed_loss = emb.get_loss(prediction=self.latent_vec,
										   target=self.latent_target_ph)
			self.train_embed = emb.train_step(self.embed_loss)
			
		with tf.name_scope("Classify"):
			clsfy = classifer("classify_")
			self.class_pred, self.prob_pick = clsfy.classify(self.latent_vec)
			self.class_pred_loss, self.prob_pick_loss, self.total_loss = clsfy.get_loss(prediction=[self.class_pred, self.prob_pick],
																						target=[self.target_ph, self.target_pick])
			self.train_embed = clsfy.train_step(self.total_loss)
			self.accuracy = clsfy.get_accuracy(prediction=self.class_pred,
											   target=self.target_ph)
			
	def get_latent(self, image):
		
		return self.sess.run(self.latent_vec, feed_dict={self.input_ph: image})
'''
Run data through bGraph and collect
outputs of bottle neck layer
'''
def pre_process_data(bGraph):
	return process_bGraph_images(bGraph, DATA_PATH)
	
def main():
	with tf.Session() as sess:
		bGraph = bottle_neck_graph(BOTTLE_NECK_GRAPH_PATH, sess)
		net = network(sess)
		sess.run(tf.global_variables_initializer())
		data = pre_process_data(bGraph)
		print(data)
		

if __name__ == "__main__":
	main()