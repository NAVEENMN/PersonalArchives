from defs import *
from models import *
from utils import *
import tensorflow as tf
import numpy as np

'''
sate en dec not good
action_endode loss not minimizing
policy needs bias advangates
'''

# ------ State encoder decoder -----#
class state_encoder_decoder():
	def __init__(self, name, latent_dim_size):
		self.name = name
		self.en_name = name+"_encode_"
		self.dec_name = name+"_decode_"
		self.latent_dim = latent_dim_size
		self.opt = tf.train.AdamOptimizer(0.0001)
	
	'''
	def encode(self, state, reuse):
		with tf.variable_scope(self.en_name, reuse=reuse):
			conv1 = tf.layers.conv2d(state, 32, 5, activation=tf.nn.relu, name=self.en_name+"conv1")
			conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name=self.en_name+"conv1_pool")
			conv2 = tf.layers.conv2d(conv1, 64, 5, activation=tf.nn.relu, name=self.en_name+"conv2")
			conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name=self.en_name+"conv2_pool")
			conv3 = tf.layers.conv2d(conv2, 128, 3, activation=tf.nn.relu, name=self.en_name+"conv3")
			conv3 = tf.layers.max_pooling2d(conv3, 2, 2, name=self.en_name+"conv3_pool")
			h_conv3_flat = tf.contrib.layers.flatten(conv3)
			fc1 = tf.layers.dense(h_conv3_flat, 1024, activation=tf.nn.tanh, name=self.en_name+"fc1")
			z = tf.layers.dense(fc1, self.latent_dim, activation=tf.nn.tanh, name=self.en_name+"fc2")
			return z
	'''
	
	def decode(self, state_latent, reuse):
		with tf.variable_scope(self.dec_name, reuse=reuse):
			fc1 = tf.layers.dense(state_latent, IMAGE_PIXELS * 4, name=self.dec_name+"fc")
			D_fc1 = tf.reshape(fc1, [-1, IMAGE_HEIGHT * 2, IMAGE_WIDTH * 2, IMAGE_CHANNEL])
			D_fc1 = tf.contrib.layers.batch_norm(D_fc1, epsilon=1e-5)
			D_fc1 = tf.nn.tanh(D_fc1)
			dconv1 = tf.layers.conv2d(D_fc1, 64, 5, activation=tf.nn.tanh, name=self.dec_name+"conv1")
			dconv1 = tf.contrib.layers.batch_norm(dconv1, epsilon=1e-5)
			dconv1 = tf.image.resize_images(dconv1, [IMAGE_HEIGHT * 2, IMAGE_WIDTH * 2])
			dconv2 = tf.layers.conv2d(dconv1, 32, 3, activation=tf.nn.tanh, name=self.dec_name+"conv2")
			dconv2 = tf.contrib.layers.batch_norm(dconv2, epsilon=1e-5)
			dconv2 = tf.image.resize_images(dconv2, [IMAGE_HEIGHT * 1, IMAGE_WIDTH * 1])
			image = tf.layers.conv2d(dconv2, 3, 1, activation=tf.nn.sigmoid, name=self.dec_name+"conv3")
			return image

	def get_loss(self, source, target):
		batch_flatten = tf.reshape(target, [BATCH_SIZE, -1])
		batch_reconstruct_flatten = tf.reshape(source, [BATCH_SIZE, -1])
		#loss = tf.losses.mean_squared_error(labels=batch_flatten, predictions=batch_reconstruct_flatten)
		#loss = tf.reduce_mean(loss)
		#return loss
		
		loss1 = batch_flatten * tf.log(1e-10 + batch_reconstruct_flatten)
		loss2 = (1 - batch_flatten) * tf.log(1e-10 + 1 - batch_reconstruct_flatten)
		loss = loss1 + loss2
		reconstruction_loss = -tf.reduce_sum(loss)
		loss = tf.reduce_mean(reconstruction_loss)
		return loss
		
	def train_step(self, loss):
		tvars = tf.trainable_variables()
		phi_vars = [var for var in tvars if self.name in var.name]
		with tf.variable_scope(tf.get_variable_scope()) as scope:
			train_ = self.opt.minimize(loss, var_list=phi_vars)
		return train_
	
	def get_vars(self):
		tvars = tf.trainable_variables()
		vars = [var for var in tvars if self.name in var.name]
		return vars

# ------ Action encoder decoder -----#
class action_encode_decoder():
	def __init__(self, name):
		self.name = name
		self.opt = tf.train.AdamOptimizer(0.001)
		self.enc_name = self.name + "_Enc_"
		self.dec_name = self.name + "_Dec_"

	def encode(self, action, reuse=False):
		n = self.enc_name
		with tf.variable_scope(n, reuse=reuse):
			fc1 = tf.layers.dense(action, LATENT_DIM * 4, activation=tf.nn.tanh, name=n + "fc1")
			fc2 = tf.layers.dense(fc1, LATENT_DIM * 2, activation=tf.nn.tanh, name=n + "fc2")
			action_latent = tf.layers.dense(fc2, LATENT_DIM, activation=tf.nn.tanh, name=n + "fc3")
		return action_latent

	def decode(self, action_latent, reuse=False):
		n = self.dec_name
		with tf.variable_scope(n, reuse=reuse):
			fc1 = tf.layers.dense(action_latent, LATENT_DIM / 2, activation=tf.nn.tanh, name=n + "fc1")
			action = tf.layers.dense(fc1, ACTION_SPACE, activation=tf.nn.softmax, name=n + "fc3")
		return action

	def get_loss(self, source, target):
		#loss = tf.losses.mean_squared_error(labels=target, predictions=source)
		loss = tf.losses.softmax_cross_entropy(onehot_labels=target, logits=source)
		#loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=source, labels=target)
		return tf.reduce_mean(loss)

	def train_step(self, loss):
		tvars = tf.trainable_variables()
		at_vars = [var for var in tvars if self.name in var.name]
		with tf.variable_scope(tf.get_variable_scope()) as scope:
			train_ = self.opt.minimize(loss, var_list=at_vars)
		return train_
	
	def get_vars(self):
		tvars = tf.trainable_variables()
		at_vars = [var for var in tvars if self.name in var.name]
		return at_vars

# critic later replace by SR
class critic():
	def __init__(self, name):
		self.name = name
		self.opt = tf.train.AdamOptimizer(learning_rate=0.001)
		
	def get_value(self, state_latent, reuse):
		n = self.name
		with tf.variable_scope(n, reuse=reuse):
			fc1 = tf.layers.dense(state_latent, 20, activation=tf.nn.relu, name=n+"fc1")
			fc2 = tf.layers.dense(fc1, 15, activation=tf.nn.tanh, name=n+"fc2")
			fc3 = tf.layers.dense(fc2, 5, activation=tf.nn.tanh, name=n+"fc3")
			value = tf.layers.dense(fc3, 1, activation=None, name=n+"fc4")
		return value
	
	def get_loss(self, value, target_value):
		loss = tf.losses.mean_squared_error(predictions=value, labels=target_value)
		return loss
	
	def train_step(self, loss):
		tvars = [var for var in  tf.trainable_variables() if self.name in var.name]
		train_ = self.opt.minimize(loss, var_list=tvars)
		return train_
	
	def get_vars(self):
		vars = [var for var in tf.trainable_variables() if self.name in var.name]
		return vars
	
class policy():
	def __init__(self, name):
		self.name = name
		self.rolls_per_batch = 1
		self.opt = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-4)
		
	def get_action(self, state, reuse):
		with tf.variable_scope(self.name, reuse=reuse):
			fc1 = tf.layers.dense(state, 30, activation=tf.nn.tanh, name=self.name + "fc1")
			fc2 = tf.layers.dense(fc1, 20, activation=tf.nn.tanh, name=self.name + "fc2")
			fc3 = tf.layers.dense(fc2, 10, activation=tf.nn.tanh, name=self.name + "fc3")
			out = tf.layers.dense(fc3, ACTION_SPACE, activation=tf.nn.softmax, name=self.name + "out")
		return out

	def get_scaled_grads(self, pi_out, advantages, rolls):
		pi_vars = [var for var in tf.trainable_variables() if self.name in var.name]
		grads = tf.gradients(tf.log(pi_out), pi_vars)
		scaled_grads = tf.gradients((tf.log(pi_out) * advantages) / rolls, pi_vars)
		return grads, scaled_grads
	
	def train_step(self, grads):
		pi_vars = [var for var in tf.trainable_variables() if self.name in var.name]
		train_step = self.opt.apply_gradients(zip(grads, pi_vars))
		return train_step
	
	def get_vars(self):
		vars = [var for var in tf.trainable_variables() if self.name in var.name]
		return vars
	
def tensorboard_summary(data):
	source_image = data["phi_input"]
	recon_image = data["phi_output"]
	with tf.name_scope("summary/images"):
		image_to_tb = tf.concat([source_image, recon_image], axis=1)
		#image_to_tb = tf.concat([image_to_tb, sr_feature_recon], axis=1)
		image_to_tb = (image_to_tb) * 255
		tf.summary.image('src', image_to_tb, 4)
		# tf.summary.image('recon', recon_image, 5)
	with tf.name_scope("summary/losses"):
		tf.summary.scalar("phi_loss", data["phi_loss"])
		tf.summary.scalar("theta_loss", data["theta_loss"])
		tf.summary.scalar("critic_loss", data["critic_loss"])