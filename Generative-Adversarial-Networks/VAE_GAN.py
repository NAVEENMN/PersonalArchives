#!/usr/bin/env python
from __future__ import print_function

import cv2
import glob
import random
import imageio
import sys
import os.path
import datetime
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

# Log & Models
DATA_PATH = "data\\"
MODEL_PATH = "models\\"
LOG = "Tensorboard\\"

# Image Parameters
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNEL = 1
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL

# Image Parameters
IMAGE_OUT_WIDTH = 28
IMAGE_OUT_HEIGHT = 28
IMAGE_OUT_CHANNEL = 1
IMAGE_OUT_PIXELS = IMAGE_OUT_HEIGHT * IMAGE_OUT_WIDTH * IMAGE_OUT_CHANNEL

# Network Parameters
z_dim = 32
learning_rate = 0.0001
global batch_size
batch_size = 50

# ======== Generative Network Hyper Parameter
G_PADDING = "SAME"
G_FC_LAYER_1_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 4  # input to fully connected layer
G_LAYER_1_FILTER_SIZE = 3  # 3x3
G_LAYER_1_FILTER_COUNT = 50  # generate 50 features
G_LAYER_1_STRIDE = 2
G_LAYER_2_FILTER_SIZE = 3  # 3x3
G_LAYER_2_FILTER_COUNT = 25  # generate 25 features
G_LAYER_2_STRIDE = 2
G_LAYER_3_FILTER_SIZE = 1  # 1x1
G_LAYER_3_FILTER_COUNT = 1
G_LAYER_3_STRIDE = 2
G_OUTPUT_SIZE_WIDTH = IMAGE_WIDTH
G_OUTPUT_SIZE_HEIGHT = IMAGE_HEIGHT
# ======== Discriminative Network Hyper Parameter
D_PADDING = "SAME"
D_INPUT_SIZE_WIDTH = IMAGE_WIDTH
D_INPUT_SIZE_HEIGHT = IMAGE_HEIGHT
D_LAYER_1_FILTER_SIZE = 5  # 5x5
D_LAYER_1_FILTER_COUNT = 32
D_LAYER_1_STRIDE = 1
D_LAYER_2_FILTER_SIZE = 5  # 5x5
D_LAYER_2_FILTER_COUNT = 64
D_LAYER_2_STRIDE = 1
D_LAYER_3_FILTER_SIZE = 3  # 3x3
D_LAYER_3_FILTER_COUNT = 64
D_LAYER_3_FILTER = 1
D_LAYER_3_STRIDE = 1
D_FC_LAYER_1_SIZE = 4 * 4 * D_LAYER_3_FILTER_COUNT  # input to fully connected layer
D_FC_LAYER_2_SIZE = 512  # fully connected layer
D_OUT_LAYER_SIZE = 1
# ======== ENCODER Network Hyper Parameter
ENCODER_INPUT_DIM = IMAGE_PIXELS
ENCODER_LATENT_DIM = z_dim


# tf Graph input
# ======== Network
def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.02))


def bias_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.constant_initializer(0))


def conv2d(x, W, stride):
	return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def avg_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# ------------------------- ENC0DER -------------------------------
'''
Encoder takes in inputs from self.batch_flatten in shape of [batch_size, image_width, image_height, image_channel]
and outputs means and standard deviations for all batch_size number of images.
There are two fully connected layer and last layer has two sets of weights and biases applied seperately.
one produces means and other standard deviations. Initially these outputs are garbage but will be trained
'''


def encoder(batch, reuse=False):
	if reuse:
		tf.get_variable_scope().reuse_variables()
	shape = batch.get_shape().as_list()
	batch_flatten = tf.reshape(batch, [batch_size, -1])
	# Encoder Layer 1
	W_efc1 = weight_variable('e_wfc1', [ENCODER_INPUT_DIM, ENCODER_LATENT_DIM])
	b_fc1 = bias_variable('e_fcb1', [ENCODER_LATENT_DIM])
	# Encoder Layer 2
	W_efc2 = weight_variable('e_wfc2', [ENCODER_LATENT_DIM, ENCODER_LATENT_DIM])
	b_fc2 = bias_variable('e_fcb2', [ENCODER_LATENT_DIM])
	# Encoder Layer 3
	W_efc3m = weight_variable('e_wfc3m', [ENCODER_LATENT_DIM, ENCODER_LATENT_DIM])
	b_fc3m = bias_variable('e_fcb3m', [ENCODER_LATENT_DIM])
	W_efc3sd = weight_variable('e_wfc3sd', [ENCODER_LATENT_DIM, ENCODER_LATENT_DIM])
	b_fc3sd = bias_variable('e_fcb3sd', [ENCODER_LATENT_DIM])
	
	EL1_OUT = tf.nn.dropout(tf.nn.softplus(tf.matmul(batch_flatten, W_efc1) + b_fc1), keep_prob=0.5)
	EL2_OUT = tf.nn.dropout(tf.nn.softplus(tf.matmul(EL1_OUT, W_efc2) + b_fc2), keep_prob=0.5)
	
	EL_MEAN = tf.matmul(EL2_OUT, W_efc3m) + b_fc3m
	EL_SD = tf.matmul(EL2_OUT, W_efc3sd) + b_fc3sd
	
	return (EL_MEAN, EL_SD)


# ------------------------------------------------------------------
# ---------------------- DISCRIMINATOR ----------------------------
'''
Discriminator takes images in shape [batch, image_width, image_height, image_channel]
and outputs an float value indicating its prediction weather the images it took as input
came from real images batch or fake generated images from generator
'''


def discriminator(x_image, keep_prob, reuse=False):
	if reuse:
		tf.get_variable_scope().reuse_variables()
	
	W_conv1 = weight_variable('d_w1',
							  [D_LAYER_1_FILTER_SIZE, D_LAYER_1_FILTER_SIZE, IMAGE_CHANNEL, D_LAYER_1_FILTER_COUNT])
	b_conv1 = bias_variable('d_b1', [D_LAYER_1_FILTER_COUNT])
	W_conv2 = weight_variable('d_w2', [D_LAYER_2_FILTER_SIZE, D_LAYER_2_FILTER_SIZE, D_LAYER_1_FILTER_COUNT,
									   D_LAYER_2_FILTER_COUNT])
	b_conv2 = bias_variable('d_b2', [D_LAYER_2_FILTER_COUNT])
	W_conv3 = weight_variable('d_w3', [D_LAYER_3_FILTER_SIZE, D_LAYER_3_FILTER_SIZE, D_LAYER_2_FILTER_COUNT,
									   D_LAYER_3_FILTER_COUNT])
	b_conv3 = bias_variable('d_b3', [D_LAYER_3_FILTER_COUNT])
	
	W_fc1 = weight_variable('d_fcw1', [D_FC_LAYER_1_SIZE, D_FC_LAYER_2_SIZE])
	b_fc1 = bias_variable('d_fcb1', [D_FC_LAYER_2_SIZE])
	W_fc2 = weight_variable('d_fcw2', [D_FC_LAYER_2_SIZE, D_OUT_LAYER_SIZE])
	b_fc2 = bias_variable('d_fcb2', [D_OUT_LAYER_SIZE])
	
	# hidden layers
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, D_LAYER_1_STRIDE) + b_conv1)
	h_pool1 = avg_pool_2x2(h_conv1)
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, D_LAYER_2_STRIDE) + b_conv2)
	h_pool2 = avg_pool_2x2(h_conv2)
	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, D_LAYER_3_STRIDE) + b_conv3)
	h_pool3 = avg_pool_2x2(h_conv3)
	h_conv3_flat = tf.reshape(h_pool3, [-1, D_FC_LAYER_1_SIZE])
	h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
	# Dropout
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	return readout


# ------------------------------------------------------------

# ------------------------ GENERATOR ------------------------------
'''
Generator takes inputs latent vector and generates images
in shapes [batch_size, image_out_width, image_out_height, image_out_channel ].
During optimization image reconstruction error will optimized and encoders output
will be optimized to produce unit gaussian distributed noise.
'''
def generator(batch_size, z, reuse=False):
	if reuse:
		tf.get_variable_scope().reuse_variables()
	W_dfc1 = weight_variable('g_wfc1', [z_dim, G_FC_LAYER_1_SIZE])
	b_dfc1 = bias_variable('g_bfc1', [G_FC_LAYER_1_SIZE])
	W_dconv1 = weight_variable('g_w1',
							   [G_LAYER_1_FILTER_SIZE, G_LAYER_1_FILTER_SIZE, IMAGE_CHANNEL, G_LAYER_1_FILTER_COUNT])
	b_dconv1 = bias_variable('g_b1', [G_LAYER_1_FILTER_COUNT])
	W_dconv2 = weight_variable('g_w2', [G_LAYER_2_FILTER_SIZE, G_LAYER_2_FILTER_SIZE, G_LAYER_1_FILTER_COUNT,
										G_LAYER_2_FILTER_COUNT])
	b_dconv2 = bias_variable('g_b2', [G_LAYER_2_FILTER_COUNT])
	W_dconv3 = weight_variable('g_w3',
							   [G_LAYER_3_FILTER_SIZE, G_LAYER_3_FILTER_SIZE, G_LAYER_2_FILTER_COUNT, IMAGE_CHANNEL])
	b_dconv3 = bias_variable('g_b3', [IMAGE_CHANNEL])
	
	# first deconv block
	h_dconv = tf.reshape((tf.matmul(z, W_dfc1) + b_dfc1), [-1, IMAGE_WIDTH * 2, IMAGE_HEIGHT * 2, IMAGE_CHANNEL])
	h_dpool = tf.contrib.layers.batch_norm(h_dconv, epsilon=1e-5, scope='bn')  # opposite to pooling
	h_dconv1 = tf.nn.relu(h_dpool)
	
	# Generate 50 features
	h_dconv2 = conv2d(h_dconv1, W_dconv1, G_LAYER_1_STRIDE) + b_dconv1
	h_dpool1 = tf.contrib.layers.batch_norm(h_dconv2, epsilon=1e-5, scope='bn1')
	h_dconv2 = tf.image.resize_images(tf.nn.relu(h_dpool1), [IMAGE_WIDTH * 2, IMAGE_HEIGHT * 2])
	
	# Generate 25 features
	h_dconv2 = conv2d(h_dconv2, W_dconv2, G_LAYER_2_STRIDE) + b_dconv2
	h_dpool2 = tf.contrib.layers.batch_norm(h_dconv2, epsilon=1e-5, scope='bn2')
	h_dconv3 = tf.image.resize_images(tf.nn.relu(h_dpool2), [IMAGE_WIDTH * 2, IMAGE_HEIGHT * 2])
	
	# Final convolution with one output channel
	h_dconv3 = conv2d(h_dconv3, W_dconv3, G_LAYER_3_STRIDE) + b_dconv3
	h_dconv3 = tf.sigmoid(h_dconv3)
	return h_dconv3

# -------------------------------------------------

# ------------------------ GENERATOR LARGE images -------------------------
'''
Previous generative network trains quick but has a limitation on generating large images for
visulization. This network for developed with the help of this blog
http :// blog.otoro.net / 2016/04/01/generating-large-images-from-latent-vectors/
and its well documented how this generator network works
'''
def coordinates(x_dim = IMAGE_OUT_WIDTH, y_dim = IMAGE_OUT_HEIGHT, scale = 1.0):
	n_pixel = x_dim * y_dim
	x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
	y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
	x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
	y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
	r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
	x_mat = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, n_pixel, 1)
	y_mat = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, n_pixel, 1)
	r_mat = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, n_pixel, 1)
	return x_mat, y_mat, r_mat

def fully_connected(input_, output_size, scope=None, stddev=0.1, with_bias = True):
	shape = input_.get_shape().as_list()
	input_ = tf.cast(input_, tf.float32)
	with tf.variable_scope(scope or "FC"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
								 tf.random_normal_initializer(stddev=stddev))
		result = tf.matmul(input_, matrix)
		if with_bias:
			bias = tf.get_variable("bias", [1, output_size],initializer=tf.random_normal_initializer(stddev=stddev))
			result += bias*tf.ones([shape[0], 1], dtype=tf.float32)
		return result

# magnified generator
def mag_generator(batch_size, z, x, y, r, scale=1.0, reuse=True):
	if reuse:
		tf.get_variable_scope().reuse_variables()
	n_network = 128
	gen_n_points = IMAGE_OUT_WIDTH * IMAGE_OUT_HEIGHT
	
	z_scaled = tf.reshape(z, [batch_size, 1, ENCODER_LATENT_DIM]) * \
			   tf.ones([gen_n_points, 1], dtype=tf.float32) * scale
	z_unroll = tf.reshape(z_scaled, [batch_size * gen_n_points, ENCODER_LATENT_DIM])
	x_unroll = tf.reshape(x, [batch_size * gen_n_points, 1])
	y_unroll = tf.reshape(y, [batch_size * gen_n_points, 1])
	r_unroll = tf.reshape(r, [batch_size * gen_n_points, 1])
	
	U = fully_connected(z_unroll, n_network, 'g_0_z') + \
		fully_connected(x_unroll, n_network, 'g_0_x', with_bias=False) + \
		fully_connected(y_unroll, n_network, 'g_0_y', with_bias=False) + \
		fully_connected(r_unroll, n_network, 'g_0_r', with_bias=False)
	
	H = tf.nn.softplus(U)
	i = 0
	for i in range(1, 4):
		H = tf.nn.tanh(fully_connected(H, n_network, 'g_tanh_' + str(i)))
	
	output = tf.sigmoid(fully_connected(H, IMAGE_OUT_CHANNEL, 'g_' + str(i)))
	result = tf.reshape(output, [batch_size, IMAGE_OUT_WIDTH, IMAGE_OUT_HEIGHT, IMAGE_OUT_CHANNEL])
	return result
# -------------------------------------------------

def trainNetwork(sess, X, TRAIN=True):
	global batch_size
	x_placeholder = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
	keep_prob = tf.placeholder(tf.float32)
	x, y, r = coordinates()
	z_mean, z_sd = encoder(x_placeholder)
	eps = tf.random_normal((batch_size, ENCODER_LATENT_DIM), 0, 1, dtype=tf.float32)
	z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_sd)), eps))
	Gz = mag_generator(batch_size, z, x, y, r, reuse=False)
	
	# Testing Part
	# generate a normally distributed noise and control
	# the mean and standard deviation to see how image transistions.
	if (TRAIN == False):
		images = []
		mean, sd = 0, 1
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		checkpoint = tf.train.get_checkpoint_state(MODEL_PATH)
		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Could not find old network weights")
		ep = np.float32(np.random.normal(0, 1, 1 * ENCODER_LATENT_DIM))
		for t in range(0, 360, 10):
			mean = 0
			sd = -1*np.sin(t*np.pi/180.)    # standard deviation
			z = np.add(mean, np.multiply(np.sqrt(np.exp(sd)), ep))
			z = np.reshape(z, [1, ENCODER_LATENT_DIM])
			image =sess.run(mag_generator(batch_size=1, z=z, x=x, y=y, r=r, reuse=True))
			visulize_a_image(image/255., mean=mean, sd=sd)
			cv2.putText(image, "Mean: " + str(mean), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
			cv2.putText(image, "SD: " + str(sd), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
			images.append(np.reshape(image, (IMAGE_OUT_WIDTH, IMAGE_OUT_HEIGHT, IMAGE_OUT_CHANNEL)))
		imageio.mimsave('images\\large17.gif', images) #gif generate
		return
	
	Dx = discriminator(x_placeholder, keep_prob)
	Dg = discriminator(Gz, keep_prob, reuse=True)
	batch_flatten = tf.reshape(x_placeholder, [batch_size, -1])
	batch_reconstruct_flatten = tf.reshape(Gz, [batch_size, -1])
	
	# Image reconstruction loss measured using negative logits
	# Targets/labels(true) = P(x), logits(predicted) = y(x)
	# -Sigma(P(x)*log(y(x))+(1-P(x))*log(1-y(x)))
	# 1e-10 added to avoid log(0)
	reconstruction_loss = -tf.reduce_sum(batch_flatten * tf.log(1e-10 + batch_reconstruct_flatten) + \
										 (1 - batch_flatten) * tf.log(1e-10 + 1 - batch_reconstruct_flatten), 1)
	# latent loss measured using KL Divergence
	# Initally latent noise is random normal distributed
	# during optimization it will converge to unit gaussian distribution
	latent_loss = -0.5 * tf.reduce_sum(1 + z_sd - tf.square(z_mean) - tf.exp(z_sd), 1)
	vae_loss = tf.reduce_mean(reconstruction_loss + latent_loss) / (IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL)
	
	# Generator loss with sigmoid cross entropy, Discriminator should think its real images
	# So try to optimize to vector of ones
	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))
	# Discriminator fed with real images, optimize to vector of ones.
	d_loss_real = tf.reduce_mean(
		tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([batch_size, 1], 0.9)))
	# Discriminator fed with fake images generated from generator optimize to zeros
	d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
	d_loss = d_loss_real + d_loss_fake
	
	tvars = tf.trainable_variables()
	d_vars = [var for var in tvars if 'd_' in var.name]
	g_vars = [var for var in tvars if 'g_' in var.name]
	e_vars = [var for var in tvars if 'e_' in var.name]
	vae_vars = e_vars + g_vars
	
	with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
		d_trainer_fake = tf.train.AdamOptimizer(learning_rate).minimize(d_loss_fake, var_list=d_vars)
		d_trainer_real = tf.train.AdamOptimizer(learning_rate).minimize(d_loss_real, var_list=d_vars)
		g_trainer = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
		vae_trainer = tf.train.AdamOptimizer(learning_rate).minimize(vae_loss, var_list=vae_vars)
	
	d_real_count_ph = tf.placeholder(tf.float32)
	d_fake_count_ph = tf.placeholder(tf.float32)
	g_count_ph = tf.placeholder(tf.float32)
	
	# Sanity check to see how the discriminator evaluates
	d_on_generated = tf.reduce_mean(discriminator(mag_generator(batch_size, z, x, y, r), keep_prob))
	d_on_real = tf.reduce_mean(discriminator(x_placeholder, keep_prob))
	
	# Outputs a Summary protocol buffer containing a single scalar value
	tf.summary.scalar('Generator_loss', g_loss)
	tf.summary.scalar('Discriminator_loss', d_loss)
	tf.summary.scalar('VAE_loss', vae_loss)
	tf.summary.scalar('Discriminator_loss_real', d_loss_real)
	tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)
	tf.summary.scalar('d_real_count', d_real_count_ph)
	tf.summary.scalar('d_fake_count', d_fake_count_ph)
	tf.summary.scalar('g_count', g_count_ph)
	tf.summary.scalar('d_on_generated_eval', d_on_generated)
	tf.summary.scalar('d_on_real_eval', d_on_real)
	
	images_for_tensorboard = mag_generator(batch_size=batch_size, z=z, x=x, y=y, r=r)
	tf.summary.image('Generated_images', images_for_tensorboard, 10)
	merged = tf.summary.merge_all()
	logdir = LOG
	writer = tf.summary.FileWriter(logdir, sess.graph)
	print(logdir)
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	
	checkpoint = tf.train.get_checkpoint_state(MODEL_PATH)
	if checkpoint and checkpoint.model_checkpoint_path:
		saver.restore(sess, checkpoint.model_checkpoint_path)
		print("Successfully loaded:", checkpoint.model_checkpoint_path)
	else:
		print("Could not find old network weights")
	
	gLoss = 0
	dLossFake, dLossReal = 1, 1
	d_real_count, d_fake_count, g_count = 0, 0, 0
	i = 0
	
	# Training Part
	while True:
		X = shuffle(X, random_state=random.randint(1, 10))  # shuffle train data
		real_image_batch = X[:batch_size]
		
		if dLossFake > 0.6:
			# Train discriminator on generated images
			_, vaeLoss, dLossReal, dLossFake, gLoss = sess.run(
				[d_trainer_fake, vae_loss, d_loss_real, d_loss_fake, g_loss],
				{x_placeholder: real_image_batch, keep_prob: 0.5})
			d_fake_count += 1
		
		if gLoss > 0.5:
			# Train the generator and variational AE
			_, dLossReal, dLossFake, gLoss, _ = sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss, vae_trainer],
														 {x_placeholder: real_image_batch, keep_prob: 0.5})
			g_count += 1
		
		if dLossReal > 0.45:
			# If the discriminator classifies real images as fake,
			# train discriminator on real values
			_, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],
													  {x_placeholder: real_image_batch, keep_prob: 0.5})
			d_real_count += 1
		
		if i % 10 == 0:
			X = shuffle(X, random_state=random.randint(1, 10))  # shuffle train data
			real_image_batch = X[:batch_size]
			summary = sess.run(merged, {x_placeholder: real_image_batch, d_real_count_ph: d_real_count,
										d_fake_count_ph: d_fake_count, g_count_ph: g_count, keep_prob: 0.1})
			writer.add_summary(summary, i)
			d_real_count, d_fake_count, g_count = 0, 0, 0
		
		if i % 1000 == 0:
			ep = tf.random_normal((batch_size, ENCODER_LATENT_DIM), 0, 1, dtype=tf.float32)
			images = sess.run(mag_generator(batch_size, z=ep, x=x, y=y, r=r))
			d_result = sess.run(discriminator(x_placeholder, keep_prob), {x_placeholder: images, keep_prob: 0.1})
			print("TRAINING STEP", i, "AT", datetime.datetime.now())
			for j in range(3):
				print("Discriminator classification", d_result[j])
				
		if i % 5000 == 0:
			save_path = saver.save(sess, MODEL_PATH + "\\pretrained_gan.ckpt", global_step=i)
			print("saved to %s" % save_path)
		i = i + 1


def visulize_a_image(image, mean=0, sd=1):
	image = np.reshape(image, (IMAGE_OUT_WIDTH, IMAGE_OUT_HEIGHT, IMAGE_OUT_CHANNEL))
	image = image * 255.
	list = os.listdir("images\\")
	number_files = len(list)
	#cv2.imwrite("images\\" + str(number_files) + ".png", image)
	cv2.putText(image, "Mean: " + str(mean), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
	cv2.putText(image, "SD: " + str(sd), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
	cv2.imshow("image", image)
	cv2.waitKey(1)

def load_data():
	X, Y = None, None
	cla = np.zeros(10)
	images = list()
	classes = list()
	for x in range(0, 10):
		for filename in glob.glob(DATA_PATH + str(x) + '\\*.jpg'):
			img = cv2.imread(filename, 0)
			img = img / 255.
			X = np.reshape(img, (-1, 28, 28, 1))
			X = X.astype(np.float32)
			cla[x] = 1.
			images.append(X)
			classes.append(cla)
	X = np.vstack(images)
	Y = np.vstack(classes)
	X, Y = shuffle(X, Y, random_state=42)  # shuffle train data
	return X, Y


def main():
	sess = tf.InteractiveSession()
	X, Y = load_data()
	trainNetwork(sess, X, TRAIN=True)


if __name__ == "__main__":
	main()