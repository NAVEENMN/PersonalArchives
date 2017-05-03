 #!/usr/bin/env python
from __future__ import print_function

import cv2
import glob
import random
import os.path
import datetime
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


#Log & Models
DATA_PATH       = "data\\"
MODEL_PATH      = "models\\"
LOG             = "Tensorboard\\"
#Image Parameters
IMAGE_WIDTH     = 28
IMAGE_HEIGHT    = 28
IMAGE_CHANNEL   = 1
#Network Parameters
z_dim           = 100 
learning_rate   = 0.0001


#======== Generative Network Hyper Parameter
G_PADDING               = "SAME"
G_FC_LAYER_1_SIZE       = IMAGE_WIDTH * IMAGE_HEIGHT * 4 # input to fully connected layer
G_LAYER_1_FILTER_SIZE   = 3 # 3x3
G_LAYER_1_FILTER_COUNT  = 50 # generate 50 features
G_LAYER_1_STRIDE        = 2
G_LAYER_2_FILTER_SIZE   = 3 # 3x3
G_LAYER_2_FILTER_COUNT  = 25 # generate 25 features
G_LAYER_2_STRIDE        = 2
G_LAYER_3_FILTER_SIZE   = 1 # 1x1
G_LAYER_3_FILTER_COUNT  = 1 
G_LAYER_3_STRIDE        = 2
G_OUTPUT_SIZE_WIDTH     = IMAGE_WIDTH
G_OUTPUT_SIZE_HEIGHT    = IMAGE_HEIGHT
#======== Discriminative Network Hyper Parameter
D_PADDING               = "SAME"
D_INPUT_SIZE_WIDTH      = IMAGE_WIDTH
D_INPUT_SIZE_HEIGHT     = IMAGE_HEIGHT
D_LAYER_1_FILTER_SIZE   = 5 # 5x5
D_LAYER_1_FILTER_COUNT  = 32
D_LAYER_1_STRIDE        = 1
D_LAYER_2_FILTER_SIZE   = 5 # 5x5
D_LAYER_2_FILTER_COUNT  = 64
D_LAYER_2_STRIDE        = 1
D_LAYER_3_FILTER_SIZE   = 3 # 3x3
D_LAYER_3_FILTER_COUNT  = 64
D_LAYER_3_FILTER        = 1
D_LAYER_3_STRIDE        = 1
D_FC_LAYER_1_SIZE       = 4*4*D_LAYER_3_FILTER_COUNT # input to fully connected layer
D_FC_LAYER_2_SIZE       = 512 # fully connected layer
D_OUT_LAYER_SIZE        = 1

# tf Graph input
#======== Network
def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.02))

def bias_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.constant_initializer(0))

def conv2d(x, W, stride):
	return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def avg_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def discriminator(x_image, keep_prob, reuse=False):
	if reuse:
		tf.get_variable_scope().reuse_variables()

	W_conv1 = weight_variable('d_w1', [D_LAYER_1_FILTER_SIZE, D_LAYER_1_FILTER_SIZE, IMAGE_CHANNEL, D_LAYER_1_FILTER_COUNT])
	b_conv1 = bias_variable('d_b1', [D_LAYER_1_FILTER_COUNT])
	W_conv2 = weight_variable('d_w2', [D_LAYER_2_FILTER_SIZE, D_LAYER_2_FILTER_SIZE, D_LAYER_1_FILTER_COUNT, D_LAYER_2_FILTER_COUNT])
	b_conv2 = bias_variable('d_b2', [D_LAYER_2_FILTER_COUNT])
	W_conv3 = weight_variable('d_w3', [D_LAYER_3_FILTER_SIZE, D_LAYER_3_FILTER_SIZE, D_LAYER_2_FILTER_COUNT, D_LAYER_3_FILTER_COUNT])
	b_conv3 = bias_variable('d_b3', [D_LAYER_3_FILTER_COUNT])

	W_fc1 = weight_variable('d_fcw1',[D_FC_LAYER_1_SIZE, D_FC_LAYER_2_SIZE])
	b_fc1 = bias_variable('d_fcb1',[D_FC_LAYER_2_SIZE])
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
	h_fc1   = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
	# Dropout
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	return readout

def generator(batch_size, z_dim):
	z = tf.truncated_normal([batch_size, z_dim], mean=0, stddev=1, name='z') #Noise

	W_dfc1   = weight_variable('g_wfc1',[z_dim, G_FC_LAYER_1_SIZE])    
	b_dfc1   = bias_variable('g_bfc1',[G_FC_LAYER_1_SIZE])
	W_dconv1 = weight_variable('g_w1', [G_LAYER_1_FILTER_SIZE, G_LAYER_1_FILTER_SIZE, IMAGE_CHANNEL, G_LAYER_1_FILTER_COUNT])
	b_dconv1 = bias_variable('g_b1', [G_LAYER_1_FILTER_COUNT])
	W_dconv2 = weight_variable('g_w2', [G_LAYER_2_FILTER_SIZE, G_LAYER_2_FILTER_SIZE, G_LAYER_1_FILTER_COUNT, G_LAYER_2_FILTER_COUNT])
	b_dconv2 = bias_variable('g_b2', [G_LAYER_2_FILTER_COUNT])
	W_dconv3 = weight_variable('g_w3', [G_LAYER_3_FILTER_SIZE, G_LAYER_3_FILTER_SIZE, G_LAYER_2_FILTER_COUNT, IMAGE_CHANNEL])
	b_dconv3 = bias_variable('g_b3', [IMAGE_CHANNEL])

	#first deconv block 
	h_dconv  = tf.reshape((tf.matmul(z, W_dfc1) + b_dfc1), [-1, IMAGE_WIDTH*2, IMAGE_HEIGHT*2, IMAGE_CHANNEL])
	h_dpool  = tf.contrib.layers.batch_norm(h_dconv, epsilon=1e-5, scope='bn') # opposite to pooling
	h_dconv1 = tf.nn.relu(h_dpool)
	
	# Generate 50 features
	h_dconv2 = conv2d(h_dconv1, W_dconv1, G_LAYER_1_STRIDE) + b_dconv1
	h_dpool1 = tf.contrib.layers.batch_norm(h_dconv2, epsilon=1e-5, scope='bn1')
	h_dconv2 = tf.image.resize_images(tf.nn.relu(h_dpool1),  [IMAGE_WIDTH*2, IMAGE_HEIGHT*2])

	# Generate 25 features
	h_dconv2 = conv2d(h_dconv2, W_dconv2, G_LAYER_2_STRIDE) + b_dconv2
	h_dpool2 = tf.contrib.layers.batch_norm(h_dconv2, epsilon=1e-5, scope='bn2')
	h_dconv3 = tf.image.resize_images(tf.nn.relu(h_dpool2),  [IMAGE_WIDTH*2, IMAGE_HEIGHT*2])

	# Final convolution with one output channel
	h_dconv3 = conv2d(h_dconv3, W_dconv3, G_LAYER_3_STRIDE) + b_dconv3
	h_dconv3 = tf.sigmoid(h_dconv3)
	return h_dconv3

def trainNetwork(sess, X):
	x_placeholder = tf.placeholder("float", [None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
	keep_prob  = tf.placeholder(tf.float32)
	batch_size = 20
	z_dimensions = 100

	Gz = generator(batch_size, z_dimensions)
	Dx = discriminator(x_placeholder, keep_prob)
	Dg = discriminator(Gz, keep_prob, reuse=True)
	
	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg))) 
	d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([batch_size, 1], 0.9)))
	d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
	d_loss = d_loss_real + d_loss_fake

	tvars  = tf.trainable_variables()
	d_vars = [var for var in tvars if 'd_' in var.name]
	g_vars = [var for var in tvars if 'g_' in var.name]
	
	with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
		d_trainer_fake = tf.train.AdamOptimizer(learning_rate).minimize(d_loss_fake, var_list=d_vars)
		d_trainer_real = tf.train.AdamOptimizer(learning_rate).minimize(d_loss_real, var_list=d_vars)
		g_trainer      = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

	d_real_count_ph = tf.placeholder(tf.float32)
	d_fake_count_ph = tf.placeholder(tf.float32)
	g_count_ph = tf.placeholder(tf.float32)
	# Sanity check to see how the discriminator evaluates
	d_on_generated = tf.reduce_mean(discriminator(generator(batch_size, z_dimensions), keep_prob))
	d_on_real = tf.reduce_mean(discriminator(x_placeholder, keep_prob))

	#Outputs a Summary protocol buffer containing a single scalar value
	tf.summary.scalar('Generator_loss', g_loss)
	tf.summary.scalar('Discriminator_loss_real', d_loss_real)
	tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)
	tf.summary.scalar('d_real_count', d_real_count_ph)
	tf.summary.scalar('d_fake_count', d_fake_count_ph)
	tf.summary.scalar('g_count', g_count_ph)
	tf.summary.scalar('d_on_generated_eval', d_on_generated)
	tf.summary.scalar('d_on_real_eval', d_on_real)

	images_for_tensorboard = generator(batch_size, z_dimensions)
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
	while True:
		X = shuffle(X, random_state=random.randint(1, 10))  # shuffle train data
		real_image_batch = X[:batch_size]
		if dLossFake > 0.6:
			# Train discriminator on generated images
			_, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake, g_loss],
													{x_placeholder: real_image_batch,  keep_prob: 0.5})
			d_fake_count += 1

		if gLoss > 0.5:
			# Train the generator
			_, dLossReal, dLossFake, gLoss = sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss],
													{x_placeholder: real_image_batch,  keep_prob: 0.5})
			g_count += 1

		if dLossReal > 0.45:
			# If the discriminator classifies real images as fake,
			# train discriminator on real values
			_, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],
													{x_placeholder: real_image_batch,  keep_prob: 0.5})
			d_real_count += 1

		if i % 10 == 0:
			X = shuffle(X, random_state=random.randint(1, 10))  # shuffle train data
			real_image_batch = X[:batch_size]
			summary = sess.run(merged, {x_placeholder: real_image_batch, d_real_count_ph: d_real_count,
									d_fake_count_ph: d_fake_count, g_count_ph: g_count,  keep_prob: 0.1})
			writer.add_summary(summary, i)
			d_real_count, d_fake_count, g_count = 0, 0, 0

		if i % 1000 == 0:
			# Periodically display a sample image in the notebook
			# (These are also being sent to TensorBoard every 10 iterations)
			images = sess.run(generator(3, z_dimensions))
			
			d_result = sess.run(discriminator(x_placeholder, keep_prob), {x_placeholder: images, keep_prob: 0.1})
			print("TRAINING STEP", i, "AT", datetime.datetime.now())
			for j in range(3):
				print("Discriminator classification", d_result[j])
				im = images[j, :, :, :]
				#visulize_a_image(im.reshape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]))

		if i % 5000 == 0:
			save_path = saver.save(sess, MODEL_PATH+"\\pretrained_gan.ckpt", global_step=i)
			print("saved to %s" % save_path)
		i = i + 1
	# evaluation
	'''
	test_images = sess.run(generator(10, 100))
	test_eval = sess.run(discriminator(x_placeholder), {x_placeholder: test_images})

	real_images = mnist.validation.next_batch(10)[0].reshape([10, 28, 28, 1])
	real_eval = sess.run(discriminator(x_placeholder), {x_placeholder: real_images})
	'''

def visulize_a_image(image):
	image = np.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
	cv2.imshow("image", image)
	print(image)
	cv2.waitKey()

def load_data():
	X, Y = None, None
	cla = np.zeros(10)
	images = list()
	classes = list()
	for x in range(0, 10):
		for filename in glob.glob(DATA_PATH+str(x)+'\\*.jpg'):
			img = cv2.imread(filename,0)
			img = img / 255.
			X = np.reshape(img, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
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
	#visulize_a_image(X[0])
	trainNetwork(sess, X)

if __name__ == "__main__":
	main()