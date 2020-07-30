import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
ACTIONS = 2 # number of valid actions

ENCODER_LATENT_DIM = 32

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

class graphs():
	def __init__(self):
		self.fig, self.ax = plt.subplots()
		self.feelings = ('Scared', 'Courage', 'Happy', 'Sad')
		self.y_pos = np.arange(len(self.feelings))
		self.error = np.random.rand(len(self.feelings))
		self.ax.set_yticks(self.y_pos)
		self.ax.set_yticklabels(self.feelings)
		self.ax.invert_yaxis()  # labels read top-to-bottom
		self.ax.set_xlabel('Feelings on scale')
		self.ax.set_title('Feelings measure')
		
	def plot(self, data):
		ax = self.ax
		plt.cla()
		ax.set_yticks(self.y_pos)
		ax.set_yticklabels(self.feelings)
		ax.barh(self.y_pos,
					 data,
					 xerr=self.error,
					 align='center',
					 color='green',
					 ecolor='black')
		plt.show()
		plt.pause(0.0001)

class familiarity():
	def __init__(self):
		# network weights
		W_conv1 = weight_variable([8, 8, 4, 32])
		b_conv1 = bias_variable([32])
		
		W_conv2 = weight_variable([4, 4, 32, 64])
		b_conv2 = bias_variable([64])
		
		W_conv3 = weight_variable([3, 3, 64, 64])
		b_conv3 = bias_variable([64])
		
		W_fc1 = weight_variable([1600, 512])
		b_fc1 = bias_variable([512])
		
		W_fc2 = weight_variable([512, ENCODER_LATENT_DIM])
		b_fc2 = bias_variable([ENCODER_LATENT_DIM])
		
		W_efc3m = weight_variable([ENCODER_LATENT_DIM, ENCODER_LATENT_DIM])
		b_fc3m = bias_variable([ENCODER_LATENT_DIM])
		W_efc3sd = weight_variable([ENCODER_LATENT_DIM, ENCODER_LATENT_DIM])
		b_fc3sd = bias_variable([ENCODER_LATENT_DIM])
		
		# input layer
		self.encoder_ph = tf.placeholder("float", [None, 80, 80, 4])
		
		# hidden layers
		h_conv1 = tf.nn.relu(conv2d(self.encoder_ph, W_conv1, 4) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)
		
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
		# h_pool2 = max_pool_2x2(h_conv2)
		
		h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
		# h_pool3 = max_pool_2x2(h_conv3)
		
		# h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
		h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
		
		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
		h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
		
		# readout layer
		self.EL_MEAN = tf.matmul(h_fc2, W_efc3m) + b_fc3m
		self.EL_SD = tf.matmul(h_fc2, W_efc3sd) + b_fc3sd
	
	def train_network(self):
		z_sd, z_mean = self.EL_SD, self.EL_MEAN
		latent_loss = -0.5 * tf.reduce_sum(1 + z_sd - tf.square(z_mean) - tf.exp(z_sd), 1)
		#self.latent_loss = tf.reduce_mean(latent_loss)
		kl_train = tf.train.AdamOptimizer(1e-6).minimize(latent_loss)
		return latent_loss, kl_train
	
	def get_place_holders(self):
		return self.encoder_ph
		
class network():
	def __init__(self):
		# network weights
		W_conv1 = weight_variable([8, 8, 4, 32])
		b_conv1 = bias_variable([32])
		
		W_conv2 = weight_variable([4, 4, 32, 64])
		b_conv2 = bias_variable([64])
		
		W_conv3 = weight_variable([3, 3, 64, 64])
		b_conv3 = bias_variable([64])
		
		W_fc1 = weight_variable([1600, 512])
		b_fc1 = bias_variable([512])
		
		W_fc2 = weight_variable([512, ACTIONS])
		b_fc2 = bias_variable([ACTIONS])
		
		# input layer
		self.input_ph = tf.placeholder("float", [None, 80, 80, 4])
		self.a_ph = tf.placeholder("float", [None, ACTIONS])
		self.y_ph = tf.placeholder("float", [None])
		
		# hidden layers
		h_conv1 = tf.nn.relu(conv2d(self.input_ph, W_conv1, 4) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)
		
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
		# h_pool2 = max_pool_2x2(h_conv2)
		
		h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
		# h_pool3 = max_pool_2x2(h_conv3)
		
		# h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
		h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
		
		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
		
		# readout layer
		self.readout = tf.matmul(h_fc1, W_fc2) + b_fc2
	
	def train_network(self):
		readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a_ph), reduction_indices=1)
		cost = tf.reduce_mean(tf.square(self.y_ph - readout_action))
		train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
		return train_step
	
	def get_place_holders(self):
		return self.input_ph, self.a_ph, self.y_ph
	
	def get_read_out(self):
		return self.readout