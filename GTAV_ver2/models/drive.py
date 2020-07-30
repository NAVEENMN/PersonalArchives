import tensorflow as tf
import numpy as np

IMAGE_HEIGHT      = 160
IMAGE_WIDTH       = 320
IMAGE_CHANNELS    = 3
IMAGE_SHAPE = [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]

def weight_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.01)
	return tf.Variable(initial, name=name)
def bias_variable(shape, name):
	initial = tf.constant(0.01, shape=shape)
	return tf.Variable(initial, name=name)
def conv2d(x, W, stride):
	return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
def weights_bias(shape, name):
	temp_w = weight_variable(shape, name+"_W")
	temp_b = bias_variable([shape[len(shape)-1]], name+"_B")
	return temp_w, temp_b
	
class drive():
	def __init__(self):
		self.data = None
		self.net = self.build_network()
		
	def build_network(self):
		# input layer
		self.drive_input_ph = tf.placeholder("float", [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
		self.drive_target_ph = tf.placeholder("float", [None, 4])
		# network weights
		with tf.name_scope('fly_drive_conv'):
			Drive_conv1_W, Drive_b_conv1_B = weights_bias([5, 5, 3, 24], "Drive_conv1")
			Drive_conv2_W, Drive_b_conv2_B = weights_bias([5, 5, 24, 36], "Drive_conv2")
			Drive_conv3_W, Drive_b_conv3_B = weights_bias([5, 5, 36, 48], "Drive_conv3")
			Drive_conv4_W, Drive_b_conv4_B = weights_bias([3, 3, 48, 64], "Drive_conv4")
			Drive_conv5_W, Drive_b_conv5_B = weights_bias([3, 3, 64, 64], "Drive_conv5")
			Drive_fc1_W, Drive_fc1_B = weights_bias([64, 60], "Drive_fc1")
			Drive_fc2_W, Drive_fc2_B = weights_bias([60, 50], "Drive_fc2")
			Drive_fc3_W, Drive_fc3_B = weights_bias([50, 10], "Drive_fc3")
			Drive_fc4_W, Drive_fc4_B = weights_bias([10, 4], "Drive_fc4")
		#hidden layers
			Drive_h_conv1 = self.feed_conv(self.drive_input_ph, Drive_conv1_W, Drive_b_conv1_B, 2)
			Drive_h_conv2 = self.feed_conv(Drive_h_conv1, Drive_conv2_W, Drive_b_conv2_B, 2)
			Drive_h_conv3 = self.feed_conv(Drive_h_conv2, Drive_conv3_W, Drive_b_conv3_B, 2)
			Drive_h_conv4 = self.feed_conv(Drive_h_conv3, Drive_conv4_W, Drive_b_conv4_B, 2)
			Drive_h_conv5 = self.feed_conv(Drive_h_conv4, Drive_conv5_W, Drive_b_conv5_B, 2)
			Drive_h_conv5_flat = tf.reshape(Drive_h_conv5, [-1, 64])
			Drive_h_fc1 = tf.nn.relu(tf.matmul(Drive_h_conv5_flat, Drive_fc1_W) + Drive_fc1_B)
			Drive_h_fc2 = tf.nn.relu(tf.matmul(Drive_h_fc1, Drive_fc2_W) + Drive_fc2_B)
			Drive_h_fc3 = tf.nn.relu(tf.matmul(Drive_h_fc2, Drive_fc3_W) + Drive_fc3_B)
			self.drive_readout = tf.matmul(Drive_h_fc3, Drive_fc4_W) + Drive_fc4_B
			
			self.cost = tf.reduce_mean(tf.squared_difference(self.drive_target_ph, self.drive_readout))
			self.test_loss = tf.reduce_mean(tf.squared_difference(self.drive_target_ph, self.drive_readout))
			tf.summary.scalar('Drive_Train_loss', self.cost)
			tf.summary.scalar('Drive_Test_loss', self.test_loss)
			self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
			
	def feed_conv(self, input, weight, bias, stride):
		temp = tf.nn.relu(conv2d(input, weight, stride) + bias)
		return max_pool_2x2(temp)
	
	def feed_forward(self, input):
		input_ph = self.drive_input_ph
		x = np.reshape(input,IMAGE_SHAPE)
		return self.drive_readout.eval(feed_dict={input_ph: x})
	
	def train(self, sess, data, batch_id):
		[X_, Y_] = data[0] #Training set
		[x_, y_] = data[1] #Test set
		feed_dict = {self.drive_input_ph: X_, self.drive_target_ph: Y_}
		train_loss, _ = sess.run([self.cost, self.train_step], feed_dict)
		feed_dict = {self.drive_input_ph: x_, self.drive_target_ph: y_}
		test_loss = sess.run(self.test_loss, feed_dict)
		print(batch_id, train_loss, test_loss)
		return feed_dict
		
	
		
		
		
		