from __future__ import print_function

import cv2
import imageio
import numpy as np
import tensorflow as tf

IMAGE_OUT_WIDTH = 500
IMAGE_OUT_HEIGHT = 500
IMAGE_OUT_CHANNEL = 3
LATENT_VEC_DIM = 100

# ======== Network
def weight_variable(name, shape):
	return tf.get_variable(name, shape, tf.float32, initializer=tf.random_normal_initializer(stddev=1.0))

def bias_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=1.0))

def feed_forward(X_, output_size, scope=None, stddev=1.0, with_bias=True):
	shape = X_.get_shape().as_list()
	with tf.variable_scope(scope or "FC"):
		W_F = weight_variable("weight", [shape[1], output_size])
		B_F = bias_variable("bias", [1, output_size])
		result = tf.matmul(X_, W_F) + (B_F * tf.ones([shape[0], 1], dtype=tf.float32))
		return result


class latent_space_to_image():
	def __init__(self, batch_size=1, scale=8.0):
		self.batch_size = batch_size
		self.image_width = IMAGE_OUT_HEIGHT
		self.image_height = IMAGE_OUT_HEIGHT
		self.scale = 0.8
		self.image_channel = IMAGE_OUT_CHANNEL
		self.z_dim = LATENT_VEC_DIM
		
		self.batch = tf.placeholder(tf.float32, [self.batch_size, \
												 self.image_width, \
												 self.image_height, \
												 self.image_channel])
		
		self.generator_x_placeholder = tf.placeholder(tf.float32, [self.batch_size, None, 1])
		self.generator_y_placeholder = tf.placeholder(tf.float32, [self.batch_size, None, 1])
		self.generator_r_placeholder = tf.placeholder(tf.float32, [self.batch_size, None, 1])
		# latent vector
		self.z_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
		
		self.number_of_pixels = self.image_width * self.image_height
		self.x_vec, self.y_vec, self.r_vec = self._coordinates(scale)
		
		self.G = self.generator()
		self.init()
	
	def init(self):
		init = tf.initialize_all_variables()
		self.sess = tf.Session()
		self.sess.run(init)
	
	def reinit(self):
		init = tf.initialize_variables(tf.trainable_variables())
		self.sess.run(init)
	
	def _coordinates(self, scale=1.0):
		'''
		pixel locations x,y,z(distance).
		'''
		x_dim = self.image_width
		y_dim = self.image_height
		x_range = scale * (np.arange(x_dim) - (x_dim - 1) / 2.0) / (x_dim - 1) / 0.5
		y_range = scale * (np.arange(y_dim) - (y_dim - 1) / 2.0) / (y_dim - 1) / 0.5
		x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
		y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
		r_mat = np.sqrt(x_mat * x_mat + y_mat * y_mat)
		x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, self.number_of_pixels, 1)
		y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, self.number_of_pixels, 1)
		r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, self.number_of_pixels, 1)
		return x_mat, y_mat, r_mat
	
	def generator(self, reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		
		z_scaled = tf.reshape(self.z_placeholder, [self.batch_size, 1, self.z_dim]) * \
				   tf.ones([self.number_of_pixels, 1], dtype=tf.float32) * self.scale
		z_unroll = tf.reshape(z_scaled, [self.batch_size * self.number_of_pixels, self.z_dim])
		x_unroll = tf.reshape(self.generator_x_placeholder, [self.batch_size * self.number_of_pixels, 1])
		y_unroll = tf.reshape(self.generator_y_placeholder, [self.batch_size * self.number_of_pixels, 1])
		r_unroll = tf.reshape(self.generator_r_placeholder, [self.batch_size * self.number_of_pixels, 1])
		
		out = feed_forward(z_unroll, self.z_dim, 'g_0_z') + \
			feed_forward(x_unroll, self.z_dim, 'g_0_x', with_bias=False) + \
			feed_forward(y_unroll, self.z_dim, 'g_0_y', with_bias=False) + \
			feed_forward(r_unroll, self.z_dim, 'g_0_r', with_bias=False)
		
		H = tf.nn.tanh(out)
		for i in range(3):
			H = tf.nn.tanh(feed_forward(H, self.z_dim, 'g_tanh_' + str(i)))
		output = tf.sigmoid(feed_forward(H, self.image_channel, 'g_final'))
		
		result = tf.reshape(output, [self.batch_size, IMAGE_OUT_HEIGHT, IMAGE_OUT_WIDTH, self.image_channel])
		return result
	
	def generate(self, z=None, scale=8.0):
		G = self.generator(reuse=True)
		x_vec, y_vec, r_vec = self._coordinates(scale)
		image = self.sess.run(G, feed_dict={self.z_placeholder: z, \
											self.generator_x_placeholder: x_vec, \
											self.generator_y_placeholder: y_vec, \
											self.generator_r_placeholder: r_vec})
		return image
	
	def close(self):
		self.sess.close()

def visulize_a_image(image):
	image = np.reshape(image, (IMAGE_OUT_WIDTH, IMAGE_OUT_HEIGHT, IMAGE_OUT_CHANNEL))
	cv2.imshow("image", image)
	cv2.waitKey(1)

def main():
	images = []
	Path = "D:\workspace\Projects\Latent_Vector_To_Image\images"
	z = np.random.uniform(-1.0, 1.0, size=(1, LATENT_VEC_DIM)).astype(np.float32)
	lt = latent_space_to_image()
	for x in range(0, 50):
		image = lt.generate(z)
		z = z + 0.009 * float(x)
		# z = 0.005*z+(0.008*float(np.sin(x*np.pi/4)))
		print(image.shape)
		images.append(np.reshape(image, (IMAGE_OUT_WIDTH, IMAGE_OUT_HEIGHT, IMAGE_OUT_CHANNEL)))
		visulize_a_image(image)
	imageio.mimsave(Path+'\out_color1.gif', images)
	lt.close()

if __name__ == "__main__":
	main()