import tensorflow as tf
import numpy as np
from PIL import Image
import glob


class NeuralNetwork:
	learning_rate = 0.5
	def __init__(self, network_desp):
		self.num_inputs = network_desp["num_inputs"]
		layers = network_desp["layers"]
		layer1 = layers[0]
		layer2 = layers[1]
		out = layers[2]
		self.l1_weights = layer1["weights"]
		self.l2_weights = layer2["weights"]
		self.out_weights = out["weights"]
		self.l1_bias = layer1["bias"]
		self.l2_bias = layer2["bias"]
		self.out_bias = out["bias"]

	def feed_forward(self, x):
		layer_1 = tf.add(tf.matmul(x, self.l1_weights), self.l1_bias)
		layer_1 = tf.nn.relu(layer_1) #activation
		print "ok"
		layer_2 = tf.add(tf.matmul(layer_1, self.l2_weights), self.l2_weights)
		layer_2 = tf.nn.relu(layer_2)
		out_layer = tf.add(tf.matmul(layer_2, self.out_weights), self.out_bias)
		return out_layer

	def train(self, features):
		prediction = self.feed_forward(features)
		print prediction
		
#load image pixels
#we will take images
def load_images():
	IMGS = list()
	image = None
	IMG = []
	IMG = np.empty((0,784, 1), dtype="uint8")
	#IMG = np.zeros((2,784), dtype=np.float)
	path = "/Users/naveenmysore/Documents/QL/data/0/*.jpg"
	files = glob.glob(path)
	test_images = [files[0], files[1]]
	filename_queue = tf.train.string_input_producer(test_images)
	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)
	images = tf.image.decode_jpeg(image_file)
	images = tf.image.rgb_to_grayscale(images)
	init_op = tf.global_variables_initializer()
	with tf.Session() as sess:
  		sess.run(init_op)
 		# Start populating the filename queue.
  		coord = tf.train.Coordinator()
  		threads = tf.train.start_queue_runners(coord=coord)
  		for i in range(2):
			image = images.eval()
			img = tf.reshape(image, [1, 784, 1])
			img = sess.run(img)
			IMGS.append(img)
			IMG = np.append(IMG, np.array(img), axis=0)
 		coord.request_stop()
 		coord.join(threads)
	return IMGS

def get_network_descpription():
	dp = dict()
	dp["num_inputs"] = 784
	dp["num_outputs"] = 10
	layer1 = dict()
	layer2 = dict()
	layer3 = dict() #ouput layer
	layer1["num_neurons"] = 784
	layer2["num_neurons"] = 20
	layer3["num_neurons"] = 10
	layer1["weights"] = tf.Variable(tf.random_normal([784, 784]))
	layer2["weights"] = tf.Variable(tf.random_normal([784, 20]))
	layer3["weights"] = tf.Variable(tf.random_normal([20, 10]))
	layer1["bias"] = tf.Variable(tf.random_normal([784]))
	layer2["bias"] = tf.Variable(tf.random_normal([20]))
	layer3["bias"] = tf.Variable(tf.random_normal([10]))
	layers = [layer1, layer2, layer3]
	dp["layers"] = layers
	return dp

def main():
	image = load_images()
	'''
	a = tf.placeholder(tf.float32, shape=(1, 784, 1))
	b = tf.placeholder(tf.float32, shape=(1, 784, 1))
	adder_node = 255 - tf.subtract(a, b)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		c = sess.run(adder_node, {a: image[0], b:image[1]})
		c = tf.reshape(c, [28, 28, 1])
		c = tf.cast(c, tf.uint8)
		resized_encoded = tf.image.encode_jpeg(c,name="save_me")
		f = open("/Users/naveenmysore/Documents/QL/data/3.jpg", "wb+")
  		f.write(resized_encoded.eval())
 		f.close()
	'''
	
	network = get_network_descpription()
	nn = NeuralNetwork(network)
	ima = np.transpose(image[0][0])
	data =  tf.cast(ima, tf.float32)
	nn.train(data)

if __name__ == "__main__":
	main()
