import tensorflow as tf
import numpy as np
from PIL import Image
import glob

class NeuralNetwork:
	learning_rate = 0.5
	def __init__(self, network_desp):
		self.true_class  = [1, 0 , 0 , 0 , 0, 0 , 0 , 0 , 0 , 0]
		self.num_inputs = network_desp["num_inputs"]
		layers = network_desp["layers"]
		self.l_weights = []
		self.l_bias = []
		self.num_layers = len(layers)
		for x in range(0, len(layers)):
			layer = layers[x]
			self.l_weights.append(layer["weights"])
			self.l_bias.append(layer["bias"])

	def feed_forward(self, x, layer_id):
		prediction = None
		weights = self.l_weights[layer_id]
		bias = self.l_bias[layer_id]
		print "--FP layer"+str(layer_id)+" to layer", str(layer_id+1)
		#print x, weights, bias
		layer_in = tf.add(tf.matmul(x, weights), bias)
		layer_in = tf.nn.relu(layer_in) #activation
		#print layer_in
		if layer_id < self.num_layers-1:
			prediction = self.feed_forward(layer_in, layer_id+1)
		else:
			return layer_in
		return prediction

	def train(self, features):
		prediction = self.feed_forward(features, 0)
		error = tf.subtract(prediction, self.true_class)
		self.backprogate(error)
		return error
	
	def backprogate(self, errors):
		print errors
		
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
	layer1["bias"] = tf.Variable(tf.random_normal([1, 784]))
	layer2["bias"] = tf.Variable(tf.random_normal([1, 20]))
	layer3["bias"] = tf.Variable(tf.random_normal([1, 10]))
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
	p = nn.train(data)
	#initialize the variable
	init_op = tf.global_variables_initializer()
	#run the graph
	with tf.Session() as sess:
    		sess.run(init_op) #execute init_op
    		#print the random values that we sample
    		print (sess.run(p))



if __name__ == "__main__":
	main()
