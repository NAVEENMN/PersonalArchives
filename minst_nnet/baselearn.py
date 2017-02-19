#tensorboard --logdir=logs
import tensorflow
import tensorflow as tf
import numpy as np
from PIL import Image
import glob
import os.path

path = "/Users/naveenmysore/Documents/QL/"

#hyperparameters
epoch = 1
learning_rate = 0.5
num_of_layers = 3
num_of_features = 784 # 28x28pixel image
num_of_neurons_layer_1 = 784
num_of_neurons_layer_2 = 20
num_of_classes = 10

class NeuralNetwork:
	def __init__(self, network_desp):
		self.true_class  = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		self.num_inputs = network_desp["num_inputs"]
		layers = network_desp["layers"]
		self.l_weights = []
		self.l_bias = []
		self.num_layers = len(layers)
		for x in range(0, len(layers)):
			layer = layers[x]
			self.l_weights.append(layer["weights"])
			self.l_bias.append(layer["bias"])
	def sigmoid(self, x):
		return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))

	def feed_forward(self, x, layer_id):
		for t in range(0, 3):
			layer_name = "layer"+str(layer_id)
			with tf.name_scope(layer_name):
				weights = self.l_weights[layer_id]
				bias = self.l_bias[layer_id]
				layer1_in = tf.add(tf.matmul(x, weights), bias)
				x = self.sigmoid(layer1_in)
				layer_id +=1
				print x
		#error = tf.subtract(self.true_class, x)
		return x

	def feed_forward1(self, x, layer_id):
		layer_name = "feedfw_"+str(layer_id)
		with tf.name_scope(layer_name):
			#nl = tf.Variable(0.0)
			weights = self.l_weights[layer_id]
			bias = self.l_bias[layer_id]
			print "-- FP layer"+str(layer_id)+" to layer", str(layer_id+1)
			layer_in = tf.add(tf.matmul(x, weights), bias)
			layer_in = self.sigmoid(layer_in) #activation
			#op = tf.assign(nl, layer_in)
			if layer_id < self.num_layers-1:
				error, prediction = self.feed_forward(layer_in, layer_id+1)
			else:
				error = tf.subtract(layer_in, self.true_class)
				#print "error"
				#print error
				return error, layer_in
		return error, prediction

	def train(self, features):
		prediction = self.feed_forward(features, 0)
		#error = tf.subtract(prediction, self.true_class)
		#self.backprogate(error)
		return prediction
	
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
	print 
	print "setting up network"
	dp = dict()
	layer1 = dict()
	layer2 = dict()
	layer3 = dict()
	dp["num_inputs"] = num_of_features
	dp["num_outputs"] = num_of_classes
	layer1["num_neurons"] = num_of_neurons_layer_1
	layer2["num_neurons"] = num_of_neurons_layer_2
	layer3["num_neurons"] = num_of_classes
	with tf.name_scope("hidden_layer"):
		with tf.name_scope("weights"):
			all_vars = tf.get_collection('vars')
			for v in all_vars:
				v_ = sess.run(v)
				print v_
			layer1["weights"] = tf.Variable(tf.truncated_normal([num_of_features, num_of_neurons_layer_1]), name="layer1_weights")
			layer2["weights"] = tf.Variable(tf.truncated_normal([num_of_neurons_layer_1, num_of_neurons_layer_2]), name="layer2_weights")
			layer3["weights"] = tf.Variable(tf.truncated_normal([num_of_neurons_layer_2, num_of_classes]), name="outlayer_weights")
		tf.add_to_collection('vars', layer1["weights"])
		tf.add_to_collection('vars', layer2["weights"])
		tf.add_to_collection('vars', layer3["weights"])
		with tf.name_scope("bias"):
			layer1["bias"] = tf.Variable(tf.truncated_normal([1, num_of_neurons_layer_1]), name="layer1_bias")
			layer2["bias"] = tf.Variable(tf.truncated_normal([1, num_of_neurons_layer_2]), name="layer2_bias")
			layer3["bias"] = tf.Variable(tf.truncated_normal([1, num_of_classes]), name="outlayer_bias")
	layers = [layer1, layer2, layer3]
	dp["layers"] = layers
	print dp
	print 
	return dp

def viz_neurons():
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

def main():
	#setup network
	with tf.name_scope("setup_network"):
		network = get_network_descpription()
		nn = NeuralNetwork(network)
	#get image features
	with tf.name_scope("load_image"):
		image = load_images()
		image = image[0][0] #take first image (784x1)
		image = np.transpose(image)# to (1x784)
		image = tf.cast(image, tf.float32)
	#train network
	with tf.name_scope("train"):
		p = nn.train(image)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		if (os.path.exists(path+"model.ckpt.meta")):
			new_saver = tf.train.import_meta_graph(path+"model.ckpt.meta")
			new_saver.restore(sess, tf.train.latest_checkpoint('./'))

		writer = tf.summary.FileWriter("logs/", sess.graph)
		save_path = saver.save(sess, path+"model.ckpt")
		
		print sess.run(p)

if __name__ == "__main__":
	main()
