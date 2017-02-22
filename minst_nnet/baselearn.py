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
num_of_layers = 3 #(hidden:2, output:1)
num_of_features = 5 # 28x28pixel image
num_of_classes = 2
num_of_neurons = [num_of_features, num_of_features, 5, num_of_classes, num_of_classes]

path = os.getcwd()
verbose = True

def pv(name, ts):
	if verbose:
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		print name, sess.run(ts)
		sess.close()
def prt(text):
	if verbose:
		print text

class layers:
	def __init__(self, layer_id, input_connections, output_connections, num_of_neurons):
		self.layer_name = "layer_"+str(layer_id)
		self.layer_id = layer_id
		wname = self.layer_name+"_weights"
		bname = self.layer_name+"_bias"
		nname = self.layer_name+"_nets"
		aname = self.layer_name+"_activations"
		self.weights = tf.Variable(tf.ones([input_connections, output_connections]), name=wname)
		self.bias = tf.Variable(tf.ones([1, output_connections]), name=bname)
		self.nets = tf.Variable(tf.zeros([1, output_connections]), name=nname)
		self.acts = tf.Variable(tf.zeros([1, output_connections]), name=aname)

	def get_data(self, key):
		data = None
		if key == "weights":
			data = self.weights
		if key == "bias":
			data = self.bias
		if key == "layer_id":
			data = self.layer_id
		if key == "layer_name":
			data = self.layer_name
		if key == "layer_nets":
			data = self.nets
		if key == "layer_acts":
			data = self.acts
		return data

	def set_data(self, key, data):
		pv(key, data)
		if key == "weights":
			tf.assign(self.weights, data)
		if key == "bias":
			tf.assign(self.bias, data)
		if key == "layer_id":
			tf.assign(self.layer_id, data)
		if key == "layer_name":
			tf.assign(self.layer_name, data)
		if key == "layer_nets":
			#tf.assign(self.nets, data)
			self.nets = data
		if key == "layer_acts":
			#tf.assign(self.acts, data)
			self.acts = data

class NeuralNetwork:
	def __init__(self, network_desp):
		self.true_class = [[1.0, 0.0]]
		self.num_inputs = network_desp["num_inputs"]
		layers_dp = network_desp["layers"]
		self.num_layers = num_of_layers
		self.layers = list()
		for layer_id in range(0, num_of_layers):
			ldata = layers_dp[layer_id]
			ip_conn = ldata["ip_connections"]
			out_conn = ldata["out_connections"]
			num_of_neurons  = ldata["num_of_neurons"]
			layer = layers(layer_id, ip_conn, out_conn, num_of_neurons)
			self.layers.append(layer)
	def l2_loss_sum(self, preds, true_classes):
		msr_error = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(preds, true_classes))))
		return tf.div(msr_error, 2.0)
	def sigmoid(self, x):
		return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))
	def feed_forward(self, x, layer_id):
		node_values = list() # [W*x+b]
		tc = tf.Variable(self.true_class, name="tc")
		prt("---> Feed Forward")
		for layer_id in range(0, num_of_layers):
			lyr = self.layers[layer_id]
			layer_name =  lyr.get_data("layer_name")
			weights = lyr.get_data("weights")
			bias = lyr.get_data("bias")
			nets = lyr.get_data("layer_nets")
			acts = lyr.get_data("layer_acts")
			with tf.name_scope(layer_name):
				prt("--> "+layer_name)
				pv("in(x): ", x)
				layer_net = tf.add(tf.matmul(x, weights), bias)
				lyr.set_data("layer_nets", layer_net)
				x = self.sigmoid(layer_net)
				lyr.set_data("layer_acts", x)
				node_values.append(x)
				pv("weights", weights)
				pv("bias", bias)
				pv("netout", layer_net)
				pv("out", x)
				
				pv("acts", lyr.get_data("layer_acts"))
				prt("----")
		prt("------")
		errors = tf.subtract(x, tc)
		neterror = self.l2_loss_sum(tc, x)
		return x, errors, neterror, node_values

	def train(self, features):
		for x in range(0, epoch):
			prt("=== pass"+str(x)+" ===")
			prediction, errors, neterror, node_net_values = self.feed_forward(features, 0)
			pred, error = self.back_propagate(prediction, errors, neterror, node_net_values)
			prt("=============")
		return pred, error

	def back_propagate(self, prediction, errors, neterror, node_net_values):
		# net <- E{(xi*wi)+b} | for all i sigma
		# out <- sigmoid(net)
		# neterror <- 0.5*sqrt(E{sqr(p-e)}) | for all connectons
		# weight to update
		# error at wi = (dEt/dout)*(dout/dnet)*(dnet/dwi)
		#(dEt/dout)
		prt("<--- Backprop")
		for layer_id in range(num_of_layers-1, 0, -1):
			lyr = self.layers[layer_id]
			layer_name =  lyr.get_data("layer_name")
			weights = lyr.get_data("weights")
			bias = lyr.get_data("bias")
			nets = lyr.get_data("layer_nets")
			acts = lyr.get_data("layer_acts")
			with tf.name_scope(layer_name):
				prt("<-- "+layer_name)
				pv("acts", acts)
				p1 = errors
				#(dout/dnet)
				m = errors.get_shape()[0]
				n = errors.get_shape()[1]
				oi = tf.subtract(tf.ones([m , n], tf.float32), prediction)
				p2 = tf.multiply(prediction, oi)
				#(dnet/dwi)
				p3 = node_net_values[layer_id]
				w = tf.multiply(tf.multiply(p1, p2), p3)
				pv("new_weights",w)
		prt("--------")
		return prediction, errors
	
def get_network_descpription():
	dp = dict()
	dp["num_inputs"] = num_of_features
	dp["num_outputs"] = num_of_classes
	LAYERS = list()
	for x in range(0, num_of_layers+1):
		l = dict()
		l["num_of_neurons"] = num_of_neurons[x]
		l["ip_connections"] = num_of_neurons[x]
		l["out_connections"] = num_of_neurons[x+1]
		LAYERS.append(l)
	dp["layers"] = LAYERS
	return dp

def main():
	data = tf.Variable([[1.0, 1.0, 1.0, 1.0, 1.0]], name="data")
	#setup network
	with tf.name_scope("setup_network"):
		network = get_network_descpription()
		nn = NeuralNetwork(network)
	pred, errors = nn.train(data)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		if (os.path.exists(path+"/load/model.ckpt.meta")):
			saver.restore(sess, path+"/load/model.ckpt")
		save_path = saver.save(sess, path+"/load/model.ckpt")
		writer = tf.summary.FileWriter("logs/", sess.graph)
		print "T", sess.run(data)
		print "P", sess.run(pred)
		print "e", sess.run(errors)

if __name__ == "__main__":
	main()
