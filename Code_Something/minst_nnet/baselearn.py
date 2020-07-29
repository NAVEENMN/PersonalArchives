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
num_of_neurons = [5, 3, 2, 2]

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
	def __init__(self, layer_id, num_of_neurons, num_of_neurons_next_layer, in_data):
		self.layer_name = "layer_"+str(layer_id)
		self.layer_id = layer_id
		wname = self.layer_name+"_weights"
		wnewname = self.layer_name+"_new_weights"
		bname = self.layer_name+"_bias"
		nname = self.layer_name+"_nets"
		aname = self.layer_name+"_activations"
		gnname = self.layer_name+"_gradient_wrt_nets"
		gnacts = self.layer_name+"_gradient_wrt_acts"
		self.weights = tf.Variable(tf.ones([num_of_neurons, num_of_neurons_next_layer]), name=wname)
		self.new_weights = tf.Variable(tf.ones([num_of_neurons, num_of_neurons_next_layer]), name=wnewname)
		self.bias = tf.Variable(tf.ones([1, num_of_neurons_next_layer]), name=bname)
		self.nets = tf.Variable(tf.zeros([1, num_of_neurons]), name=nname)
		if in_data == None:
			self.acts = tf.Variable(tf.zeros([1, num_of_neurons]), name=aname)
		else:
			self.acts = in_data	
		self.grad_nets =  tf.Variable(tf.zeros([1, num_of_neurons_next_layer]), name=gnname)
		self.grad_acts =  tf.Variable(tf.zeros([1, num_of_neurons_next_layer]), name=gnacts)

	def get_data(self, key):
		data = None
		if key == "weights":
			data = self.weights
		if key == "new_weights":
			data = self.new_weights
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
		if key == "layer_g_acts":
			data = self.grad_acts
		if key == "layer_g_nets":
			data = self.grad_nets
		return data

	def set_data(self, key, data):
		pv(key, data)
		if key == "weights":
			self.weights = data
		if key == "new_weights":
			self.new_weights = data
		if key == "bias":
			self.bias = data
		if key == "layer_id":
			self.layer_id = data
		if key == "layer_name":
			self.layer_name = data
		if key == "layer_nets":
			self.nets = data
		if key == "layer_acts":
			self.acts = data
		if key == "layer_g_acts":
			self.grad_acts = data
		if key == "layer_g_nets":
			self.grad_nets = data

class NeuralNetwork:
	def __init__(self, network_desp):
		self.true_class = [[1.0, 0.0]]
		self.num_inputs = network_desp["num_inputs"]
		layers_dp = network_desp["layers"]
		data = network_desp["input"]
		self.num_layers = num_of_layers
		self.layers = list()
		for layer_id in range(0, num_of_layers):
			print layer_id
			ldata = layers_dp[layer_id]
			num_of_neurons  = ldata["num_of_neurons"]
			num_of_neurons_next_layer = ldata["num_of_neurons_next_layer"]
			if layer_id == 0:
				layer = layers(layer_id, num_of_neurons, num_of_neurons_next_layer, data)
			else:
				layer = layers(layer_id, num_of_neurons, num_of_neurons_next_layer, None)
			self.layers.append(layer)
	def l2_loss_sum(self, preds, true_classes):
		msr_error = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(preds, true_classes))))
		return tf.div(msr_error, 2.0)
	def sigmoid(self, x):
		return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))
	def feed_forward(self, x, layer_id):
		tc = tf.Variable(self.true_class, name="tc")
		prt("---> Feed Forward")
		for layer_id in range(0, num_of_layers-1):
			lyr = self.layers[layer_id]
			nxtlyr = self.layers[layer_id+1]
			layer_name =  lyr.get_data("layer_name")
			weights = lyr.get_data("weights")
			bias = lyr.get_data("bias")
			nets = lyr.get_data("layer_nets")
			acts = lyr.get_data("layer_acts")
			with tf.name_scope(layer_name):
				prt("--> "+layer_name)
				pv("in(x): ", x)
				pv("weights:", weights)
				pv("bias", bias)
				layer_net = tf.add(tf.matmul(x, weights), bias)
				nxtlyr.set_data("layer_nets", layer_net)
				x = self.sigmoid(layer_net)
				nxtlyr.set_data("layer_acts", x)
				pv("bias", bias)
				pv("netout", layer_net)
				pv("out", x)
				pv("acts", lyr.get_data("layer_acts"))
				prt("----")
		prt("------")
		errors = tf.subtract(x, tc)
		neterror = self.l2_loss_sum(tc, x)
		return x, tc

	def train(self, features):
		for x in range(0, epoch):
			prt("=== pass"+str(x)+" ===")
			prediction, tc = self.feed_forward(features, 0)
			self.back_propagate(prediction, tc)
			self.update_weights()
			prt("=============")
			return prediction

	def back_propagate(self, prediction, actual):
		# net <- E{(xi*wi)+b} | for all i sigma
		# out <- sigmoid(net)
		# neterror <- 0.5*sqrt(E{sqr(p-e)}) | for all connectons
		# weight to update
		# error at wi = (dEt/dout)*(dout/dnet)*(dnet/dwi)
		#(dEt/dout)
		# W := W - (a) * (gradient(W))
		# error at wi = (dEt/dout)*(dout/dnet)*(dnet/dwi)
		# for last layer
		# (dEt/dout) => prediction - actual
		# for rest all full connected
		# (dEt/dout) => (Sigma((dEi/dout))) where Ei is all neurons of next layer
		prt("<--- Backprop")
		for x in range(num_of_layers-1, 0, -1):
			lyr = self.layers[x]
			plyr = self.layers[x-1]
			layer_name =  lyr.get_data("layer_name")
			nets = lyr.get_data("layer_nets")
			outs = lyr.get_data("layer_acts")
			pacts = plyr.get_data("layer_acts")
			pweights = plyr.get_data("weights")
			prt("layer data")
			pv("layer_acts", outs)
			pv("layer_nets", nets)
			prt("----------")
			with tf.name_scope(layer_name):
				prt("<-- "+layer_name)
				m = outs.get_shape()[0]
				n = outs.get_shape()[1]
				g_Et_outs = tf.ones([m , n], tf.float32)
				if x == num_of_layers-1: #last layer is independ of those nodes
					g_Et_outs = tf.subtract(prediction, actual)
				else:
					g_Et_nets = lyr.get_data("layer_g_nets")
					g_Et_nets = tf.transpose(g_Et_nets)
					weights = lyr.get_data("weights")
					pv("weights", weights)
					pv("g_Et_nets", g_Et_nets)
					g_Et_outs = tf.matmul(weights, g_Et_nets)	
					g_Et_outs = tf.transpose(g_Et_outs)
					#g_Et_outs = tf.subtract(prediction, actual)
				pv("g_Et_outs", g_Et_outs)
				#(dout/dnet)
				oi = tf.subtract(tf.ones([m , n], tf.float32), outs)
				g_out_nets = tf.multiply(outs, oi)
				pv("g_out_nets", g_out_nets)
				g_Et_nets = tf.multiply(g_Et_outs, g_out_nets)
				pv("g_Et_nets", g_Et_nets)
				plyr.set_data("layer_g_nets", g_Et_nets)
				#(dnet/dwi)
				pacts = tf.transpose(pacts)
				pv("pacts", pacts)
				lr = tf.constant(learning_rate)
				g_Et_weights = tf.matmul(pacts, g_Et_nets)
				pv("g_Et_weights",g_Et_weights)
				g_Et_weights = tf.multiply(lr, g_Et_weights) # with learning rate
				pv("g_Et_weights",g_Et_weights)
				new_weights = tf.subtract(pweights, g_Et_weights)
				lyr.set_data("new_weights", new_weights)
				pv("pre_w", pweights)
				pv("new_w",new_weights)
		prt("--------")
	def update_weights(self):
		prt("Update weights")
		for x in range(num_of_layers-1, 0, -1):
			print x
			lyr = self.layers[x]
			lyr.set_data("weights", lyr.get_data("new_weights"))
	
def get_network_descpription(data):
	dp = dict()
	dp["num_inputs"] = num_of_features
	dp["num_outputs"] = num_of_classes
	dp["input"] = data
	LAYERS = list()
	for x in range(0, num_of_layers):
		l = dict()
		l["num_of_neurons"] = num_of_neurons[x]
		l["num_of_neurons_next_layer"] = num_of_neurons[x+1]
		LAYERS.append(l)
	dp["layers"] = LAYERS
	return dp

def main():
	data = tf.Variable([[1.0, 1.0, 1.0, 1.0, 1.0]], name="data")
	#setup network
	with tf.name_scope("setup_network"):
		network = get_network_descpription(data)
		print network
		nn = NeuralNetwork(network)
	pred = nn.train(data)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter("logs/", sess.graph)
		print "T", sess.run(data)
		print "P", sess.run(pred)

if __name__ == "__main__":
	main()
