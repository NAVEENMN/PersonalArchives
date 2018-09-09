import os
import numpy as np
import tensorflow as tf
from collections import deque
import random
from bGraph import *
from utils import *

SAVED = "saved_model/"
LOG = "tensor_board/"
DATA_PATH = "data/"
BOTTLE_NECK_GRAPH_PATH = "pb_models/classify_image_graph_def.pb"

BATCH_SIZE = 30
BNECK_SIZE = 2048
CLASS_SIZE = 4
SHAPE_IM = [None, BNECK_SIZE] # shape depends on the bottle neck layer
SHAPE_CLS = [None, CLASS_SIZE] # set the shape based on number of classes
SHAPE_PICK = [None, 1]

class network():
    def __init__(self, sess):
        self.sess = sess
        self.tb_data = dict() # data for the tensorboad
        self.input_ph = tf.placeholder("float", SHAPE_IM, name="input_ph")
        self.target = tf.placeholder("float", SHAPE_CLS, name="target_ph")
        self.target_pick = tf.placeholder("float", SHAPE_PICK, name="target_pick_ph")
        
        self.class_pred, self.prob_pick = self.train_graph(reuse=False)
        self.test_class_pred, self.test_prob_pick = self.train_graph(reuse=True)
        self.tvars = tf.trainable_variables()
        self.tensors = tf.all_variables()
        
        self.class_pred_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.target,
                                                               logits=self.class_pred)
        self.prob_pick_loss = tf.losses.log_loss(labels=self.target_pick,
                                                 predictions=self.prob_pick)
        
        self.class_pred_accuracy = tf.contrib.metrics.accuracy(predictions=tf.argmax(self.class_pred, 1),
                                                               labels=tf.argmax(self.target, 1))
          
        self.total_loss = tf.reduce_mean(self.class_pred_loss)+tf.reduce_mean(self.prob_pick_loss)
        
        self.opt = tf.train.AdamOptimizer(0.0001).minimize(self.total_loss,
                                                          var_list=self.tvars)

        # load all op nodes for tensorboard
        #self.tb_data["source_images"] = self.input_ph
        self.tb_data["class_pred_loss"] = self.class_pred_loss
        self.tb_data["prob_pick_loss"] = self.prob_pick_loss
        self.tb_data["total_loss"] = self.total_loss
        self.tb_data["class_pred_acc"] = self.class_pred_accuracy
        
        # op for tensorboard
        self.feed_tensorboard()
        self.merged = tf.summary.merge_all()

    def feed_tensorboard(self):
        with tf.name_scope("summary/losses"):
            tf.summary.scalar("class_pred_loss", self.tb_data["class_pred_loss"])
            tf.summary.scalar("prob_pick_loss", self.tb_data["prob_pick_loss"])
            tf.summary.scalar("total_loss", self.tb_data["total_loss"])

        with tf.name_scope("summary/metrics"):
            tf.summary.scalar("class_pred_acc", self.tb_data["class_pred_acc"])
        
    def train_graph(self, reuse):
        with tf.variable_scope('FC_layers', reuse=reuse):
            fc1 = tf.layers.dense(self.input_ph, 1200, activation=tf.nn.relu, name="FC_1")
            fc6 = tf.layers.dense(self.input_ph, 800, activation=tf.nn.relu, name="FC_6")
            
            fc2 = tf.layers.dense(fc1, 800, activation=tf.nn.tanh, name="FC_2")
            fc3 = tf.layers.dense(fc2, 500, activation=tf.nn.tanh, name="FC_3")
            
            class_pred = tf.layers.dense(fc3, CLASS_SIZE, activation=tf.nn.softmax, name="pred_out")
            prob_pick = tf.layers.dense(fc6, 1, activation=tf.nn.sigmoid, name="pick_out")
        return class_pred, prob_pick

    def print_graph(self, graph):
        print("tGraph layers")
        tensors = []
        with tf.Session(graph=graph) as _sess:
            op = _sess.graph.get_operations()
            tensors = [m.values() for m in op]
            for tensor in tensors:
                print(tensor)

    def get_loss(self, data_in, target_in, probpick_in):
        feed_dict = {self.input_ph: data_in,
                     self.target: target_in,
                     self.target_pick: probpick_in}
        return self.sess.run(self.total_loss, feed_dict=feed_dict)
    
    def test_out(self, data_in, target_in, probpick_in):
        feed_dict = {self.input_ph: data_in,
                     self.target: target_in,
                     self.target_pick: probpick_in}
        res = self.sess.run([self.target, self.class_pred, self.target_pick, self.prob_pick], feed_dict=feed_dict)
        print("target ", res[0])
        print("class_pred ", res[1])
        print("target_pick", res[2])
        print("prob_pick", res[3])
        
    def train_step(self, data_in, target_in, probpick_in, step):
        feed_dict = {self.input_ph: data_in,
                     self.target: target_in,
                     self.target_pick: probpick_in}
        self.sess.run(self.opt, feed_dict=feed_dict)

    def get_summary(self, data_in, target_in, probpick_in):
        feed_dict = {self.input_ph: data_in,
                     self.target: target_in,
                     self.target_pick: probpick_in}
        summary = self.sess.run(self.merged, feed_dict=feed_dict)
        return summary

'''
Run data through bGraph and collect 
outputs of bottle neck layer
'''
def pre_process_data(bGraph):
    return process_bGraph_images(bGraph, DATA_PATH)

def run(sess, tGraph, data, saver, writer):

    for step in range(100000):
        minibatch = random.sample(data, BATCH_SIZE)
        input_images = np.asarray([x[0] for x in minibatch])
        targets = np.asarray([x[1] for x in minibatch])
        prob_pick =  np.asarray([x[2] for x in minibatch])
        input_images = np.reshape(input_images, [BATCH_SIZE, BNECK_SIZE])
        input_targets = np.reshape(targets, [BATCH_SIZE, CLASS_SIZE])
        input_probpick = np.reshape(prob_pick, [BATCH_SIZE, 1])
        #tGraph.test_out(input_images, input_targets, input_probpick)
        tGraph.train_step(input_images, input_targets, input_probpick, step)
        if step % 10000 == 0:
            loss = tGraph.get_loss(input_images, input_targets, input_probpick)
            print("epoch {}, loss {}".format(step, loss))
            summary = tGraph.get_summary(input_images, input_targets, input_probpick)
            writer.add_summary(summary, step)
            save_path = saver.save(sess, SAVED + "pretrained.ckpt", global_step=step)
            print("model saved to", save_path)
            

def main():
    with tf.Session() as sess:
        bGraph = bottle_neck_graph(BOTTLE_NECK_GRAPH_PATH, sess)
        tGraph = network(sess)
        tGraph.print_graph(sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(LOG, sess.graph)

        # load pre trained sub graph
        checkpoint = tf.train.get_checkpoint_state(SAVED)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        data = pre_process_data(bGraph)
        run(sess, tGraph, data, saver, writer)

if __name__ == "__main__":
    main()