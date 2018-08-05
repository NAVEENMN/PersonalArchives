import os
import numpy as np
import tensorflow as tf
from collections import deque
import random
from bGraph import *
from utils import *

SAVED = "saved_model/"
LOG = "log/"
DATA_PATH = "data/"
BOTTLE_NECK_GRAPH_PATH = "pb_models/classify_image_graph_def.pb"

BATCH_SIZE = 10
BNECK_SIZE = 2048
CLASS_SIZE = 6
SHAPE_IM = [None, BNECK_SIZE] # shape depends on the bottle neck layer
SHAPE_CLS = [None, CLASS_SIZE] # set the shape based on number of classes

class network():
    def __init__(self, sess):
        self.sess = sess
        self.tb_data = dict() # data for the tensorboad
        self.input_ph = tf.placeholder("float", SHAPE_IM, name="input_ph")
        self.target = tf.placeholder("float", SHAPE_CLS, name="target_ph")
        self.train_out = self.train_graph(reuse=False)
        self.test_out = self.train_graph(reuse=True)
        self.tvars = tf.trainable_variables()
        self.loss = tf.losses.mean_squared_error(labels=self.target,
                                                 predictions=self.train_out)
        #self.loss = tf.losses.softmax_cross_entropy(logits=self.train_out,
        #                                            onehot_labels=self.target)
        self.opt = tf.train.AdamOptimizer(0.001).minimize(self.loss,
                                                          var_list=self.tvars)

        # load all op nodes for tensorboard
        self.tb_data["train_loss"] = self.loss

        # op for tensorboard
        self.feed_tensorboard()
        self.merged = tf.summary.merge_all()

    def feed_tensorboard(self):
        tf.summary.scalar("train_loss", self.tb_data["train_loss"])

    def train_graph(self, reuse):
        with tf.variable_scope('FC_layers', reuse=reuse):
            fc1 = tf.layers.dense(self.input_ph, 800, activation=tf.nn.relu, name="FC_1")
            fc2 = tf.layers.dense(fc1, 400, activation=tf.nn.relu, name="FC_2")
            fc3 = tf.layers.dense(fc2, 100, activation=tf.nn.relu, name="FC_3")
            fc4 = tf.layers.dense(fc3, CLASS_SIZE, activation=tf.nn.softmax, name="FC_4")
        return fc4

    def get_loss(self, data_in, target_in):
        feed_dict = {self.input_ph: data_in, self.target: target_in}
        return self.sess.run(self.loss, feed_dict=feed_dict)

    def train_step(self, data_in, target_in, step):
        feed_dict = {self.input_ph: data_in, self.target: target_in}
        self.sess.run(self.opt, feed_dict=feed_dict)

    def get_summary(self, data_in, target_in):
        feed_dict = {self.input_ph: data_in, self.target: target_in}
        summary = self.sess.run(self.merged, feed_dict=feed_dict)
        return summary

'''
Run data through bGraph and collect 
outputs of bottle neck layer
'''
def pre_process_data(bGraph):
    return process_bGraph_images(bGraph, DATA_PATH)

def run(sess, tGraph, data, saver, writer):

    for step in range(400):
        minibatch = random.sample(data, BATCH_SIZE)
        input_images = np.asarray([x[0] for x in minibatch])
        targets = np.asarray([x[1] for x in minibatch])
        input_images = np.reshape(input_images, [BATCH_SIZE, BNECK_SIZE])
        input_targets = np.reshape(targets, [BATCH_SIZE, CLASS_SIZE])
        tGraph.train_step(input_images, input_targets, step)
        if step % 100 == 0:
            loss = tGraph.get_loss(input_images, input_targets)
            summary = tGraph.get_summary(input_images, input_targets)
            print("epoch {}, loss {}".format(step, loss))
            save_path = saver.save(sess, SAVED + "pretrained.ckpt", global_step=step)
            print("model saved to", save_path)
            writer.add_summary(summary, step)

def main():
    with tf.Session() as sess:
        bGraph = bottle_neck_graph(BOTTLE_NECK_GRAPH_PATH, sess)
        tGraph = network(sess)
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