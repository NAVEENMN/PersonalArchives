import tensorflow as tf
from defs import *
import numpy as np

#------ State encoder decoder -----#
class state_encoder_decoder():
    def __init__(self, name):
        self.name = name
        self.en_name = name+"_encode_"
        self.dec_name = name+"_decode_"
        self.opt = tf.train.AdamOptimizer(0.0001)
        
    def encode(self, state, reuse):
        with tf.variable_scope(self.en_name, reuse=reuse):
            conv1 = tf.layers.conv2d(state, 32, 5, activation=tf.nn.relu, name=self.en_name+"conv1")
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name=self.en_name+"conv1_pool")
            conv2 = tf.layers.conv2d(conv1, 64, 5, activation=tf.nn.relu, name=self.en_name+"conv2")
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name=self.en_name+"conv2_pool")
            conv3 = tf.layers.conv2d(conv2, 128, 3, activation=tf.nn.relu, name=self.en_name+"conv3")
            conv3 = tf.layers.max_pooling2d(conv3, 2, 2, name=self.en_name+"conv3_pool")
            h_conv3_flat = tf.contrib.layers.flatten(conv3)
            fc1 = tf.layers.dense(h_conv3_flat, 1024, activation=tf.nn.tanh, name=self.en_name+"fc1")
            z = tf.layers.dense(fc1, LATENT_DIM, activation=tf.nn.tanh, name=self.en_name+"fc2")
        return z
    
    def decode(self, state_latent, reuse):
        with tf.variable_scope(self.dec_name, reuse=reuse):
            fc1 = tf.layers.dense(state_latent, IMAGE_PIXELS * 4, name=self.dec_name+"fc")
            D_fc1 = tf.reshape(fc1, [-1, IMAGE_HEIGHT * 2, IMAGE_WIDTH * 2, IMAGE_CHANNEL])
            D_fc1 = tf.contrib.layers.batch_norm(D_fc1, epsilon=1e-5)
            D_fc1 = tf.nn.tanh(D_fc1)
            dconv1 = tf.layers.conv2d(D_fc1, 64, 5, activation=tf.nn.tanh, name=self.dec_name+"conv1")
            dconv1 = tf.contrib.layers.batch_norm(dconv1, epsilon=1e-5)
            dconv1 = tf.image.resize_images(dconv1, [IMAGE_HEIGHT * 2, IMAGE_WIDTH * 2])
            dconv2 = tf.layers.conv2d(dconv1, 32, 3, activation=tf.nn.tanh, name=self.dec_name+"conv2")
            dconv2 = tf.contrib.layers.batch_norm(dconv2, epsilon=1e-5)
            dconv2 = tf.image.resize_images(dconv2, [IMAGE_HEIGHT * 1, IMAGE_WIDTH * 1])
            image = tf.layers.conv2d(dconv2, 3, 1, activation=tf.nn.sigmoid, name=self.dec_name+"conv3")
        return image
    
    def get_loss(self, source, target):
        batch_flatten = tf.reshape(target, [BATCH_SIZE, -1])
        batch_reconstruct_flatten = tf.reshape(source, [BATCH_SIZE, -1]) 
        loss1 = batch_flatten * tf.log(1e-10 + batch_reconstruct_flatten)
        loss2 = (1 - batch_flatten) * tf.log(1e-10 + 1 - batch_reconstruct_flatten)
        loss = loss1 + loss2
        reconstruction_loss = -tf.reduce_sum(loss)
        loss = tf.reduce_mean(reconstruction_loss)
        return loss
    
    def train_step(self, loss):
        tvars = tf.trainable_variables()
        phi_vars = [var for var in tvars if self.name in var.name]
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            train_ = self.opt.minimize(loss, var_list=phi_vars)
        return train_
        
    def get_vars(self):
        tvars = tf.trainable_variables()
        vars = [var for var in tvars if self.name in var.name]
        return vars


#------ Policy -----#
class policy():
    def __init__(self, name):
        self.name = name
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-4)

    def sample(self, state_latent, reuse=False):
        n = self.name
        with tf.variable_scope(self.name, reuse=reuse):
            ex_fc1 = tf.layers.dense(state_latent, 32, activation=tf.nn.relu, name=n+"Ex_fc1")
            ex_fc2 = tf.layers.dense(ex_fc1, 20, activation=tf.nn.tanh, name=n+"Ex_fc2")
            ex_out = tf.layers.dense(ex_fc2, 3, activation=tf.nn.sigmoid, name=n+"Ex_out")

            in_fc1 = tf.layers.dense(state_latent, 32, activation=tf.nn.relu, name=n+"In_fc1")
            in_fc2 = tf.layers.dense(in_fc1, 20, activation=tf.nn.tanh, name=n+"In_fc2")
            in_out = tf.layers.dense(in_fc2, 3, activation=tf.nn.sigmoid, name=n+"In_out")
            in_out = tf.multiply(-1.0, in_out)
        return ex_out, in_out

    def get_vars(self):
        tvars = tf.trainable_variables()
        vars = [var for var in tvars if self.name in var.name]
        return vars

#------ Tensorboard ----#
def tensorboard_summary(data):
    st = data["source_image"]
    st_recon = data["reconstructed_image"]
    #sr_feature_recon = data["sr_feature"]
    with tf.name_scope("summary/images"):
        source_image = st * 255.0 
        recon_image = st_recon * 255.0
        #sr_feature_recon = sr_feature_recon * 255.0
        image_to_tb = tf.concat([source_image, recon_image], axis=1)
        #image_to_tb = tf.concat([image_to_tb, sr_feature_recon], axis=1)
        tf.summary.image('src', image_to_tb, 5)

    with tf.name_scope("summary/losses"):
        tf.summary.scalar("StRecon_loss", data["reconstruction_loss"])
        #tf.summary.scalar("SR_loss", data["sr_loss"])
        
