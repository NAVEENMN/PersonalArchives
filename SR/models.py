import tensorflow as tf
import numpy as np


#------ State encoder decoder -----#
class state_encoder_decoder():
    def __init__(self, name):
        self.name = name
        self.en_name = name+"_encode_"
        self.dec_name = name+"_decode_"
        self.en_latent_dim = 32
        self.dec_latent_dim = 32
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
            z = tf.layers.dense(fc1, self.en_latent_dim, activation=tf.nn.tanh, name=self.en_name+"fc2")
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

