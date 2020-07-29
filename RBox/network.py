import numpy as np
import random
import tensorflow as tf

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
IMAGE_CHANNEL = 3
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL
BATCH_SHAPE = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
BATCH_SIZE = 50

#------ State encoder decoder -----#
class state_encoder_decoder():
  def __init__(self, name, batch_size):
    self.name = name
    self.learning_rate = 0.001
    self.latent_dim = 512
    self.batch_size = batch_size
    self.enc_name = "StEnc_"
    self.dec_name = "StDec_"
  
  def encode(self, image, reuse=False):
    n = self.enc_name
    with tf.variable_scope('Encoder_ConvNet', reuse=reuse):
      input_st = image
      conv1 = tf.layers.conv2d(input_st,32,5, activation=tf.nn.relu, name=n+"conv1")
      conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name=n+"conv1_pool")
      conv2 = tf.layers.conv2d(conv1, 64, 5, activation=tf.nn.relu, name=n+"conv2")
      conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name=n+"conv2_pool")
      conv3 = tf.layers.conv2d(conv2, 128, 5, activation=tf.nn.relu, name=n+"conv3")
      conv3 = tf.layers.max_pooling2d(conv3, 2, 2, name=n+"conv3_pool")
      h_conv3_flat = tf.contrib.layers.flatten(conv3)
      fc1 = tf.layers.dense(h_conv3_flat, 1024, name=n+"E_fc1")
      z = tf.layers.dense(fc1, self.latent_dim, name=n+"E_fc2")
    return z
  def decode(self, latent,reuse=False):
    n = self.dec_name
    with tf.variable_scope('Decoder_ConvNet',reuse=reuse):
      fc1 = tf.layers.dense(latent, IMAGE_PIXELS*4, name=n+"fc")
      D_fc1 = tf.reshape(fc1, [-1, IMAGE_WIDTH*2, IMAGE_HEIGHT*2, IMAGE_CHANNEL])
      D_fc1 = tf.contrib.layers.batch_norm(D_fc1, epsilon=1e-5, scope='bn')
      D_fc1 = tf.nn.relu(D_fc1)
      dconv1 = tf.layers.conv2d(D_fc1, 64, 5, activation=tf.nn.relu, name=n+"conv1")
      dconv1 = tf.contrib.layers.batch_norm(dconv1, epsilon=1e-5, scope='bn1')
      dconv1 = tf.image.resize_images(dconv1, [IMAGE_WIDTH*2, IMAGE_HEIGHT*2])
      dconv2 = tf.layers.conv2d(dconv1, 32, 3, activation=tf.nn.relu, name=n+"conv2")
      dconv2 = tf.contrib.layers.batch_norm(dconv2, epsilon=1e-5, scope='bn2')
      dconv2 = tf.image.resize_images(dconv2, [IMAGE_WIDTH*1, IMAGE_HEIGHT*1])
      image  = tf.layers.conv2d(dconv2,3,1,activation=tf.nn.sigmoid, name=n+"conv3")
    return image
  def get_loss(self, source, target):
    with tf.name_scope('St_endecLoss'):
      batch_flatten = tf.reshape(target, [self.batch_size, -1])
      batch_reconstruct_flatten = tf.reshape(source, [self.batch_size, -1])
      loss1 = batch_flatten*tf.log(1e-10+batch_reconstruct_flatten)
      loss2 = (1 - batch_flatten) * tf.log(1e-10 + 1 - batch_reconstruct_flatten)
      loss = loss1+loss2
      reconstruction_loss = -tf.reduce_sum(loss)
    return tf.reduce_mean(reconstruction_loss)
  def train_step(self, loss):
    lr = self.learning_rate
    tvars = tf.trainable_variables()
    st_encoder_vars = [var for var in tvars if self.enc_name in var.name]
    st_decoder_vars = [var for var in tvars if self.dec_name in var.name]
    st_vars = st_encoder_vars+st_decoder_vars
    with tf.variable_scope(tf.get_variable_scope()) as scope:
      train_step = tf.train.AdamOptimizer(lr).minimize(loss,var_list = st_vars)
    return train_step

def tensorboard_summary(data):
  with tf.name_scope("summary/images"):
    source_image = data["source_image"]
    recon_image = data["recon_image"]
    images = tf.concat([source_image, recon_image], axis=1)
    images = images * 255.0
    tf.summary.image('phi', images, 5)
  with tf.name_scope("summary/losses"):
    tf.summary.scalar("phi_loss", data["phi_loss"])

def preprocess(img):
  img = np.reshape(img, [1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
  img = img / 255.0
  return img

def get_batch(D):
  minibatch = random.sample(D, BATCH_SIZE)
  shape_im = [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
  shape_ac = [BATCH_SIZE, 3]
  shape_rw = [BATCH_SIZE, 1]
  shape_sr = [BATCH_SIZE, 512]
  st_batch  = np.reshape([d[0] for d in minibatch], shape_im)
  at_batch  = np.reshape([d[1] for d in minibatch], shape_ac)
  rt1_batch = np.reshape([d[2] for d in minibatch], shape_rw)
  st1_batch = np.reshape([d[3] for d in minibatch], shape_im)
  return st_batch, at_batch, rt1_batch, st1_batch
