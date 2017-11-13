import tensorflow as tf
from PIL import Image
import numpy as np
import glob
import random
from sklearn.utils import shuffle

# Image Parameters
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNEL = 1
#IMAGE_SHAPE = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL]
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL
IMAGE_SHAPE = [1, IMAGE_PIXELS]

LATENT_SIZE = 10

def weight_variable(name, shape):
  return tf.get_variable(name, 
      shape, initializer=tf.truncated_normal_initializer(stddev=0.02))

def bias_variable(name, shape):
  return tf.get_variable(name, 
      shape, initializer=tf.constant_initializer(0))

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

def avg_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def load_data():
  data = list()
  for i in range(0, 4):
    path = "data/"+str(i)+"/*.jpg"
    print(path)
    for filename in glob.glob(path):
      im = np.array(Image.open(filename))
      im = np.reshape(im, IMAGE_SHAPE)
      im = im / 255
      data.append(im)
  random.shuffle(data)
  return data

class encoder():
  def __init__(self):
    self.encoder_placeholder = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
    # Encoder Layer 1
    W_efc1 = weight_variable('e_wfc1', [784, 500])
    b_fc1 = bias_variable('e_fcb1', [500])
    # Encoder Layer 2
    W_efc2 = weight_variable('e_wfc2', [500, 50])
    b_fc2 = bias_variable('e_fcb2', [50])
    # Encoder Layer 3
    W_efc3 = weight_variable('e_wfc3', [50, LATENT_SIZE])
    b_fc3 = bias_variable('e_fcb3', [LATENT_SIZE])

    h_fc1 = tf.matmul(self.encoder_placeholder, W_efc1) + b_fc1
    h_fc1 = tf.sigmoid(h_fc1)
    h_fc2 = tf.sigmoid(tf.matmul(h_fc1, W_efc2) + b_fc2)
    h_fc3 = tf.sigmoid(tf.matmul(h_fc2, W_efc3) + b_fc3)

    self.e_readout = h_fc3

  def feed_forward(self, x):
    input_ph = self.encoder_placeholder
    x = np.reshape(x, [1,784])
    return self.e_readout.eval(feed_dict={input_ph : x})[0]

  def get_encoder_readout(self):
    return self.e_readout
  def get_input_ph(self):
    return self.encoder_placeholder

class decoder():
  def __init__(self, code):

    # Decoder Layer 1
    self.W_dfc1 = weight_variable('d_wfc1', [LATENT_SIZE, 50])
    self.b_dfc1 = bias_variable('d_fcb1', [50])
    # Decoder Layer 2
    self.W_dfc2 = weight_variable('d_wfc2', [50, 500])
    self.b_dfc2 = bias_variable('d_fcb2', [500])
    # Decoder Layer 3
    self.W_dfc3 = weight_variable('d_wfc3', [500, 784])
    self.b_dfc3 = bias_variable('d_fcb3', [784])

    h_dfc1 = tf.matmul(code, self.W_dfc1) + self.b_dfc1
    h_dfc1 = tf.sigmoid(h_dfc1)
    h_dfc2 = tf.sigmoid(tf.matmul(h_dfc1, self.W_dfc2) + self.b_dfc2)
    h_dfc3 = tf.sigmoid(tf.matmul(h_dfc2, self.W_dfc3) + self.b_dfc3)
    self.d_readout = h_dfc3

  '''
  def feed_forward(self, x):
    input_ph = self.decoder_placeholder
    x = np.reshape(x, [1,2])
    return self.d_readout.eval(feed_dict={input_ph : x})
  '''

  def get_decoder_readout(self):
    return self.d_readout

class network():
  def __init__(self, data):
    self.data = data
    self.sess = sess = tf.InteractiveSession()
    self.encoder = encoder()
    self.e_ph = self.encoder.get_input_ph()
    self.e_readout = self.encoder.get_encoder_readout()
    self.decoder = decoder(self.e_readout)
    self.d_readout = self.decoder.get_decoder_readout()

    self.cost = tf.reduce_mean(tf.squared_difference(self.e_ph, self.d_readout))
    self.train_step = tf.train.AdamOptimizer(0.1).minimize(self.cost)

    self.sess.run(tf.global_variables_initializer())
  
  def encode(self, x):
    return self.encoder.feed_forward(x)
  def decode(self, code):
    return self.decoder.feed_forward(code)

  def train(self):
    data = self.data
    batch_size = 100

    for ep in range(10000):
      batch = shuffle(data, random_state=random.randint(1, 10))
      image_batch = batch[:batch_size]
      image_batch = np.reshape(image_batch, [-1, IMAGE_PIXELS])
      feed_dict = {self.e_ph : image_batch}
      loss, _ = self.sess.run([self.cost, self.train_step], feed_dict)
      if ep % 100 == 0:
        print(ep, loss)

def main():
  data = load_data()
  net = network(data)
  net.train()

if __name__ == "__main__":
  main()
