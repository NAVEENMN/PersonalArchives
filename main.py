import tensorflow as tf
import util as ut
import gym
import numpy as np
from PIL import Image

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
IMAGE_CHANNEL = 3
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL
BATCH_SHAPE = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]

def weights_bias(shape, name):
  weight = tf.get_variable(name+"_W", shape)
  bias = tf.get_variable(name+"_B", [shape[len(shape)-1]])
  return weight, bias
def conv2d(x, W, stride):
  strides = [1, stride, stride, 1]
  return tf.nn.conv2d(x, W, strides=strides, padding="SAME")
def max_pool_2x2(x):
  strides = [1, 2, 2, 1]
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=strides, padding="SAME")

class network():
  def __init__(self):
    self.input_St_ph = tf.placeholder("float", BATCH_SHAPE, name="input_St")
    self.input_St1_ph = tf.placeholder("float", BATCH_SHAPE, name="input_St1")

    #==== Encoder ====#
    with tf.variable_scope('Encoder_ConvNet'):
      E_conv1_W, E_conv1_B = weights_bias([5, 5, 3, 32], "E_conv1")
      h_conv1 = self.feed_conv(self.input_St_ph, E_conv1_W, E_conv1_B, 1)
      E_conv2_W, E_conv2_B = weights_bias([5, 5, 32, 64], "E_conv2")
      h_conv2 = self.feed_conv(h_conv1, E_conv2_W, E_conv2_B, 1)
      E_conv3_W, E_conv3_B = weights_bias([5, 5, 64, 128], "E_conv3")
      h_conv3 = self.feed_conv(h_conv2, E_conv3_W, E_conv3_B, 1)
      h_conv3_flat = tf.contrib.layers.flatten(h_conv3)
      E_fc1_W, E_fc1_B = weights_bias([20*20*128, 1024], "E_fc1")
      h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, E_fc1_W) + E_fc1_B)
      E_fc2_W, E_fc2_B = weights_bias([1024, 512], "E_fc2")
      self.z = tf.nn.relu(tf.matmul(h_fc1, E_fc2_W) + E_fc2_B)
    #==== Decoder ====#
    with tf.variable_scope('Decoder_ConvNet'):
      D_fc1_W, D_fc2_B = weights_bias([512, IMAGE_PIXELS*4], "D_fc1")
      D_fc1 = tf.matmul(self.z, D_fc1_W) + D_fc2_B
      D_fc1 = tf.reshape(D_fc1, [-1, IMAGE_WIDTH*2, IMAGE_HEIGHT*2, IMAGE_CHANNEL])
      D_fc1 = tf.contrib.layers.batch_norm(D_fc1, epsilon=1e-5, scope='bn')
      D_fc1 = tf.nn.relu(D_fc1)

      D_conv1_W, D_conv1_B = weights_bias([3, 3, 3, 64], "D_conv1")
      h_dconv1 = conv2d(D_fc1, D_conv1_W, 1) + D_conv1_B
      h_dpool1 = tf.contrib.layers.batch_norm(h_dconv1, epsilon=1e-5, scope='bn1')
      h_dpool1 = tf.nn.relu(h_dpool1)
      h_dconv1 = tf.image.resize_images(h_dpool1, [IMAGE_WIDTH * 2, IMAGE_HEIGHT * 2])

      D_conv2_W, D_conv2_B = weights_bias([3, 3, 64, 32], "D_conv2")
      h_dconv2 = conv2d(h_dconv1, D_conv2_W, 1) + D_conv2_B
      h_dpool2 = tf.contrib.layers.batch_norm(h_dconv2, epsilon=1e-5, scope='bn2')
      h_dpool2 = tf.nn.relu(h_dpool2)
      h_dconv2 = tf.image.resize_images(h_dpool2, [IMAGE_WIDTH * 2, IMAGE_HEIGHT * 2])

      D_conv3_W, D_conv3_B = weights_bias([1, 1, 32, 3], "D_conv3")
      self.h_dconv3 = conv2d(h_dconv2, D_conv3_W, 2) + D_conv3_B


  def feed_conv(self, input, weight, bias, stride):
    temp = tf.nn.relu(conv2d(input, weight, stride) + bias)
    return max_pool_2x2(temp)
 
  def feed_forward(self, St_):
    X_ = np.reshape(St_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    readout_t = self.h_dconv3.eval(feed_dict={self.input_St_ph : X_})[0]
    return readout_t

class game():
  def __init__(self):
    self.sess = sess = tf.InteractiveSession()
    self.env = gym.make('MontezumaRevenge-v0')
    self.St = self.env.reset()
    self.net = network()
    self.sess.run(tf.global_variables_initializer())

  def take_action(self, St, At):
    readout = self.net.feed_forward(St)
    print(readout.shape)
    St1, Rt1, done, info = self.env.step(At)
    St1 = ut.preprocess(St1)
    Tr = [St, At, Rt1, St1, done]
    return Tr

  def run(self):
    St = self.St
    St = ut.preprocess(St)
    for _ in range(1):
      self.env.render()
      At = self.env.action_space.sample()
      Tr = self.take_action(St, At)

def main():
  gm = game()
  gm.run()

if __name__ == "__main__":
  main()
