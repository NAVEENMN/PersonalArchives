import tensorflow as tf
import util as ut
import gym
import numpy as np
from collections import deque
from PIL import Image
import random
from network import *

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
IMAGE_CHANNEL = 3
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL
SHAPE_IM = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
SHAPE_RW = [None, 1]
SHAPE_AC = [None, 18]

LOG = "/output/Tensorboard/"
#LOG = "Tensorboard"
BATCH_SIZE = 10

class network():
  def __init__(self):
    self.input_st_ph  = tf.placeholder("float", SHAPE_IM, name="input_st")
    self.input_at_ph  = tf.placeholder("float", SHAPE_AC, name="input_at")
    self.input_rt1_ph = tf.placeholder("float", SHAPE_RW, name="input_rt1")
    self.input_st1_ph = tf.placeholder("float", SHAPE_IM, name="input_st1")
    #self.st   = tf.image.rgb_to_grayscale(self.input_st_ph)
    #self.st_1 = tf.image.rgb_to_grayscale(self.input_st1_ph)

    StEndec = state_encoder_decoder("state", BATCH_SIZE)
    latent = StEndec.encode(self.input_st_ph)
    self.reconstruction = StEndec.decode(latent)
    self.reconstruction_loss = StEndec.get_loss(source=self.reconstruction, 
                                              target=self.input_st_ph)
    self.reconstruction_train_step = StEndec.train_step(self.reconstruction_loss)

    data = dict()
    data["source_image"] = self.input_st_ph
    data["reconstructed_image"] = self.reconstruction
    data["reconstruction_loss"] = self.reconstruction_loss
    tensorboard_summary(data)
    self.merged = tf.summary.merge_all()
 
  def feed_forward(self, st_):
    X_ = np.reshape(st_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    readout_t = self.reconstruction.eval(feed_dict={self.input_st_ph : X_})
    return readout_t

  def get_batch(self, D):
    minibatch = random.sample(D, BATCH_SIZE)
    shape_im = [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
    shape_ac = [BATCH_SIZE, 18]
    shape_rw = [BATCH_SIZE, 1]
    st_batch  = np.reshape([d[0] for d in minibatch], shape_im)
    st1_batch = np.reshape([d[3] for d in minibatch], shape_im)
    at_batch  = np.reshape([d[1] for d in minibatch], shape_ac)
    rt1_batch = np.reshape([d[2] for d in minibatch], shape_rw)
    return st_batch, at_batch, rt1_batch, st1_batch

  def train(self, sess, D, writer, step):
    st_batch, at_batch, rt1_batch, st1_batch = self.get_batch(D)

    ops = []
    feed_dict = {self.input_st_ph: st_batch}
    ops.append(self.reconstruction_train_step)
    ops.append(self.reconstruction_loss)
    _, loss = sess.run(ops, feed_dict=feed_dict)

    if step % 10 == 0:
      ops = []
      feed_dict = {self.input_st_ph: st_batch}
      ops.append(self.reconstruction_loss)
      ops.append(self.merged)
      loss, summary = sess.run(ops, feed_dict=feed_dict)
      writer.add_summary(summary, step)
      print(step, loss)

class game():
  def __init__(self):
    self.D = deque()
    self.sess = sess = tf.InteractiveSession()
    self.env = gym.make('MontezumaRevenge-v0')
    self.St = self.env.reset()
    self.net = network()
    self.sess.run(tf.global_variables_initializer())
    self.writer = tf.summary.FileWriter(LOG, sess.graph)

  def take_action(self, St):
    #readout = self.net.feed_forward(St)
    #print(readout.shape)
    actions = np.zeros([1, 18])
    At = self.env.action_space.sample()
    actions[0][At-1] = 1
    St1, Rt1, done, info = self.env.step(At)
    St1 = ut.preprocess(St1)
    Rt1 = 1
    if done:
      Rt1 = -1
      self.env.reset()
    Tr = [St, actions, Rt1, St1, done]
    return Tr

  def run(self):
    St = self.St
    St = ut.preprocess(St)
    for step in range(1000):
      #self.env.render()
      Tr = self.take_action(St)
      St1 = Tr[3]
      self.D.append(Tr)
      if len(self.D) > 100:
        self.D.popleft()
        self.net.train(self.sess, self.D, self.writer, step)
      St = St1

def main():
  gm = game()
  gm.run()

if __name__ == "__main__":
  main()
