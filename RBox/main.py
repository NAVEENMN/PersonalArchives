import gym
import numpy as np
from network import *
import tensorflow as tf
from collections import deque

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
IMAGE_CHANNEL = 3
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL
SHAPE_IM = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
SHAPE_RW = [None, 1]
SHAPE_AC = [None, 18]
SHAPE_SR = [None, 512]

BATCH_SIZE = 50
LOG = "Tensorboard"

class Network():
  def __init__(self):
    self.input_st_ph  = tf.placeholder("float", SHAPE_IM, name="input_st")
    tboard_data = dict()

    # State Encoder Decoder
    phi = state_encoder_decoder("stendec", BATCH_SIZE)
    self.phi_st = phi.encode(self.input_st_ph)
    self.stp = phi.decode(self.phi_st)
    self.phi_loss = phi.get_loss(source=self.stp, target=self.input_st_ph)
    self.phi_train = phi.train_step(self.phi_loss)
    tboard_data["source_image"] = self.input_st_ph
    tboard_data["recon_image"] = self.stp
    tboard_data["phi_loss"] = self.phi_loss/IMAGE_PIXELS

    # Tensorboard logging
    tensorboard_summary(tboard_data)
    self.merged = tf.summary.merge_all()

  def test(self, sess, x):
    x = np.reshape(x, [1, 96, 96, 3])
    op = self.phi_loss
    feed_dict = {self.input_st_ph: x}
    result = sess.run(op, feed_dict={self.input_st_ph: x})
    print(result.shape)

  def train(self, sess, memory, writer, step):
    st_batch, at_batch, rt1_batch, st1_batch = get_batch(memory)

    # Train Phi
    feed_dict = {self.input_st_ph: st_batch}
    ops = []
    ops.append(self.phi_train)
    sess.run(ops, feed_dict=feed_dict)

    # Log to Tensorboard
    if step % 10 == 0:
      feed_dict = {self.input_st_ph: st_batch}
      ops = []
      ops.append(self.phi_loss)
      ops.append(self.merged)
      phi_loss,_=sess.run(ops, feed_dict=feed_dict)
      print(step, phi_loss)


class game():
  def __init__(self):
    self.memory = deque()
    self.env = gym.make('CarRacing-v0')
    self.st = self.env.reset()
    self.net = Network()
    self.sess  = tf.InteractiveSession()
    self.saver = tf.train.Saver()
    self.sess.run(tf.global_variables_initializer())
    self.writer = tf.summary.FileWriter(LOG, self.sess.graph)
  
  def take_action(self, sess, st):
    at = self.env.action_space.sample()
    st1, rt1, done, info = self.env.step(at)
    st1 = preprocess(st1)
    rt1 = 1
    if done:
      rt1 = -1
    tr = [st, at, rt1, st1]
    self.memory.append(tr)
    if len(self.memory) > 100:
      self.memory.popleft()
    return st1, done

  def run(self):
    st = preprocess(self.st)
    step = 1
    while True:
      #self.env.render()
      st1, done = self.take_action(self.sess, st)
      st = st1
      if done:
        st = preprocess(self.env.reset())
      if step % 100 == 0:
        self.net.train(self.sess, self.memory, self.writer, step)
        break
      step = step + 1

def main():
  gm = game()
  gm.run()

if __name__ == "__main__":
  main()
