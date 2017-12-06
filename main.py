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
SHAPE_SR = [None, 512]

#LOG = "/output/Tensorboard/"
LOG = "Tensorboard"
BATCH_SIZE = 32

class network():
  def __init__(self):
    self.input_st_ph  = tf.placeholder("float", SHAPE_IM, name="input_st")
    self.input_at_ph  = tf.placeholder("float", SHAPE_AC, name="input_at")
    self.input_rt1_ph = tf.placeholder("float", SHAPE_RW, name="input_rt1")
    self.input_st1_ph = tf.placeholder("float", SHAPE_IM, name="input_st1")
    self.input_sr_ph = tf.placeholder("float", SHAPE_SR, name="input_sr")


    with tf.device('/gpu:0'):
      # State Encoder Decoder
      StEndec = state_encoder_decoder("state", BATCH_SIZE)
      self.statelatent = StEndec.encode(self.input_st_ph)
      self.reconstruction = StEndec.decode(self.statelatent)
      self.reconstruction_loss, self.per_px_loss = StEndec.get_loss(source=self.reconstruction,
                                                                  target=self.input_st_ph)
      self.reconstruction_train_step = StEndec.train_step(self.reconstruction_loss)

      # Action Encoder Decoder
      AtEndec = action_encode_decoder("action", BATCH_SIZE)
      self.actionlatent = AtEndec.encode(self.input_at_ph)
      self.action_reconstruction = AtEndec.decode(self.actionlatent)
      self.action_reconstruction_loss = AtEndec.get_loss(source=self.action_reconstruction,
                                                         target=self.input_at_ph)
      self.at_reconstruction_train_step = AtEndec.train_step(self.action_reconstruction_loss)

    with tf.device('/gpu:1'):
      # Reward Predictor
      RwPred = reward_predictor("reward", BATCH_SIZE)
      RwPredicted = RwPred.predict_reward(state_latent=self.statelatent)
      self.RwPredictionloss = RwPred.get_loss(source=RwPredicted, target=self.input_rt1_ph)
      self.rw_train_step = RwPred.rw_train_step(self.RwPredictionloss)

      # SR representation
      self.srlatents = tf.multiply(self.statelatent, self.actionlatent, name="SROp")
      sr = sr_representation(name="sr", batch_size=BATCH_SIZE)
      self.sr_feature = sr.get_sucessor_feature(srlatent=self.srlatents)
      self.q_values = RwPred.predict_reward(state_latent=self.sr_feature, reuse=True)
      self.sr_train_step, self.sr_loss = sr.sr_train_step(statelatent=self.statelatent,
                                                          predicted=self.sr_feature,
                                                          target=self.input_sr_ph)

    data = dict()
    data["source_image"] = self.input_st_ph
    data["sr_feature"] = StEndec.decode(self.input_sr_ph, reuse=True)
    data["reconstructed_image"] = self.reconstruction
    data["reconstruction_loss"] = self.per_px_loss
    data["action_recon_loss"] = self.action_reconstruction_loss
    data["reward_pred_loss"]  = self.RwPredictionloss
    data["sr_loss"] = self.sr_loss
    tensorboard_summary(data)
    self.merged = tf.summary.merge_all()

  def get_sr_action(self, sess, st_):
    # st_ : [1, 160, 160, 3]
    S_ = np.reshape(st_, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
    acts = []
    stats = []
    for i in range(0, 18):
      act = np.zeros([1, 18])
      act[0][i] = 1
      acts.append(act)
      stats.append(S_)
    S_ = np.reshape(stats, [18, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
    A_ = np.reshape(acts, [18, 18])
    sr_features = sess.run(self.sr_feature, feed_dict={self.input_st_ph: S_, self.input_at_ph: A_})
    q_values = sess.run(self.q_values, feed_dict={self.input_st_ph: S_, self.input_at_ph: A_})
    index = np.argmax(q_values)
    return sr_features, index

  def get_batch(self, sess, D):

    minibatch = random.sample(D, BATCH_SIZE)
    shape_im = [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
    shape_ac = [BATCH_SIZE, 18]
    shape_rw = [BATCH_SIZE, 1]
    shape_sr = [BATCH_SIZE, 512]
    st_batch  = np.reshape([d[0] for d in minibatch], shape_im)
    at_batch  = np.reshape([d[1] for d in minibatch], shape_ac)
    rt1_batch = np.reshape([d[2] for d in minibatch], shape_rw)
    st1_batch = np.reshape([d[3] for d in minibatch], shape_im)
    sr_batch = np.reshape([d[4] for d in minibatch], shape_sr)
    return st_batch, at_batch, rt1_batch, st1_batch, sr_batch

  def train(self, sess, D, writer, step):
    st_batch, at_batch, rt1_batch, st1_batch, sr_batch = self.get_batch(sess, D)

    ops = []
    feed_dict = {self.input_st_ph: st_batch,
                 self.input_at_ph: at_batch,
                 self.input_rt1_ph: rt1_batch,
                 self.input_sr_ph: sr_batch}
    ops.append(self.reconstruction_train_step)
    ops.append(self.at_reconstruction_train_step)
    ops.append(self.rw_train_step)
    ops.append(self.sr_train_step)
    sess.run(ops, feed_dict=feed_dict)

    if step % 10 == 0:
      ops = []
      feed_dict = {self.input_st_ph: st_batch,
                   self.input_at_ph: at_batch,
                   self.input_rt1_ph: rt1_batch,
                   self.input_sr_ph: sr_batch}
      ops.append(self.reconstruction_loss)
      ops.append(self.action_reconstruction_loss)
      ops.append(self.RwPredictionloss)
      ops.append(self.sr_loss)
      ops.append(self.merged)
      stloss,atloss,rwloss,srloss,summary = sess.run(ops, feed_dict=feed_dict)
      writer.add_summary(summary, step)
      print(step, stloss, atloss, rwloss, srloss)

class game():
  def __init__(self):
    self.D = deque()
    self.sess = sess = tf.InteractiveSession()
    self.env = gym.make('MontezumaRevenge-v0')
    self.St = self.env.reset()
    self.net = network()
    self.sess.run(tf.global_variables_initializer())
    self.writer = tf.summary.FileWriter(LOG, sess.graph)

  def take_action(self, sess, st):
    actions = np.zeros([1, 18])
    _, at = self.net.get_sr_action(sess, st)
    actions[0][at] = 1
    st1, rt1, done, info = self.env.step(at)
    st1 = ut.preprocess(st1)
    sr, index = self.net.get_sr_action(sess, st1)
    sucessor_features = sr[index]
    rt1 = 1
    if done:
      rt1 = -1
      self.env.reset()
    Tr = [st, actions, rt1, st1, sucessor_features, done]
    return Tr

  def run(self):
    St = self.St
    St = ut.preprocess(St)
    for step in range(1000):
      #self.env.render()
      Tr = self.take_action(self.sess, St)
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
