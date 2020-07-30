import tensorflow as tf
import gym
import numpy as np
from collections import deque
from PIL import Image
import random
from network1 import *

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
IMAGE_CHANNEL = 3
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL
SHAPE_IM = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
SHAPE_RW = [None, 1]
SHAPE_AC = [None, 18]
SHAPE_SR = [None, 512]

#LOG = "/output/Tensorboard/"
MODEL_PATH = "saved_models/"
LOG = "Tensorboard"
BATCH_SIZE = 32

class network():
  def __init__(self):
    self.input_st_ph  = tf.placeholder("float", SHAPE_IM, name="input_st")
    self.input_at_ph  = tf.placeholder("float", SHAPE_AC, name="input_at")
    self.input_rt1_ph = tf.placeholder("float", SHAPE_RW, name="input_rt1")
    self.input_st1_ph = tf.placeholder("float", SHAPE_IM, name="input_st1")
    self.input_sr_ph = tf.placeholder("float", SHAPE_SR, name="input_sr")

    # State Encoder Decoder
    StEndec = state_encoder_decoder("state", BATCH_SIZE)
    self.pi_st = StEndec.encode(self.input_st_ph)
    self._st = StEndec.decode(self.pi_st)
    self.pi_st_loss = StEndec.get_loss(source=self._st,
                                       target=self.input_st_ph)
    self.pi_st_train_step = StEndec.train_step(self.pi_st_loss)

    # SR representation
    sr = sr_representation(name="sr", batch_size=BATCH_SIZE)
    self.sr_feature = sr.get_sucessor_feature(pi_st=self.pi_st)
    self.sr_train_step, self.sr_loss = sr.sr_train_step(phi_st=self.phi_st,
                                                        si_st=self.sr_feature,
                                                        si_st1=self.input_sr_ph)
 
    # Reward Predictor
    RwPred = reward_predictor("reward", BATCH_SIZE)
    RwPredicted = RwPred.predict_reward(pi_st=self.pi_st)
    self.RwPredictionloss = RwPred.get_loss(source=RwPredicted, 
                                            target=self.input_rt1_ph)
    self.rw_train_step = RwPred.rw_train_step(self.RwPredictionloss)

    # V(st)
    self.v_st = tf.reduce_sum(tf.multiply(self.sr_feature, RwPredicted))

    # Actor
    actor = Actor("Actor", BATCH_SIZE)
    self.actor_pred = actor.predict(st_latent=self.pi_st)
    self.ac_loss = actor.get_loss(pi_st=self.pi_st, 
                                  rt_st=self.input_rt1_ph,
                                  v_st=self.v_st)
    self.ac_train_step = actor.train_step(self.ac_loss)

    data = dict()
    data["source_image"] = self.input_st_ph
    data["sr_feature"] = StEndec.decode(self.input_sr_ph, reuse=True)
    data["reconstructed_image"] = self._st
    data["reconstruction_loss"] = self.pi_st_loss/IMAGE_PIXELS
    data["reward_pred_loss"]  = self.RwPredictionloss
    data["sr_loss"] = self.sr_loss
    data["ac_loss"] = self.ac_loss
    tensorboard_summary(data)
    self.merged = tf.summary.merge_all()

  def get_sr_features(self, sess, st_):
    st_ = np.reshape(st_, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
    sr_features = sess.run(self.sr_feature, feed_dict={self.input_st_ph: st_})
    return sr_features


  def get_action(self, sess, st_):
    st_ = np.reshape(st_, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
    sr_features = self.get_sr_features(sess, st_)
    pi_st = sess.run(self.actor_pred, feed_dict={self.input_st_ph: st_})
    index = np.argmax(pi_st)
    return index

  def get_batch(self, D):
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

  def train(self, sess, memory, writer, step):
    st_batch, at_batch, rt1_batch, st1_batch, sr_batch = self.get_batch(memory)

    # SR
    ops = []
    feed_dict = {self.input_st_ph: st_batch,
                 self.input_at_ph: at_batch,
                 self.input_rt1_ph: rt1_batch,
                 self.input_sr_ph: sr_batch}
    ops.append(self.pi_st_train_step)
    ops.append(self.rw_train_step)
    ops.append(self.sr_train_step)
    ops.append(self.ac_train_step)
    sess.run(ops, feed_dict=feed_dict)
    if step % 10 == 0:
      ops = []
      ops.append(self.pi_st_loss)
      ops.append(self.RwPredictionloss)
      ops.append(self.sr_loss)
      ops.append(self.ac_loss)
      ops.append(self.merged)
      stloss,rwloss,srloss,acloss,summary = sess.run(ops, feed_dict=feed_dict)
      writer.add_summary(summary, step)
      print(step, stloss, rwloss, srloss, acloss)

class game():
  def __init__(self):
    self.memory = deque()
    self.sess = sess = tf.InteractiveSession()
    self.env = gym.make('MontezumaRevenge-v0')
    self.St = self.env.reset()
    self.net = network()
    self.saver = tf.train.Saver()
    self.sess.run(tf.global_variables_initializer())
    self.writer = tf.summary.FileWriter(LOG, sess.graph)
    checkpoint = tf.train.get_checkpoint_state(MODEL_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
      self.saver.restore(sess, checkpoint.model_checkpoint_path)
      print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
      print("Could not find old network weights")

  def preprocess(self, img):
    img = img[20:]
    img = img[:160]
    img = np.reshape(img, [1, 160, 160, 3])
    img = img / 255.0  # (img/127.5)-1.0
    return img

  def take_action(self, sess, st):
    action = np.zeros([1, 18])
    at = self.net.get_action(sess, st)
    action[0][at] = 1
    st1, rt1, done, info = self.env.step(at)
    st1 = self.preprocess(st1)
    rt1 = 1
    if done:
      rt1 = 0
    M_S_Ap = self.net.get_sr_features(sess, st1)
    sucessor_features = M_S_Ap
    Tr = [st, action, rt1, st1, sucessor_features]
    if len(self.memory) > 100:
      self.memory.popleft()
    self.memory.append(Tr)
    return st1, done

  def run(self, mode):
    st = self.St
    st = self.preprocess(st)
    step = 0
    while True:
      #self.env.render()
      st1, done = self.take_action(self.sess, st)
      #print(len(self.SR_memory), len(self.AC_memory))
      if len(self.memory) > 100 and mode == "train":
        self.net.train(self.sess, self.memory, self.writer, step)
      st = st1
      if done:
        st = self.env.reset()
        st = self.preprocess(st)
      step = step + 1
      if mode == "train":
        if step % 10000 == 0:
          save_path = self.saver.save(self.sess, MODEL_PATH + "pretrained.ckpt", global_step=step)
          print("saved to %s" % save_path)

def main():
  modes = ["train", "test"]
  mode = modes[0]
  gm = game()
  gm.run(mode)

if __name__ == "__main__":
  main()
