import tensorflow as tf
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
    self.input_pib_ph = tf.placeholder("float", SHAPE_RW, name="input_pib")
    # Action space
    acts = []
    for i in range(0, 18):
      act = np.zeros([1, 18])
      act[0][i] = 1
      acts.append(act)
    self.action_space = np.reshape(acts, [18, 18])

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

    bhev = Behavior("controller",BATCH_SIZE)
    self.bhev_predict = bhev.predict_confidence(st_latent=self.statelatent)

    self.habitual_loss, self.habitual_train_step = bhev.habit_loss(confidence=self.bhev_predict,
                                                                   reward=self.input_rt1_ph)
    '''
    self.restimation_loss= bhev.reestimation_loss(actions_pib=self.actor_pred,
                                                  actions_ve=self.input_at_ph)
    '''
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

    actor = Actor("Actor", BATCH_SIZE)
    self.actor_pred = actor.predict(st_latent=self.statelatent)
    self.ac_loss = actor.get_loss(self.actor_pred, self.input_at_ph)
    self.ac_train_step = actor.train_step(self.ac_loss)

    data = dict()
    data["source_image"] = self.input_st_ph
    data["sr_feature"] = StEndec.decode(self.input_sr_ph, reuse=True)
    data["reconstructed_image"] = self.reconstruction
    data["reconstruction_loss"] = self.per_px_loss
    data["action_recon_loss"] = self.action_reconstruction_loss
    data["reward_pred_loss"]  = self.RwPredictionloss
    data["sr_loss"] = self.sr_loss
    data["hb_loss"] = self.habitual_loss
    data["ac_loss"] = self.ac_loss
    tensorboard_summary(data)
    self.merged = tf.summary.merge_all()

  def get_sr_features(self, sess, st_):
    st_ = np.reshape(st_, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
    states = []
    for i in range(0, 18):
      states.append(st_)
    st_ = np.reshape(states, [18, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
    at_ = self.action_space
    sr_features = sess.run(self.sr_feature, feed_dict={self.input_st_ph: st_, self.input_at_ph: at_})
    return sr_features, st_, at_


  def get_action(self, sess, st_):
    st_ = np.reshape(st_, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
    #selects behaviour policy or recompute policy
    pi_choice = sess.run(self.bhev_predict, feed_dict={self.input_st_ph: st_})[0]
    if np.random.random_sample() > 0.8:
      pi_choice = random.randint(0, 1)
    if pi_choice > 0.8:
      sr_features, st_, at_ = self.get_sr_features(sess, st_)
      q_values = sess.run(self.q_values, feed_dict={self.input_st_ph: st_, self.input_at_ph: at_})
      index = np.argmax(q_values)
    else:
      action = sess.run(self.actor_pred, feed_dict={self.input_st_ph: st_})
      index = np.argmax(action)
    return pi_choice, index

  def get_batch(self, D):
    minibatch = random.sample(D, BATCH_SIZE)
    shape_im = [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
    shape_ac = [BATCH_SIZE, 18]
    shape_rw = [BATCH_SIZE, 1]
    shape_sr = [BATCH_SIZE, 512]
    pib_batch = np.reshape([d[0] for d in minibatch], shape_rw)
    st_batch  = np.reshape([d[1] for d in minibatch], shape_im)
    at_batch  = np.reshape([d[2] for d in minibatch], shape_ac)
    rt1_batch = np.reshape([d[3] for d in minibatch], shape_rw)
    st1_batch = np.reshape([d[4] for d in minibatch], shape_im)
    sr_batch = np.reshape([d[5] for d in minibatch], shape_sr)
    return pib_batch, st_batch, at_batch, rt1_batch, st1_batch, sr_batch

  def train(self, sess, AC_mem, SR_mem, writer, step):
    sr_pib_batch, sr_st_batch, sr_at_batch, sr_rt1_batch, sr_st1_batch, sr_sr_batch = self.get_batch(SR_mem)
    ac_pib_batch, ac_st_batch, ac_at_batch, ac_rt1_batch, ac_st1_batch, _ = self.get_batch(AC_mem)

    # SR
    ops = []
    feed_dict = {self.input_st_ph: sr_st_batch,
                 self.input_at_ph: sr_at_batch,
                 self.input_rt1_ph: sr_rt1_batch,
                 self.input_sr_ph: sr_sr_batch,
                 self.input_pib_ph: sr_pib_batch}
    ops.append(self.reconstruction_train_step)
    ops.append(self.at_reconstruction_train_step)
    ops.append(self.rw_train_step)
    ops.append(self.sr_train_step)
    ops.append(self.habitual_train_step)
    ops.append(self.ac_train_step)
    #ops.append(self.restim_train_step)
    sess.run(ops, feed_dict=feed_dict)
    if step % 10 == 0:
      ops = []
      ops.append(self.reconstruction_loss)
      ops.append(self.action_reconstruction_loss)
      ops.append(self.RwPredictionloss)
      ops.append(self.sr_loss)
      ops.append(self.habitual_loss)
      #ops.append(self.restimation_loss)
      ops.append(self.ac_loss)
      ops.append(self.merged)
      stloss,atloss,rwloss,srloss,hloss, acloss, summary = sess.run(ops, feed_dict=feed_dict)
      writer.add_summary(summary, step)
      print(step, stloss, atloss, rwloss, srloss, hloss, acloss)

    # AC
    ops = []
    feed_dict = {self.input_st_ph: ac_st_batch,
                 self.input_at_ph: ac_at_batch,
                 self.input_rt1_ph: ac_rt1_batch}
    ops.append(self.reconstruction_train_step)
    ops.append(self.at_reconstruction_train_step)
    ops.append(self.habitual_train_step)
    sess.run(ops, feed_dict=feed_dict)


class game():
  def __init__(self):
    self.SR_memory = deque()
    self.AC_memory = deque()
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
    pi_b, at = self.net.get_action(sess, st)
    action[0][at] = 1
    st1, rt1, done, info = self.env.step(at)
    st1 = self.preprocess(st1)
    rt1 = 1
    if done:
      rt1 = 0
    if pi_b > 0.8:
      M_S_Ap, _, _ = self.net.get_sr_features(sess, st1)
      sucessor_features = M_S_Ap[at]
      Tr = [pi_b, st, action, rt1, st1, sucessor_features]
      if len(self.SR_memory) > 100:
        self.SR_memory.popleft()
      self.SR_memory.append(Tr)
    else:
      Tr =[pi_b, st, action, rt1, st1, np.zeros([1, 512])]
      if len(self.AC_memory) > 100:
        self.AC_memory.popleft()
      self.AC_memory.append(Tr)
    return st1, done

  def run(self, mode):
    st = self.St
    st = self.preprocess(st)
    step = 0
    while True:
      #self.env.render()
      st1, done = self.take_action(self.sess, st)
      #print(len(self.SR_memory), len(self.AC_memory))
      if len(self.SR_memory) > 100 and len(self.AC_memory)>100:
        if mode == "train":
          self.net.train(self.sess, self.AC_memory, self.SR_memory, self.writer, step)
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
