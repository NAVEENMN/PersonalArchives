import tensorflow as tf
import gym
import numpy as np
from collections import deque
from PIL import Image
import random
from network import *
import glob
import os

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
IMAGE_CHANNEL = 3
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL
SHAPE_IM = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
SHAPE_RW = [None, 1]
SHAPE_AC = [None, 18]
SHAPE_SR = [None, 512]

# LOG = "/output/Tensorboard/"
MODEL_PATH = "saved_models/"
LOG = "Tensorboard"
BATCH_SIZE = 18

class network():
    def __init__(self):
        with tf.name_scope("Inputs"):
            self.input_st_ph = tf.placeholder("float", SHAPE_IM, name="input_st")
            self.input_at_ph = tf.placeholder("float", SHAPE_AC, name="input_at")
            self.input_rt1_ph = tf.placeholder("float", SHAPE_RW, name="input_rt1")
            self.input_sr_ph = tf.placeholder("float", SHAPE_SR, name="input_sr")

        # Action space
        acts = []
        for i in range(0, 18):
            act = np.zeros([1, 18])
            act[0][i] = 1
            acts.append(act)
        self.action_space = np.reshape(acts, [18, 18])

        with tf.name_scope("Phi"):
            # State Encoder Decoder
            stendec = state_encoder_decoder("state", BATCH_SIZE)
            with tf.name_scope("Phi_Encode"):
                self.phi_st = stendec.encode(self.input_st_ph)
            with tf.name_scope("Phi_Decode"):
                self.stp = stendec.decode(self.phi_st)
            with tf.name_scope("Phi_loss"):
                self.phi_st_loss = stendec.get_loss(source=self.stp,
                                                    target=self.input_st_ph)
            with tf.name_scope("Phi_train"):
                self.phi_st_train_step = stendec.train_step(self.phi_st_loss)

        with tf.name_scope("Theta"):
            # Action Encoder Decoder
            atendec = action_encode_decoder("action", BATCH_SIZE)
            with tf.name_scope("Theta_Encode"):
                self.theta_at = atendec.encode(self.input_at_ph)
            with tf.name_scope("Theta_Decode"):
                self.atp = atendec.decode(self.theta_at)
            with tf.name_scope("Theta_loss"):
                self.theta_at_loss = atendec.get_loss(source=self.atp,
                                                      target=self.input_at_ph)
            with tf.name_scope("Theta_train"):
                self.theta_at_train_step = atendec.train_step(self.theta_at_loss)

        with tf.name_scope("ETA"):
            # Reward Predictor
            rwpred = reward_predictor("reward", BATCH_SIZE)
            with tf.name_scope("Reward_predict"):
                self.eta_st = rwpred.predict_reward(phi_st=self.phi_st)
                #self.eta_sft = rwpred.predict_reward(phi_st=self.input_sr_ph, reuse=True)
            with tf.name_scope("ETA_loss"):
                self.eta_st_loss = rwpred.get_loss(source=self.eta_st,
                                                   target=self.input_rt1_ph)
            with tf.name_scope("ETA_train"):
                self.eta_st_train_step = rwpred.rw_train_step(self.eta_st_loss)

        with tf.name_scope("St_At"):
            self.psi_st_at = tf.multiply(self.phi_st, self.theta_at)

        with tf.name_scope("SR_representation"):
            # SR representation
            sr = sr_representation(name="sr", batch_size=BATCH_SIZE)
            with tf.name_scope("SR_features"):
                self.sr_feature = sr.get_sucessor_feature(phi_st=self.psi_st_at)
            with tf.name_scope("SR_loss"):
                self.sr_train_step, self.sr_loss = sr.sr_train_step(phi_st=self.phi_st,
                                                                    si_st=self.sr_feature,
                                                                    si_st1=self.input_sr_ph)

        # Q(st, at)
        with tf.name_scope("Q_predictor"):
            self.q_st_at = rwpred.predict_reward(phi_st=self.input_sr_ph, reuse=True)


        data = dict()
        data["source_image"] = self.input_st_ph
        data["sr_feature"] = stendec.decode(self.input_sr_ph, reuse=True)
        data["reconstructed_image"] = self.stp
        data["reconstruction_loss"] = self.phi_st_loss / IMAGE_PIXELS
        data["reward_pred_loss"] = self.eta_st_loss
        data["sr_loss"] = self.sr_loss
        data["at_loss"] = self.theta_at_loss
        tensorboard_summary(data)
        with tf.name_scope("Tensorboard"):
            self.merged = tf.summary.merge_all()

    def get_sr_features(self, sess, st_):
        # Action space
        sts = []
        st_ = np.reshape(st_, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
        for i in range(0, 18):
            sts.append(st_)
        state_space = np.reshape(sts, [18, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
        feed_dict = {self.input_st_ph: state_space, self.input_at_ph: self.action_space}
        sr_features = sess.run(self.sr_feature, feed_dict=feed_dict)
        feed_dict = {self.input_sr_ph: sr_features}
        q_st_at = sess.run(self.q_st_at, feed_dict=feed_dict)
        idx = np.argmax(q_st_at)
        return sr_features[idx]

    def get_action(self, sess, st_, step):
        # Action space
        sts = []
        st_ = np.reshape(st_, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
        for i in range(0, 18):
            sts.append(st_)
        state_space = np.reshape(sts, [18, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
        feed_dict = {self.input_st_ph: state_space, self.input_at_ph: self.action_space}
        sr_features = sess.run(self.sr_feature, feed_dict=feed_dict)
        feed_dict = {self.input_sr_ph: sr_features}
        q_st_at = sess.run(self.q_st_at, feed_dict=feed_dict)
        index = np.argmax(q_st_at)
        if step % 10 == 0:
            print("--")
            print(q_st_at, index)
            print("---")
        return index

    def get_batch(self, D, demo):
        minibatch = random.sample(D, BATCH_SIZE)
        shape_im = [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
        shape_ac = [BATCH_SIZE, 18]
        shape_rw = [BATCH_SIZE, 1]
        shape_sr = [BATCH_SIZE, 512]
        st_batch = np.reshape([d[0] for d in minibatch], shape_im)
        at_batch = np.reshape([d[1] for d in minibatch], shape_ac)
        rt1_batch = np.reshape([d[2] for d in minibatch], shape_rw)
        st1_batch = np.reshape([d[3] for d in minibatch], shape_im)
        sr_batch = np.reshape([d[4] for d in minibatch], shape_sr)
        b1 = [st_batch, at_batch, rt1_batch, st1_batch, sr_batch]

        demobatch = random.sample(demo, BATCH_SIZE)
        shape_im = [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
        shape_rw = [BATCH_SIZE, 1]
        str_batch = np.reshape([d[0] for d in demobatch], shape_im)
        rtr_batch = np.reshape([d[1] for d in demobatch], shape_rw)
        b2 = [str_batch, rtr_batch]
        return b1, b2

    def train(self, sess, memory, demo, writer, step):
        b1, b2 = self.get_batch(memory, demo)
        [st_batch, at_batch, rt1_batch, st1_batch, sr_batch] = b1
        [str_batch, rtr_batch] = b2

        # Train reward predictor
        #if step % 10 == 0:
        op = self.eta_st_train_step
        feed_dict = {self.input_st_ph: str_batch,
                    self.input_rt1_ph: rtr_batch}
        sess.run(op, feed_dict=feed_dict)
        feed_dict = {self.input_st_ph: st_batch,
                     self.input_rt1_ph: rt1_batch}
        sess.run(op, feed_dict=feed_dict)

        # Train action space
        op = self.theta_at_train_step
        sess.run(op, feed_dict={self.input_at_ph: self.action_space})

        # Train state space
        op = self.phi_st_train_step
        sess.run(op, feed_dict={self.input_st_ph: st_batch})

        # SR
        ops = []
        feed_dict = {self.input_st_ph: st_batch,
                     self.input_at_ph: at_batch,
                     self.input_sr_ph: sr_batch}
        ops.append(self.sr_train_step)
        sess.run(ops, feed_dict=feed_dict)

        if step % 10 == 0:
            ops = []
            feed_dict = {self.input_st_ph: st_batch,
                         self.input_at_ph: at_batch,
                         self.input_rt1_ph: rtr_batch,
                         self.input_sr_ph: sr_batch}
            ops.append(self.phi_st_loss)
            ops.append(self.theta_at_loss)
            ops.append(self.eta_st_loss)
            ops.append(self.sr_loss)
            ops.append(self.merged)
            stloss, atloss, rwloss, srloss,summary = sess.run(ops, feed_dict=feed_dict)
            writer.add_summary(summary, step)
            print(step, atloss, stloss, rwloss, srloss)


class game():
    def __init__(self):
        self.demo = deque()
        self.memory = deque()
        self.rewards = list()
        self.load_demo()
        self.train_iteration = 0
        self.exploration = 0.1
        self.init_exp = 0.5
        self.final_exp = 0.0
        self.anneal_steps = 10000
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

    def load_demo(self):
        max_rwd = len(os.listdir('play/'))
        max_rwd = float(max_rwd)
        for filename in glob.glob('play/*.jpg'):
            parts = filename.split(".")
            part = parts[0].split("frame")
            rwd = float(part[1])
            reward = (rwd / max_rwd)
            im = Image.open(filename)
            im = im.resize((160, 210), Image.ANTIALIAS)
            im = np.asarray(im)
            im = self.preprocess(im)
            print(filename, reward)
            self.demo.append([im, reward])

    def annealExploration(self, stategy='linear'):
        ratio = max((self.anneal_steps - self.train_iteration) / float(self.anneal_steps), 0)
        self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp

    def take_action(self, sess, st, step):
        action = np.zeros([1, 18])
        if random.random() < self.exploration:
            at = self.env.action_space.sample()
        else:
            at = self.net.get_action(sess, st, step)
        action[0][at] = 1
        st1, rt1, done, info = self.env.step(at)
        st1 = self.preprocess(st1)
        rt1 = 0
        if done:
            rt1 = -1
        m_s_ap = self.net.get_sr_features(sess, st1)
        Tr = [st, action, rt1, st1, m_s_ap, np.mean(self.rewards)]

        if len(self.memory) > 10000:
            self.memory.popleft()
        self.memory.append(Tr)

        return st1, done

    def run(self, mode):
        st = self.St
        st = self.preprocess(st)
        step = 1
        running_rewards = 0
        while True:
            self.env.render()
            st1, done = self.take_action(self.sess, st, step)
            # print(len(self.SR_memory), len(self.AC_memory))
            if len(self.memory) > 500 and mode == "train":
                self.annealExploration()
                self.net.train(self.sess, self.memory, self.demo, self.writer, step)
                self.train_iteration += 1
            st = st1
            if done:
                st = self.env.reset()
                st = self.preprocess(st)
                self.rewards.append(running_rewards)
                running_rewards = 0
            if len(self.rewards) >= 10:
                self.rewards.pop(0)
            step = step + 1
            running_rewards += 1
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
