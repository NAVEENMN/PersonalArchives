from defs import *
from models import *
from utils import *
import gym
import numpy as np
import tensorflow as tf

class network():
    def __init__(self, sess):
        self.sess = sess

        with tf.name_scope("Inputs"):
            self.input_st_ph = tf.placeholder("float", ST_SHAPES, name="input_st")

        with tf.name_scope("Phi"):
            stendec = state_encoder_decoder("Phi")

            with tf.name_scope("Phi_Encode"):
                self.phi = stendec.encode(self.input_st_ph, False)
            with tf.name_scope("Phi_Decode"):
                self.phi_ = stendec.decode(self.phi, False)
            with tf.name_scope("Phi_loss"):
                self.phi_loss = stendec.get_loss(source=self.phi_,
                                                 target=self.input_st_ph)
            with tf.name_scope("Phi_train"):
                self.phi_train_step = stendec.train_step(self.phi_loss)

        with tf.name_scope("pi"):
            pi = policy("pi")

            with tf.name_scope("Pi_sample"):
                self.pi_ex_at, self.pi_in_at = pi.sample(state_latent=self.phi)

        data = dict()
        data["source_image"] = self.input_st_ph
        data["reconstructed_image"] = self.phi_
        data["reconstruction_loss"] = self.phi_loss / IMAGE_PIXELS
        tensorboard_summary(data)
        with tf.name_scope("Tensorboard"):
            self.merged = tf.summary.merge_all()

    def get_state_latent(self, input_st):
        input_st = np.reshape(input_st, ST_SHAPE)
        latent = self.sess.run(self.phi, feed_dict={self.input_st_ph: input_st})
        return latent

    def get_st_reconstructed(self, input_st):
        input_st = np.reshape(input_st, ST_SHAPE)
        st_ = self.sess.run(self.phi_, feed_dict={self.input_st_ph: input_st})
        return st_

    def get_pi_sample(self, input_st):
        input_st = np.reshape(input_st, ST_SHAPE)
        feed_dict={self.input_st_ph: input_st}
        ext, inh = self.sess.run([self.pi_ex_at, self.pi_in_at], feed_dict=feed_dict)
        net_at = np.add(ext, inh) # Actions are bound 0 to 1
        return net_at

    def train_step(self, memory, writer, step):
        st, at, r, st2, done = memory.sample_batch(BATCH_SIZE)

        # Train State Space
        feed_dict={self.input_st_ph: st}
        self.sess.run(self.phi_train_step, feed_dict=feed_dict)

        # losses
        feed_dict={self.input_st_ph: st}
        ops = []
        ops.append(self.phi_loss)
        ops.append(self.merged)
        stloss, summary = self.sess.run(ops, feed_dict=feed_dict)
        writer.add_summary(summary, step)
        print(step, stloss)

class game():
    def __init__(self, sess):
        self.sess = sess
        self.env = gym.make('CarRacing-v0')
        self.net = network(self.sess)
        self.saver = tf.train.Saver()
        self.memory = ReplayBuffer(10000)
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(LOG, sess.graph)
        checkpoint = tf.train.get_checkpoint_state(MODEL_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
            
    def run(self, global_step):
        done = False
        running_step = global_step
        st = np.asarray(self.env.reset())
        st = np.reshape(st, ST_SHAPE)
        st_memory = ReplayBuffer(MAX_EP_STEP+1)
        while not done:
            self.env.render()
            at = self.net.get_pi_sample(st)
            print("step {}, at {}", running_step, at)
            st1, r, done, info = self.env.step(at[0])
            st1 = np.reshape(st1, ST_SHAPE)
            at = np.reshape(at, AT_SHAPE)
            tr = (st, at, r, st1, done)
            st_memory.add(tr)
            running_step += 1

            # each trajectory at max = MAX_EP_STEP
            if (running_step % MAX_EP_STEP == 0) or done:
                # flush buffer to memory
                self.memory.extend(st_memory)
                st_memory.flush()
                
            if (running_step % 1000 == 0):
                self.net.train_step(self.memory, self.writer, running_step)
            st = st1
        return running_step

def main():
    sess = tf.Session()
    gm = game(sess)
    global_step = 0
    for _ in range(0, 10):
        global_step = gm.run(global_step)

if __name__ == "__main__":
    main()
