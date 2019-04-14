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

        with tf.name_scope("pi"):
            pi = policy("pi")

            with tf.name_scope("Pi_sample"):
                self.pi_ex_at, self.pi_in_at = pi.sample(state_latent=self.phi)

    def get_state_latent(self, input_st):
        input_st = np.reshape(input_st, ST_SHAPE)
        latent = self.sess.run(self.phi, feed_dict={self.input_st_ph: input_st})
        return latent

    def get_pi_sample(self, input_st):
        input_st = np.reshape(input_st, ST_SHAPE)
        feed_dict={self.input_st_ph: input_st}
        ext, inh = self.sess.run([self.pi_ex_at, self.pi_in_at], feed_dict=feed_dict)
        net_at = np.add(ext, inh) # Actions are bound 0 to 1
        return net_at

class game():
    def __init__(self, sess):
        self.sess = sess
        self.env = gym.make('CarRacing-v0')
        self.net = network(self.sess)
        self.memory = ReplayBuffer(10000)
        self.sess.run(tf.global_variables_initializer())
            
    def run(self):
        done = False
        running_step = 0
        st = np.asarray(self.env.reset())
        st_memory = ReplayBuffer(MAX_EP_STEP+1)
        while not done:
            self.env.render()
            at = self.net.get_pi_sample(st)
            st1, r, done, info = self.env.step(at[0])
            tr = (st, at, r, st1, done)
            st_memory.add(tr)
            print(r, done)
            running_step += 1
            if (running_step % MAX_EP_STEP == 0) or done:
                # flush buffer to memory
                print(st_memory.size(), self.memory.size())
                self.memory.extend(st_memory)
                st_memory.flush()
                print(st_memory.size(), self.memory.size())
            st = st1
def main():
    sess = tf.Session()
    gm = game(sess)
    gm.run()

if __name__ == "__main__":
    main()
